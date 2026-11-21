"""Batched variational Gaussian-random-intercept logistic mixed models.

The likelihood is integrated under a factorized Gaussian variational posterior
using Gauss-Hermite quadrature. This is a Bayesian approximation, not lme4 MLE
or the small restoration runner's direct marginal-likelihood optimizer.
"""
import math

import numpy as np
import torch
from numpy.polynomial.hermite import hermgauss
from torch import nn
from torch.nn import functional as F

from .embedding_classification import WordContextClassifier


def group_mapping(contexts, train, fold_vocab, min_count=30):
    counts = np.zeros(len(fold_vocab.words),dtype=np.int64)
    for start in range(0,len(train),65536):
        ids = fold_vocab.transform(contexts[train[start:start+65536],1])
        counts += np.bincount(ids,minlength=len(counts))
    mapping = np.zeros(len(counts),dtype=np.int64)
    mapping[(counts>0)&(counts<min_count)] = 1  # A shared rare-word group.
    common = np.flatnonzero(counts>=min_count)
    mapping[common] = np.arange(2,len(common)+2)
    mapping[0] = 0  # Unseen context words use the population prior at inference.
    return mapping


class VariationalWordGLMM(WordContextClassifier):
    def __init__(self, weights, classes, mapping, quadrature=12):
        super().__init__(weights,classes,frozen=True,classifier='logistic')
        self.classifier_kind = 'glmm'
        self.register_buffer('group_mapping',torch.as_tensor(mapping,dtype=torch.long))
        groups = int(np.max(mapping))+1
        self.random_mean = nn.Parameter(torch.zeros(groups,classes))
        self.random_log_std = nn.Parameter(torch.full((groups,classes),math.log(.5)))
        self.prior_log_std = nn.Parameter(torch.full((classes,),math.log(.5)))
        nodes,weights = hermgauss(quadrature)
        self.register_buffer('nodes',torch.tensor(np.sqrt(2)*nodes,dtype=torch.float32))
        self.register_buffer('node_weights',torch.tensor(weights/np.sqrt(np.pi),dtype=torch.float32))

    def additional_parameters(self):
        return [self.random_mean,self.random_log_std,self.prior_log_std]

    def _quadrature_logits(self, ids):
        fixed = self.head(self.encoder(ids).flatten(1))
        groups = self.group_mapping[ids[:,1]]
        means = self.random_mean[groups]
        log_std = self.random_log_std[groups].clamp(-8,3)
        unseen = (groups==0).unsqueeze(-1)
        means = torch.where(unseen,torch.zeros_like(means),means)
        log_std = torch.where(unseen,self.prior_log_std.clamp(-5,2),log_std)
        return fixed.unsqueeze(-1)+means.unsqueeze(-1)+log_std.exp().unsqueeze(-1)*self.nodes

    def training_losses(self, ids, y):
        logits = self._quadrature_logits(ids)
        target = F.one_hot(y,num_classes=logits.shape[1]).to(logits.dtype)
        loss = F.binary_cross_entropy_with_logits(logits,target.unsqueeze(-1).expand_as(logits),reduction='none')
        return (loss*self.node_weights).sum(-1).mean(-1)

    def regularization(self, rows):
        # KL[q(b_g)||N(0,tau^2)] summed once over observed groups, not once per row.
        mean = self.random_mean[1:]
        log_std = self.random_log_std[1:].clamp(-8,3)
        log_prior = self.prior_log_std.clamp(-5,2)
        kl = (log_prior-log_std+.5*((2*log_std).exp()+mean.square())/((2*log_prior).exp())-.5).sum()
        # Half-normal population SD prior (scale 1); constants are irrelevant.
        prior = .5*(2*log_prior).exp().sum()
        return (kl+prior)/(rows*self.head[-1].out_features)

    def forward(self, ids):
        probabilities = (self._quadrature_logits(ids).sigmoid()*self.node_weights).sum(-1)
        return torch.logit(probabilities.clamp(1e-6,1-1e-6))

    def diagnostics(self):
        return {'estimator':'factorized Gaussian variational GLMM; expected-log-likelihood quadrature',
                'random_effect':'preceding fluent context word; rare words pooled from training counts',
                'random_groups':len(self.random_mean)-1,'quadrature_points':len(self.nodes),
                'population_std':self.prior_log_std.detach().clamp(-5,2).exp().cpu().tolist(),
                'posterior_mean_abs_max':float(self.random_mean.detach().abs().max().cpu()),
                'frozen_fixed_effect_features':True}
