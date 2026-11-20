"""Logistic and marginal-likelihood binomial random-intercept GLMM baselines."""
from collections import Counter

import numpy as np
from numpy.polynomial.hermite import hermgauss
from scipy.optimize import minimize
from scipy.special import expit, logsumexp
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier

from .data import NONE, slot_key, targets
from .sampling import TrainingSpace, resample
from .representations import RepresentationSpace


def flatten(examples, builder):
    rows, labels, groups, tasks = [], [], [], []
    for ex in examples:
        feature_rows = builder.rows(ex)
        rows.extend(feature_rows)
        labels.extend(targets(ex) if 'word_labels' in ex else ['word:O']*len(ex['words']) + ['gap:'+NONE]*(len(ex['words'])+1))
        for row in feature_rows:
            groups.append(row['word0'])
            tasks.append(row['task'])
    return rows, np.asarray(labels), np.asarray(groups), tasks


def restrict_predictions(probabilities, classes, tasks):
    classes = np.asarray(classes)
    result = []
    for row, task in zip(probabilities, tasks):
        allowed = np.array([c.startswith(task+':') for c in classes])
        if not allowed.any():
            result.append('word:O' if task == 'word' else 'gap:'+NONE)
        else:
            result.append(str(classes[np.where(allowed, row, -np.inf).argmax()]))
    return result


class RandomInterceptLogit:
    """Binomial GLMM with Gaussian word intercepts integrated by GH quadrature.

    Maximizes approximate marginal likelihood (12 Gauss-Hermite nodes), rather
    than treating word identity as an ordinary fixed-effect feature. Unseen
    words get the zero population intercept. This is not the paper's lme4 fit.
    """
    def fit(self, x, y, groups):
        self.groups_, codes = np.unique(groups, return_inverse=True)
        nodes, weights = hermgauss(12)
        logweights = np.log(weights) - .5*np.log(np.pi)
        design = np.column_stack([np.ones(len(x)), x])
        n_fixed = design.shape[1]

        def objective(params, posterior=False):
            beta, tau = params[:-1], np.exp(params[-1])
            random = np.sqrt(2)*tau*nodes
            eta = design @ beta[:, None] + random[None, :]
            ll = y[:, None]*eta - np.logaddexp(0, eta)
            grouped = np.zeros((len(self.groups_), len(nodes)))
            np.add.at(grouped, codes, ll)
            joint = grouped + logweights
            logz = logsumexp(joint, axis=1)
            post = np.exp(joint-logz[:, None])
            if posterior:
                return post @ random
            residual = (y[:, None]-expit(eta))*post[codes]
            grad_beta = design.T @ residual.sum(axis=1) - .01*beta
            grad_scale = (residual*random).sum() - params[-1]/4
            loss = -logz.sum() + .005*np.dot(beta, beta) + params[-1]**2/8
            return loss, -np.r_[grad_beta, grad_scale]

        initial = np.zeros(n_fixed+1)
        result = minimize(objective, initial, jac=True, method='L-BFGS-B',
                          bounds=[(None, None)]*n_fixed + [(-5, 3)],
                          options={'maxiter': 750, 'maxls': 50})
        self.coef_ = result.x[:-1]
        self.intercepts_ = dict(zip(self.groups_, objective(result.x, posterior=True)))
        self.converged_ = bool(result.success)
        self.optimization_message_ = str(result.message)
        self.iterations_ = int(result.nit)
        return self

    def predict_proba(self, x, groups):
        intercept = np.array([self.intercepts_.get(g, 0.) for g in groups])
        return expit(self.coef_[0] + x @ self.coef_[1:] + intercept)


class ClassicalTagger:
    def __init__(self, builder, model='multinomial', balancing='none', seed=42,
                 representation='tfidf', vectors=None, encoder='bert-base-uncased',
                 cache_dir='.cache/hf/hub', local_only=True, batch_size=32, device='cpu'):
        self.builder, self.model, self.balancing, self.seed = builder, model, balancing, seed
        self.representation = representation
        self.representation_args = dict(kind=representation, vectors=vectors, encoder=encoder,
                                        cache_dir=cache_dir, local_only=local_only,
                                        batch_size=batch_size, device=device)

    def fit(self, examples):
        self.builder.fit(examples)
        rows, y, groups, _ = flatten(examples, self.builder)
        if self.model == 'glmm':
            # Numeric features as fixed effects; context word is a random effect.
            rows = [{k: v for k, v in row.items() if isinstance(v, (int, float))} for row in rows]
            if not any(rows):
                rows = [{'intercept_only': 1.} for _ in rows]
            self.vectorizer = DictVectorizer()
        else:
            self.vectorizer = RepresentationSpace(**self.representation_args)
        x = self.vectorizer.fit(rows).transform(rows)
        # TF-IDF/embeddings are already normalized. Rescaling every sparse
        # lexical coordinate by inverse frequency destabilizes the SVM.
        self.scaler = StandardScaler(with_mean=False, with_std=self.model == 'glmm').fit(x)
        x = self.scaler.transform(x)
        self.space = None
        if self.balancing != 'none' or self.model in ['glmm', 'mlp']:
            self.space = TrainingSpace().fit(x)
            x = self.space.transform(x)
        x, y, origins, self.sampling_report = resample(x, y, self.balancing, self.seed)
        self.classes_ = np.unique(y)
        self.sampling_report['representation'] = 'numeric' if self.model == 'glmm' else self.representation
        if self.model != 'glmm' and self.representation in ['contextual', 'sentence']:
            self.sampling_report['embedding_model'] = self.representation_args['encoder']
        if self.model == 'glmm':
            raw_counts = Counter(groups)
            self.known_words = {g for g, count in raw_counts.items() if count >= 30}
            groups = np.array([g if g in self.known_words else '<rare>' for g in groups])[origins]
            self.classifiers = [RandomInterceptLogit().fit(x, (y == c).astype(float), groups) for c in self.classes_]
            self.sampling_report['glmm_converged'] = all(c.converged_ for c in self.classifiers)
            self.sampling_report['glmm_diagnostics'] = [
                {'class': str(label), 'converged': classifier.converged_,
                 'iterations': classifier.iterations_, 'message': classifier.optimization_message_}
                for label, classifier in zip(self.classes_, self.classifiers)]
        else:
            base = LogisticRegression(max_iter=500, random_state=self.seed)
            self.classifier = (OneVsRestClassifier(base) if self.model == 'logistic' else base)
            if self.model == 'svm':
                self.classifier = LinearSVC(random_state=self.seed, max_iter=5000, C=.1, tol=1e-3)
            elif self.model == 'mlp':
                self.classifier = MLPClassifier(hidden_layer_sizes=(128,), max_iter=500,
                                                random_state=self.seed)
            if len(self.classes_) == 1:
                self.classifier = DummyClassifier(strategy='most_frequent')
            self.classifier.fit(x, y)
            self.classes_ = self.classifier.classes_
            iterations = getattr(self.classifier, 'n_iter_', None)
            if iterations is not None:
                self.sampling_report['iterations'] = np.asarray(iterations).tolist()
                self.sampling_report['iteration_limit_reached'] = bool(np.asarray(iterations).max() >= self.classifier.max_iter)
        return self

    def predict(self, examples):
        rows, _, groups, tasks = flatten(examples, self.builder)
        if self.model == 'glmm':
            rows = [{k: v for k, v in row.items() if isinstance(v, (int, float))} or {'intercept_only': 1.}
                    for row in rows]
        x = self.scaler.transform(self.vectorizer.transform(rows))
        if self.space:
            x = self.space.transform(x)
        if self.model == 'glmm':
            groups = [g if g in self.known_words else '<rare>' for g in groups]
            prob = np.column_stack([c.predict_proba(x, groups) for c in self.classifiers])
        else:
            if hasattr(self.classifier, 'predict_proba'):
                prob = self.classifier.predict_proba(x)
            else:
                prob = self.classifier.decision_function(x)
                if prob.ndim == 1:
                    prob = np.column_stack([-prob, prob])
        labels = restrict_predictions(prob, self.classes_, tasks)
        output, cursor = [], 0
        for ex in examples:
            size = 2*len(ex['words'])+1
            output.append(labels[cursor:cursor+size])
            cursor += size
        return output


def rule_predictions(examples):
    """Tag visible filled pauses and first copies; never insert unseen words."""
    output = []
    for ex in examples:
        words = [w['text'] for w in ex['words']]
        labels = ['O'] * len(words)
        for i, word in enumerate(words):
            if word in {'uh', 'um', 'ah', 'hm', 'hmm', 'uhh', 'umm'}:
                labels[i] = 'B-FILLER_HESITATION'
            elif i+1 < len(words) and word == words[i+1]:
                labels[i] = 'B-REPETITION'
        output.append(['word:'+l for l in labels] + ['gap:'+NONE]*(len(words)+1))
    return output
