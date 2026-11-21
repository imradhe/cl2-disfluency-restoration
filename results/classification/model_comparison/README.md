# Comparison across modelling techniques

All **3,085,832 lexical examples** from **2,438 conversations** are evaluated in the same ten
conversation-separated folds. Each technique uses the same 50-dimensional pretrained GloVe
initialization and four positional context words (200 input coordinates). Target words, timings,
phones and audio are excluded. Candidate contexts are constructed from the existing annotations;
these are contextual classification results, not end-to-end restoration.

All models use training-only inverse-frequency sample weights, two complete optimization epochs,
batches of 8,192 and AdamW with learning rate 0.001 and weight decay 0.01. Linear heads have no
hidden layer; MLP has one 64-unit ReLU layer and 0.1 dropout. This is a **fixed-budget comparison**,
not a claim that every optimizer has converged or that hyperparameters are optimal. No outer-test
fold is used for epoch/hyperparameter selection.

One-vs-rest logistic uses independent sigmoid binary cross-entropy; multinomial logistic uses
joint softmax cross-entropy; linear SVM uses one-vs-rest hinge loss. MLP uses softmax cross-entropy.
All are trained with minibatches in PyTorch. The SVM is a stochastic hinge-loss model, not a liblinear
fit. Class weighting applies to training rows for each model; evaluation retains natural prevalence.

Percentages below are means across ten held-out folds. Binary accuracy is evaluated on the same
fixed equal-class held-out subsets. Binary positive F1 and all multiclass metrics use the complete
natural-prevalence test data. Standard deviations, precision/recall and counts are retained in CSV.

## Frozen GloVe: classifier comparison

| Technique | Binary balanced accuracy | Binary positive F1 | Ten-class accuracy | Ten-class macro F1 |
|---|---:|---:|---:|---:|
| One-vs-rest logistic | 69.39% | 41.84% | 26.44% | 15.36% |
| Multinomial logistic | 69.80% | 42.04% | 28.35% | 15.20% |
| Linear SVM (hinge) | 68.89% | 43.37% | 22.33% | 12.74% |
| MLP | 73.20% | 47.06% | 32.44% | 16.99% |
| Variational GLMM | 70.93% | 43.92% | 28.51% | 15.84% |

Ten-class F1 by label:

| Class | One-vs-rest logistic | Multinomial logistic | Linear SVM (hinge) | MLP | Variational GLMM |
|---|---:|---:|---:|---:|---:|
| O | 40.20% | 42.36% | 34.09% | 47.63% | 42.82% |
| FILLER_DM | 22.33% | 21.68% | 18.72% | 25.05% | 22.88% |
| FILLER_HESITATION | 7.61% | 8.95% | 6.71% | 8.65% | 10.19% |
| REPETITION | 7.62% | 8.00% | 7.42% | 8.24% | 8.57% |
| FILLER_BACKCHANNEL | 67.22% | 61.98% | 52.25% | 70.05% | 64.54% |
| PARTIAL_WORD | 3.98% | 3.94% | 3.87% | 4.13% | 4.13% |
| REPAIR | 1.90% | 2.30% | 1.68% | 2.87% | 2.30% |
| PROLONGATION | 1.37% | 1.41% | 1.29% | 1.46% | 1.49% |
| FALSE_START | 1.32% | 1.38% | 1.31% | 1.79% | 1.39% |
| MISPRONUNCIATION | 0.04% | 0.06% | 0.04% | 0.05% | 0.04% |

## Task-updated GloVe: joint representation/classifier training

| Technique | Binary balanced accuracy | Binary positive F1 | Ten-class accuracy | Ten-class macro F1 |
|---|---:|---:|---:|---:|
| One-vs-rest logistic | 74.48% | 47.32% | 30.38% | 16.06% |
| Multinomial logistic | 74.57% | 47.53% | 34.51% | 17.53% |
| Linear SVM (hinge) | 74.57% | 48.25% | 20.99% | 12.60% |
| MLP | 75.54% | 49.98% | 33.58% | 17.82% |

Ten-class F1 by label:

| Class | One-vs-rest logistic | Multinomial logistic | Linear SVM (hinge) | MLP |
|---|---:|---:|---:|---:|
| O | 44.87% | 50.21% | 32.09% | 48.89% |
| FILLER_DM | 25.61% | 25.26% | 16.52% | 26.20% |
| FILLER_HESITATION | 8.99% | 10.06% | 7.14% | 10.02% |
| REPETITION | 9.49% | 10.98% | 8.70% | 10.96% |
| FILLER_BACKCHANNEL | 61.68% | 67.08% | 53.37% | 70.08% |
| PARTIAL_WORD | 3.89% | 4.78% | 2.65% | 4.97% |
| REPAIR | 2.67% | 3.10% | 2.24% | 3.30% |
| PROLONGATION | 1.39% | 1.54% | 1.34% | 1.60% |
| FALSE_START | 1.96% | 2.19% | 1.88% | 2.17% |
| MISPRONUNCIATION | 0.05% | 0.06% | 0.05% | 0.06% |

## Other baselines and missing experiments

The hashed-feature SGD logistic baseline obtained 76.27% binary balanced accuracy, 50.78% binary
positive F1, 82.64% ten-class accuracy and 24.86% ten-class macro F1. It uses a different feature
representation and is therefore a separate baseline, not a matched-GloVe classifier comparison.

The GLMM row, when present, uses a factorized Gaussian variational approximation for a logistic
model with Gaussian random intercepts by preceding context word. Infrequent training words share
a rare-word group; unseen groups use the population prior. Fixed effects use the same frozen GloVe
coordinates. The expected log likelihood uses 12-point Gauss-Hermite quadrature and a Gaussian KL
penalty; this is not the paper's lme4 estimator or the restoration pilot's direct marginal likelihood
optimizer. The two-epoch budget does not establish variational convergence. Posterior diagnostics
are retained in each fold report.

Full-corpus BERT progress, completion state and resume command are in
[the BERT run record](../bert_full/README.md). Small restoration integration runs
do not belong in these tables because their data size, target and metrics differ. No pilot result
is presented as full-corpus evidence. Other unrun pretrained embedding families are also omitted.

## Files and reproduction

- [All comparison metrics](comparison.csv)
- [Per-class precision, recall, F1 and support](per_class.csv)
- [Matched-data/fold/initialization verification](verification.json)
- [MLP fold results and final model paths](../glove50/README.md)
- [OvR logistic report](../glove50_logistic/summary.json)
- [Multinomial logistic report](../glove50_multinomial/summary.json)
- [Linear SVM report](../glove50_svm/summary.json)
- [Variational GLMM report](../glove50_glmm/summary.json)

For each `CLASSIFIER` in `logistic multinomial svm mlp`, run:

```bash
.venv/bin/python scripts/classify_embeddings.py \
  --embedding static --classifier CLASSIFIER \
  --vectors .cache/vectors/glove-wiki-gigaword-50.txt \
  --tasks binary multiclass --fold-manifest results/classification/full/settings.json \
  --epochs 2 --batch-size 8192 --hidden 64 \
  --out results/classification/glove50_CLASSIFIER --resume \
  --fit-final --checkpoints ckpt/classification_glove50_CLASSIFIER
```

The existing MLP report is `glove50/` (omit `_mlp` in its output path). Final-model refits use all
eligible rows and are saved locally under ignored `ckpt/`; their training data are not scored as
held-out evaluation. Run `python scripts/compare_classifiers.py` to validate and recreate these tables.
