# Transcript-only Disfluency Prediction: Results Through Frozen BERT

**Date:** 21 November 2026  
**Dataset:** existing Switchboard MS98 project-labelled transcripts  
**Scope:** completed contextual-classification baselines, static-embedding experiments, GLMM, and frozen-BERT validation status. Ongoing end-to-end BERT fine-tuning is excluded from the numerical tables.

## 1. Summary

The completed full-corpus experiments use **all 3,085,832 lexical examples** from **2,438 conversations**, evaluated with ten conversation-separated folds. No window or sample cap is applied to these benchmarks.

The hashed word/bigram logistic baseline achieves **76.27% accuracy on a balanced binary test subset**, **50.78% disfluency F1**, and **24.86% ten-class macro F1**. Within the comparison holding frozen GloVe inputs and training budget fixed, MLP has the highest binary balanced-test accuracy (**73.20%**) and ten-class macro F1 (**16.99%**).

**Frozen BERT does not yet have a completed full-corpus benchmark.** Its encoder/classifier integration and reload path were checked, but no defensible ten-fold accuracy or F1 is available. Missing scores are reported as **not available**, not as zero. Frozen GloVe results and fine-tuned BERT results are not relabelled as frozen-BERT results.

The full-corpus scores below measure **contextual disfluency classification at known candidate positions**. They do not measure exact reconstruction of missing words from an unannotated transcript.

## 2. Dataset and prediction tasks

### 2.1 Corpus coverage

| Item | Count |
| --- | ---: |
| Source token rows | 3,131,723 |
| Excluded generic noise/laughter-marker rows | 45,891 |
| Eligible lexical examples | 3,085,832 |
| Conversations | 2,438 |
| Speaker sides | 4,876 |
| Cross-validation folds | 10 |

Source labels retain the existing project tagset. Several categories are derived from MS98 project rules rather than human NXT/Treebank disfluency annotation. The existing processed MS98 corpus is used throughout; this is not the paper's exact NXT release.

### 2.2 Class distribution

| Class | Lexical examples | Share (%) |
| --- | ---: | ---: |
| O | 2,631,755 | 85.29 |
| FILLER_HESITATION | 94,533 | 3.06 |
| FILLER_BACKCHANNEL | 68,152 | 2.21 |
| FILLER_DM | 133,671 | 4.33 |
| REPETITION | 86,403 | 2.80 |
| REPAIR | 17,310 | 0.56 |
| FALSE_START | 9,119 | 0.30 |
| PARTIAL_WORD | 28,564 | 0.93 |
| PROLONGATION | 16,017 | 0.52 |
| MISPRONUNCIATION | 308 | 0.01 |

The fluent class accounts for **85.29%** of examples. An always-fluent predictor would therefore obtain 85.29% natural-prevalence accuracy and **zero disfluency F1**. Accuracy alone is insufficient to assess this dataset.

### 2.3 Tasks and inputs

- **Any-disfluency binary:** all nine lexical disfluency types versus fluent (`O`).
- **Ten-class:** fluent plus the nine lexical disfluency types.
- **Filler-family binary:** hesitation, backchannel and discourse-marker filler versus fluent; other disfluency types are excluded.
- **Reparandum-family binary:** repetition, repair, false start and partial word versus fluent; other types are excluded.

Family groupings are explicit project-label mappings and are not asserted to match NXT event definitions exactly.

For the full-corpus experiments, each candidate uses the nearest two fluent words before and after it within the same utterance. The candidate's own word, phones, markup and duration are withheld. **Source annotations determine candidate positions and which context words are fluent.** The benchmark therefore depends on annotated context construction; it is not blind detection or restoration from ordinary unannotated text.

The primary results use no audio, phones, timing or duration features. Historical timing-enriched controls remain in the source files but are excluded from the transcript-only tables here.

## 3. Evaluation protocol

Both speaker sides of a conversation remain in the same held-out fold. Models use the same ten conversation assignments; each eligible example is tested exactly once per configuration. Fitted vocabularies, class weights and learned model coefficients use training conversations only.

All reported performance values are percentages. Full-corpus tables show the **mean across ten folds**; `±` denotes the sample standard deviation across folds, not a confidence interval.

- **Binary balanced-test accuracy:** ordinary accuracy on a fixed equal-class subset of held-out predictions. This differs from directly reporting scikit-learn's balanced-accuracy statistic on the full test distribution.
- **Binary precision, recall and F1:** the disfluency-positive class, evaluated on the complete held-out data at natural prevalence.
- **Ten-class accuracy and macro F1:** evaluated on the complete natural-prevalence held-out data; macro F1 gives each of the ten labels equal weight.
- **Class support:** summed held-out support over all ten folds. Precision/recall/F1 are fold means.

Binary balanced-test accuracy and natural-prevalence F1 use different evaluation distributions and should not be interpreted as interchangeable measures. No outer-test fold selects epochs or hyperparameters, and no statistical significance claim is made from the descriptive rankings below.

## 4. Full-corpus lexical baseline

The baseline uses averaged SGD logistic regression over a fixed 2^18-dimensional hash space containing four positional word identities and three positional bigrams. The multiclass estimator uses one-versus-rest classification. It trains for five complete passes with inverse-frequency weights computed separately in each training fold, alpha = 10^-6, and seed 42.

| Binary task | Examples | Balanced-test accuracy | Positive precision | Positive recall | Positive F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Any disfluency | 3,085,832 | 76.27 ± 0.18 | 39.34 | 71.59 | 50.78 ± 0.30 |
| Filler family | 2,928,111 | 78.91 ± 0.28 | 34.04 | 73.89 | 46.61 ± 0.36 |
| Reparandum family | 2,773,151 | 75.51 ± 0.24 | 15.45 | 72.28 | 25.45 ± 0.63 |

Ten-class accuracy: **82.64 ± 0.16%**. Ten-class macro F1: **24.86 ± 0.15%**.

### 4.1 Class-wise lexical-baseline results

| Class | Support | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: |
| O | 2,631,755 | 92.12 | 91.60 | 91.86 |
| FILLER_HESITATION | 94,533 | 21.91 | 10.58 | 14.26 |
| FILLER_BACKCHANNEL | 68,152 | 60.91 | 91.15 | 73.02 |
| FILLER_DM | 133,671 | 30.70 | 37.35 | 33.70 |
| REPETITION | 86,403 | 21.61 | 15.92 | 18.33 |
| REPAIR | 17,310 | 4.61 | 5.16 | 4.86 |
| FALSE_START | 9,119 | 2.72 | 8.73 | 4.15 |
| PARTIAL_WORD | 28,564 | 8.11 | 5.87 | 6.80 |
| PROLONGATION | 16,017 | 2.03 | 1.21 | 1.52 |
| MISPRONUNCIATION | 308 | 0.05 | 2.66 | 0.09 |

Source: [full lexical summary](../results/classification/full/summary.json), [per-class CSV](../results/classification/full/per_class.csv).

## 5. Matched comparison with frozen GloVe embeddings

Every model receives the same external **50-dimensional GloVe** vectors at each of four context positions, concatenated into 200 input coordinates. Training-only vocabularies map unseen held-out words to `UNK`. Words missing from the external vectors receive seeded random initializations. In this section the lookup table remains frozen.

Each model uses **two full training epochs**, batches of 8,192, AdamW with learning rate 0.001 and weight decay 0.01, and training-fold inverse-frequency sample weights. The training seed is 42 plus the fold index. These are fixed-budget fits; convergence and optimal hyperparameters have not been established.

| Technique | Definition |
|---|---|
| OvR logistic regression | Linear scores with independent sigmoid binary cross-entropy |
| Multinomial logistic regression | Linear scores with joint softmax cross-entropy |
| Linear SVM | One-versus-rest hinge-loss linear model trained in minibatches; not a liblinear fit |
| MLP | One 64-unit ReLU hidden layer, 0.1 dropout, softmax output |
| Variational GLMM | GloVe fixed effects plus Gaussian random intercepts by preceding fluent context word |

The GLMM pools infrequent training words into a rare-word group and uses the population distribution for unseen groups. It fits factorized Gaussian variational posteriors with 12-point Gauss–Hermite expected-log-likelihood quadrature, Gaussian KL shrinkage and a half-normal population-scale prior. This is a Bayesian approximation, distinct from the paper's frequentist lme4 fit and the small restoration prototype's direct marginal-likelihood estimator.

### 5.1 Binary results

| Technique | Balanced-test accuracy | Positive precision | Positive recall | Positive F1 |
| --- | ---: | ---: | ---: | ---: |
| OvR logistic regression | 69.39 ± 0.28 | 31.61 | 61.89 | 41.84 ± 0.52 |
| Multinomial logistic regression | 69.80 ± 0.24 | 31.41 | 63.57 | 42.04 ± 0.50 |
| Linear SVM (hinge loss) | 68.89 ± 0.24 | 36.03 | 54.48 | 43.37 ± 0.42 |
| MLP | 73.20 ± 0.20 | 36.47 | 66.44 | 47.06 ± 0.63 |
| Variational GLMM | 70.93 ± 0.26 | 33.53 | 63.64 | 43.92 ± 0.46 |

### 5.2 Ten-class results

| Technique | Accuracy | Macro F1 |
| --- | ---: | ---: |
| OvR logistic regression | 26.44 ± 3.22 | 15.36 ± 0.50 |
| Multinomial logistic regression | 28.35 ± 3.20 | 15.20 ± 0.58 |
| Linear SVM (hinge loss) | 22.33 ± 5.73 | 12.74 ± 0.91 |
| MLP | 32.44 ± 4.65 | 16.99 ± 0.74 |
| Variational GLMM | 28.51 ± 2.97 | 15.84 ± 0.51 |

### 5.3 Ten-class F1 by modelling technique

| Class | OvR logistic | Multinomial | Linear SVM | MLP | Variational GLMM |
| --- | ---: | ---: | ---: | ---: | ---: |
| O | 40.20 | 42.36 | 34.09 | 47.63 | 42.82 |
| FILLER_HESITATION | 7.61 | 8.95 | 6.71 | 8.65 | 10.19 |
| FILLER_BACKCHANNEL | 67.22 | 61.98 | 52.25 | 70.05 | 64.54 |
| FILLER_DM | 22.33 | 21.68 | 18.72 | 25.05 | 22.88 |
| REPETITION | 7.62 | 8.00 | 7.42 | 8.24 | 8.57 |
| REPAIR | 1.90 | 2.30 | 1.68 | 2.87 | 2.30 |
| FALSE_START | 1.32 | 1.38 | 1.31 | 1.79 | 1.39 |
| PARTIAL_WORD | 3.98 | 3.94 | 3.87 | 4.13 | 4.13 |
| PROLONGATION | 1.37 | 1.41 | 1.29 | 1.46 | 1.49 |
| MISPRONUNCIATION | 0.04 | 0.06 | 0.04 | 0.05 | 0.04 |

Complete precision/recall/F1, supports, fold standard deviations and the representation-update controls are available in the [model-comparison CSV](../results/classification/model_comparison/comparison.csv) and [per-class CSV](../results/classification/model_comparison/per_class.csv).

### 5.4 Interpretation

MLP has the highest binary balanced-test accuracy and ten-class macro F1 within this matched frozen-GloVe comparison. The variational GLMM improves those two measures over its corresponding OvR linear baseline, but does not exceed MLP. These are descriptive comparisons under a common training budget.

The hashed lexical baseline has higher binary positive F1 and ten-class macro F1 than these frozen-GloVe models. That comparison changes both the representation and training recipe—five SGD passes versus two AdamW passes—so it does not isolate an embedding or classifier effect.

Low natural-prevalence ten-class accuracy is retained in the tables rather than obscured by balanced-test binary scores. Several rare-type F1 scores remain very low despite full corpus coverage. The observed class frequencies and project-rule annotation provenance are relevant to interpretation; these results alone do not establish the cause of each class's errors.

## 6. Frozen BERT: validation completed, benchmark unavailable

**Encoder:** `bert-base-uncased`.

The restoration prototype implements frozen transformer features using an attention-mask-weighted mean of final-layer token representations over local context. Encoder inference runs in evaluation mode without gradients; a supervised classifier is fitted over the resulting features. This is downstream modelling, with no BERT probing experiment.

Saved verification records report that a multinomial classifier using frozen BERT features was trained and reloaded for inference without gold target fields. The pretrained-model checks are documented as small engineering checks, not full-corpus cross-validation.

| Frozen-BERT item | Status |
| --- | ---: |
| Encoder loading and feature extraction | Checked |
| Supervised classifier and checkpoint reload | Checked |
| Inference without target labels | Checked |
| Full-corpus ten-fold binary benchmark | Not completed |
| Full-corpus ten-fold ten-class benchmark | Not completed |
| Accuracy, precision, recall and F1 | Not available |

Consequently, there is **no frozen-BERT performance row to rank against the completed baselines**. The ongoing Colab run updates BERT's encoder weights; its partial results are fine-tuned BERT results and fall outside this report's frozen-BERT endpoint.

Evidence: [engineering-check description](../results/restoration/README.md), [verification metadata](../results/restoration/verification.json), [frozen representation implementation](../src/disfluency/representations.py).

## 7. Relation to Raja's paper

Dammalapati, Rajkumar, Ranjan and Agarwal (2021) report **81.21% filler accuracy** and **71.11% reparandum accuracy** for their final GLMM, evaluated with five-fold cross-validation. Their study uses NXT Switchboard annotations, linguistic predictors including duration, and word random effects. These are binary prediction accuracies, not exact restoration F1 scores. [Paper, Tables 1–2 and §3](https://aclanthology.org/2021.scil-1.9.pdf).

The present study uses the existing MS98 project labels, ten conversation-held-out folds, explicit project family mappings, transcript-only primary inputs and different GLMM estimation. These differences prevent a direct numerical reproduction claim. In particular, our broader filler family includes backchannels and discourse-marker fillers; its score should not be equated with the paper's filler score.

The paper motivates the contextual prediction task and mixed-effects comparison. It does not supply a missing frozen-BERT benchmark or establish that classification accuracy and missing-word restoration F1 are comparable.

## 8. Conclusions at this cutoff

1. Full-corpus binary and ten-class baselines have been completed with conversation-level separation; the report is not based on the earlier artificial pilot cap.
2. The hashed lexical model is the strongest completed primary baseline by binary disfluency F1 and ten-class macro F1 in this report.
3. MLP leads the matched frozen-GloVe classifier comparison, while many rare types remain difficult to predict from the withheld-target context construction.
4. Frozen BERT has an implemented and checked inference path, but no completed quantitative full-corpus result. Its performance remains an open comparison.
5. The results establish contextual label-prediction performance under annotation-dependent inputs. They do not establish full-corpus end-to-end disfluency restoration performance.

## Appendix A. Completed GloVe update ablation

This completed ablation is included for completeness among experiments performed before the BERT stage. It updates the static GloVe lookup table jointly with the classifier, and must not be confused with frozen embeddings or BERT fine-tuning. Each frozen/updated pair starts from identical embeddings and classifier initialization for the same fold/task; data, folds, class weighting and the two-epoch budget are unchanged.

| Technique | Binary balanced-test accuracy | Binary positive F1 | Ten-class accuracy | Ten-class macro F1 |
| --- | ---: | ---: | ---: | ---: |
| OvR logistic regression | 74.48 ± 0.19 | 47.32 ± 0.30 | 30.38 ± 3.95 | 16.06 ± 0.67 |
| Multinomial logistic regression | 74.57 ± 0.18 | 47.53 ± 0.34 | 34.51 ± 3.39 | 17.53 ± 0.47 |
| Linear SVM (hinge loss) | 74.57 ± 0.17 | 48.25 ± 0.36 | 20.99 ± 5.41 | 12.60 ± 1.12 |
| MLP | 75.54 ± 0.12 | 49.98 ± 0.56 | 33.58 ± 4.73 | 17.82 ± 0.72 |

Updating GloVe improves binary balanced-test accuracy for all four models. Ten-class macro F1 improves for the two logistic heads and MLP, while it decreases slightly for SVM. The best updated-GloVe MLP reaches 75.54% binary balanced-test accuracy and 17.82% ten-class macro F1 under this recipe.

## Appendix B. Earlier text-only restoration pilot

The prototype was checked on **240 prepared windows from 20 conversations**, using ten conversation-held-out folds. This was an engineering sample deliberately capped by the experiment configuration, not a statement about the size of the available corpus. The table below is restricted to condition A: incomplete transcript words only, without simulated auxiliary phone evidence.

These scores concern word tagging plus insertion-slot restoration, so they are not directly comparable with the full-corpus contextual-classification tables. Overall exact-span F1 includes visible disfluencies; insertion-only F1 evaluates missing material separately. Tag-aware WER is lower-is-better. All values are percentages and means across folds.

| Model | Representation | Overall exact-span F1 | Insertion-only exact F1 | Tag-aware WER |
| --- | ---: | ---: | ---: | ---: |
| multinomial | ppmi | 0.00 ± 0.00 | 0.00 | 12.82 |
| multinomial | tfidf | 20.62 ± 17.41 | 0.00 | 11.21 |
| rule | rules | 20.11 ± 7.23 | 0.00 | 11.18 |
| svm | ppmi | 0.00 ± 0.00 | 0.00 | 12.51 |
| svm | tfidf | 15.92 ± 15.57 | 0.00 | 11.57 |

TF–IDF and PPMI in this appendix are pilot representations, not completed full-corpus embedding comparisons. No correctly matched inserted spans were obtained in these condition-A pilot rows. Their visible-tagging F1 cannot be presented as evidence of successful missing-word restoration. Other prototype conditions used dictionary-derived simulated evidence and/or existing timings and are excluded from the main transcript-only results.

Source: [pilot summary](../results/restoration/validation/summary.json), [pilot interpretation](../results/restoration/README.md).

## Appendix C. Evidence and reproducibility

| Evidence | Location |
|---|---|
| Corpus counts and original labels | [label statistics](../data/processed/label_stats.json) |
| Full-corpus source identity and counts | [data manifest](../results/classification/full/data_manifest.json) |
| Shared conversation-to-fold map | [baseline settings](../results/classification/full/settings.json) |
| Lexical baseline metrics and confusion counts | [full summary](../results/classification/full/summary.json) |
| Matched GloVe comparison | [comparison report](../results/classification/model_comparison/README.md) |
| Model-comparison data/fold/initialization checks | [verification](../results/classification/model_comparison/verification.json) |
| Frozen and updated GloVe MLP | [summary](../results/classification/glove50/summary.json) |
| Frozen and updated GloVe logistic heads | [OvR](../results/classification/glove50_logistic/summary.json), [multinomial](../results/classification/glove50_multinomial/summary.json) |
| Frozen and updated GloVe SVM | [summary](../results/classification/glove50_svm/summary.json) |
| Variational GLMM and posterior diagnostics | [summary](../results/classification/glove50_glmm/summary.json) |
| Frozen-BERT validation provenance | [engineering verification](../results/restoration/verification.json) |
| Prediction/evaluation methodology | [full-corpus protocol](../docs/full-corpus-classification.md) |

Source dataset SHA-256:

```text
c2f7a82de3c57796f4de221704fc4c77cf056a77fe1593a189c8a814ccafb9e1
```

Recreate the existing matched-model tables with:

```bash
.venv/bin/python scripts/compare_classifiers.py
```

The archived source JSON/CSV files and per-fold reports are the quantitative evidence for this report. No ongoing fine-tuned-BERT partial score has been incorporated into the completed tables.
