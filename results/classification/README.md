# Full existing-corpus classification results

All **3,085,832 lexical tokens**, from **4,876 speaker sides / 2,438 conversations**, were used.
There is no sample or window cap. 45,891 noise-marker rows are excluded from word classification.
Each conversation belongs to exactly one of ten held-out folds; both speaker sides stay together.
All ten folds completed for every configuration (120 full-corpus fold fits in total).

## Primary results: transcripts only

Averaged logistic regression with training-fold class weights and surrounding fluent-word context.
The primary results use no durations, timing, phones or audio. Percentages are fold means;
± is the sample standard deviation. Balanced-test accuracy uses an equal-class subset of held-out
predictions; positive-class F1 uses the full natural-prevalence held-out data.

| Task | Balanced-test accuracy (%) | Natural-prevalence disfluency F1 (%) |
|---|---:|---:|
| Any disfluency vs fluent | 76.27 ± 0.18 | 50.78 |
| Filler family vs fluent | 78.91 ± 0.28 | 46.61 |
| Reparandum family vs fluent | 75.51 ± 0.24 | 25.45 |

## Class-wise prediction: transcripts only

Ten-class accuracy **82.64%**; macro F1 **24.86%**.
Supports sum over all ten folds; precision/recall/F1 are fold means, in percent.

| Class | Support | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
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

## Supplementary timing and weighting controls

A separate context+timing experiment used existing neighbouring word durations. It obtained
78.43% / 82.05% / 76.78% balanced-test accuracy for any-disfluency / filler / reparandum,
and 26.32% ten-class macro F1. These are supplementary and do not meet the stricter transcripts-only
feature constraint of the primary results.

Without class weighting, context+timing binary natural-prevalence accuracy was 88.85%,
compared with 85.29% for always predicting fluent. Positive-class precision was 71.91%, recall
39.70%, and F1 51.16%. Unweighted ten-class accuracy was 87.02% and macro F1 21.16%.

## Interpretation and files

These are **contextual classification** results, not exact missing-word restoration scores.
The target word is withheld. Existing labels identify the surrounding fluent contexts;
candidate positions are known. The dataset retains its existing project-rule annotation provenance.
Filler and reparandum family groupings are documented explicitly; these results are not claimed
to be an exact NXT/SCiL reproduction.

- [Weighted comparison](full/comparison.csv)
- [Weighted per-class precision/recall/F1 and support](full/per_class.csv)
- [Weighted full report and confusion matrices](full/summary.json)
- [Unweighted comparison](unbalanced/comparison.csv)
- [Unweighted per-class results](unbalanced/per_class.csv)
- [Corpus counts and source hash](full/data_manifest.json)
- [Methodology](../../docs/full-corpus-classification.md)

The cache and training pipeline use every lexical source row. Class counts were verified
against `data/processed/label_stats.json`. Supports in out-of-fold multiclass predictions were
verified against those same source counts, and weighted/unweighted runs share identical folds.
