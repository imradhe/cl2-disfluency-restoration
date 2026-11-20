# Full-corpus binary and class-wise prediction

This experiment uses the existing labelled MS98 corpus, without NXT acquisition or external
communication. It corrects the earlier pilot's artificial 240-window limit.

## Corpus coverage

- Source: `data/processed/swb_tokens.jsonl.gz`.
- 3,131,723 source rows, 4,876 speaker sides and 2,438 conversations.
- Every one of the 3,085,832 lexical tokens is an example.
- The 45,891 generic noise/laughter marker rows are excluded from word classification.
- Source SHA-256, all class counts and the absence of a sample limit are recorded in
  `results/classification/full/data_manifest.json`.
- Counts in the compact cache are checked against `data/processed/label_stats.json`.

## Prediction task

Each lexical token supplies its existing primary label as the target. The target word, its phones,
raw markup and its own duration are withheld. Features consist of the nearest two fluent words on
each side in the same utterance, word/bigram identities, and optionally the preceding/following
fluent words' durations. Utterance boundaries use explicit BOS/EOS context and missing-time flags.

The source annotations define which surrounding words are fluent, as a contextual classification
construction. Candidate positions and fluent context extraction are known in this experiment.
Scores are therefore token-level contextual classification scores; they are not end-to-end
restoration of unknown missing words from an unannotated transcript.

The four tasks are:

| Task | Positive classes / outputs |
|---|---|
| Binary | Any of the nine lexical disfluency labels versus O |
| Multiclass | O, hesitation, backchannel, DM filler, repetition, repair, false start, partial word, prolongation, mispronunciation |
| Filler family | Hesitation, backchannel and DM filler versus O; other types excluded |
| Reparandum family | Repetition, repair, false start and partial word versus O; other types excluded |

The family mappings are explicit project-label groupings. They are not asserted to reproduce
NXT's event definitions or the paper's filtered sample exactly. Original annotation provenance
is retained; no heuristic type is relabelled as human gold.

## Features and model

Context features use a fixed 2^18-dimensional hash space. Four positional word identities and
three positional bigram identities are encoded. Global integer word codes are only a lossless
identifier schema; no IDF, vocabulary-frequency estimate or language model is fitted on held-out
labels. All feature coefficients are learned from training conversations only.

The timing view adds preceding/following duration, log(1+duration), and availability flags.
The scaler is fitted only on training rows. Neither audio nor a new alignment is required.
This first full-corpus baseline does not include dependency-locality or surprisal estimates;
those are separate additions supported elsewhere in the modelling code.

Averaged SGD logistic regression makes it feasible to fit every training example in all ten folds.
The run fixes five complete optimization passes, alpha=1e-6 and seed=42. There is no held-out-fold
hyperparameter selection. The multiclass estimator uses sklearn's one-versus-rest construction.

The main run uses inverse-frequency class weights computed from each training fold. A second
run uses no class weighting. No majority examples are discarded in either training run.

## Folds and metrics

Ten GroupKFold partitions hold out whole conversations, including both speaker sides. The
conversation-to-fold map is saved, and train/test conversation disjointness is asserted before
fitting. Each eligible example is tested exactly once per view/task; it participates in the
other nine training folds.

Each fold reports accuracy, macro F1, per-class precision/recall/F1/support and a confusion matrix.
Binary tasks also report predictions evaluated on a fixed equal-class subset of the held-out
fold. This balanced evaluation changes no training data and fits nothing on test examples.
The full natural-prevalence scores remain available alongside it.

The all-fluent baseline and positive-class F1 are included in the readable comparison table so
high accuracy on an imbalanced corpus is not confused with disfluency recall. Means and sample
standard deviations are computed across folds; pooled confusion matrices are also preserved.

## Reproduce

```bash
.venv/bin/python scripts/classify_full.py --resume
.venv/bin/python scripts/classify_full.py \
  --views context_timing --balancing none \
  --out results/classification/unbalanced --resume
```

The cache occupies approximately 92 MB; no large per-example Python dictionaries or text-embedding
matrix are retained. The cache is ignored by Git. Fold results are written immediately, and resume
requires identical source/experiment settings.

Reports are in `results/classification/full/` and `results/classification/unbalanced/`:
`summary.json`, `summary.csv`, `comparison.csv`, `per_class.csv`, `settings.json`,
`data_manifest.json`, and one JSON result per fold/view/task.
