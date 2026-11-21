# Transcript-only disfluency restoration

Restore omitted disfluencies and tag visible disfluencies in Switchboard transcripts.
The active implementation is `src/disfluency/`; it uses **no audio**.

The modelling methodology follows the [CL2 class project](https://docs.google.com/document/d/1Hq2tNdsbh3rG28jTLDcmvMvbgPwo6QkifkVIXaErbFE/edit)
and [revised proposal](https://docs.google.com/document/d/1xtpreJPhNfsSNpj6aYadWsE3IoMpHcneUIrysRVeSwA/edit),
with the user's subsequent changes: transcript-only inputs, multiple text representations, and modelling without BERT probing.
The linguistic feature reference is [Dammalapati, Rajkumar, Ranjan and Agarwal (2021)](https://aclanthology.org/2021.scil-1.9/).
The earlier audio/NPTEL experiments are preserved in [the historical README](docs/legacy-experiments.md).

## Setup

Python 3.10–3.12:

```bash
uv venv --python 3.12 .venv
uv pip install -e .
# Optional sentence encoders, dependency parser, and corpus-statistics plots:
uv pip install -e '.[embeddings,syntax,stats]'
# If condition E needs automatically generated dependency parses:
uv pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl
```

Existing corpus inputs:

- `data/processed/swb_tokens.jsonl.gz`: the project's token labels and timings.
- `data/raw/switchboard_ms98/sw-ms98-dict.text`: canonical phone dictionary.
- `notebooks/switchboard_stats.ipynb`: corpus statistics and the revised mispronunciation mapping.

The free MS98 release has explicit partial-word/mispronunciation markup, but repetitions,
repairs, false starts and several filler categories in the processed file are **rule-derived**.
These are not human-annotated NXT/Treebank disfluency labels.

## Full-corpus binary and class-wise prediction

The substantive classification experiment uses **all 3,085,832 lexical tokens** across
**2,438 conversations / 4,876 speaker sides** in the existing processed dataset.
The 45,891 generic noise-marker rows are excluded from word classification. There is no
window or sample cap. The small 240-window restoration pilot below is a separate earlier check.

```bash
.venv/bin/python scripts/classify_full.py --resume
```

This runs ten conversation-held-out folds for:

- Any disfluency versus fluent.
- Multiclass prediction of fluent plus the nine existing lexical disfluency types.
- Filler-family versus fluent (hesitation, backchannel and discourse-marker filler).
- Reparandum-family versus fluent (repetition, repair, false start and partial word).

Each task compares surrounding-word context with context plus existing word-alignment durations.
The target word and its own duration are withheld. Features use the nearest two fluent words
on either side, within the utterance. The existing annotations define those fluent contexts;
this is contextual classification with known candidate positions, not blind end-to-end restoration.

Training uses averaged SGD logistic regression over fixed hashed word/bigram features, allowing
all examples to be used without constructing millions of Python feature dictionaries. Training-fold
class weights are enabled by default (`--balancing none` disables them); timing scaling is fitted on
training conversations only. Both natural-prevalence test scores and equal-class held-out binary
scores are exported. The latter are for balanced-task comparison; neither is a claim of an exact
NXT/paper reproduction. Reports retain per-class precision, recall, F1, support and confusion matrices.

Results: `results/classification/full/summary.csv` and `summary.json`. Intermediate fold JSON files
are written after every fit, and `--resume` reuses completed folds with matching settings.
The compact cached arrays are under `data/classification/full/` and excluded from Git.

The [full-corpus GloVe comparison](results/classification/glove50/README.md) additionally evaluates
frozen versus task-fine-tuned embeddings with an MLP on the same folds. Adapted embedding tables
and usable final models are saved locally; their locations are recorded in that report.
The [comparison across modelling techniques](results/classification/model_comparison/README.md)
holds GloVe inputs and folds fixed across OvR logistic, multinomial logistic, linear SVM and MLP,
with both frozen and task-updated embedding variants. It also includes a frozen-feature variational
GLMM with Gaussian context-word random intercepts. The [full-corpus BERT run](results/classification/bert_full/README.md)
is training on the Mac GPU; that record contains progress and the resumable command.

## Prepare and run

Build a small sample spanning 20 conversations:

```bash
.venv/bin/python scripts/restoration.py prepare \
  --max-utterances 240 --max-per-conversation 12 --max-words 8 \
  --out data/restoration/smoke.jsonl.gz

.venv/bin/python scripts/restoration.py run \
  --data data/restoration/smoke.jsonl.gz \
  --conditions A B C D --models rule multinomial svm \
  --representations tfidf ppmi --folds 10 --balancing none \
  --out results/restoration/smoke
```

For the full corpus, omit `--max-utterances` and `--max-per-conversation`.
The preparation step removes selected labelled tokens, retains some visible disfluencies
(`--keep-probability 0.3`), and keeps the original verbatim tokens as restoration targets.
Filler-only windows retain one observed token so they remain represented.
The optional phone noise rates are configurable; set all three to zero for clean dictionary phones.

For the linguistic-feature condition, prepare with an installed parser:

```bash
.venv/bin/python scripts/restoration.py prepare \
  --parser-model en_core_web_sm --out data/restoration/examples.jsonl.gz
.venv/bin/python scripts/restoration.py run --config configs/restoration.yaml
```

The full configuration is a substantial experiment matrix. Start with a sample and selected
models/representations. `--fold-manifest` reuses a validated fold assignment across runs on the same data.

## Models and representations

See [method and embedding coverage](docs/method-embedding-coverage.md) for the distinction between
implemented methods, small integration runs, and completed full-corpus benchmarks.
The new `scripts/classify_embeddings.py` compares frozen versus task-updated static vectors,
transformer encoders and sentence encoders on the full transcript-only classification data, using
the same conversation folds. Pretrained encoders are fine-tuned; random embeddings are trained
from scratch. The existing BERT restoration model already fine-tunes its encoder end to end.

| Model | CLI value | Behaviour |
|---|---|---|
| Visible filler/repetition rules | `rule` | Tags observed fillers and adjacent repeats; inserts nothing |
| Binary one-versus-rest logistic models | `logistic` | Separate binary classifiers over restoration/BIO classes |
| Multinomial logistic regression | `multinomial` | Joint softmax classification |
| Linear SVM | `svm` | One-versus-rest linear decision scores |
| MLP | `mlp` | Dense hidden layer over continuous representations |
| Random-intercept GLMM | `glmm` | Numeric fixed effects and Gaussian context-word random intercepts |
| Fine-tuned BERT | `bert` | Shared contextual encoder, BIO word labels and insertion-slot classes |

For classical models, `--representations` accepts:

- `tfidf`: combined word uni/bigrams and character 3–5-grams.
- `ppmi`: static distributional embeddings trained on each training fold using positive PMI and SVD.
- `static`: supplied Word2Vec/fastText **text** vectors (`--vectors /path/to/vectors.vec`).
- `contextual`: frozen transformer mean-pooled context embeddings (`--encoder bert-base-uncased`, or another compatible encoder).
- `sentence`: SentenceTransformer context embeddings (`--encoder MODEL_OR_LOCAL_DIRECTORY`).

Frozen embeddings use the local three-word context for each word or insertion boundary.
Auxiliary phone and linguistic features are concatenated separately, keeping the lexical representation
consistent across ablations. The GLMM uses numeric features; the fine-tuned BERT has its own representation.
Pretrained models are loaded from cache by default; `--allow-download` explicitly enables downloading.

Example frozen embeddings:

```bash
.venv/bin/python scripts/restoration.py run \
  --data data/restoration/smoke.jsonl.gz --conditions A B \
  --models multinomial --representations contextual --encoder bert-base-uncased \
  --out results/restoration/contextual
```

## What the ablations mean without audio

| Condition | Inputs | Interpretation |
|---|---|---|
| A | Incomplete transcript words | Deployable text-only baseline |
| B | A + canonical phones of those words | Same information, different representation |
| C | A + dictionary-derived phones of the original verbatim text | Simulated auxiliary evidence |
| D | A + edit alignment of the two text-derived phone streams | Simulated mismatch evidence |
| E | D + surprisal, dependency locality and timed-transcript duration | Linguistic feature contribution |
| F | Supplied paired gold/ASR transcripts under D or E | Optional paired-text comparison; no ASR stage is run |

C–E have access to an auxiliary stream derived from the original text. They are controlled simulations,
not evidence that missing speech can be recovered from a lone incomplete transcript.
A and B are the appropriate conditions for plain-text inference. Each output records its phone provenance.

Alignment spans neighbouring words with unit-cost edit distance. It reports matching, substituted,
deleted and inserted phones, including unassigned boundary phones. These operations are observations,
not disfluency labels. Condition C sees the raw phone stream without canonical-derived allocation.
BERT D/E serialize `word | /canonical/ | /realized/ | ops | gap /phones/`.

The linguistic features include preceding/following trigram surprisal, duration, elongation,
syllable-normalized duration, and dependency-locality integration/storage costs. The latter follow
§2.2 of the paper; condition E requires real dependency heads and existing alignment durations.
A training-fold add-0.1 trigram model replaces the paper's OANC/SRILM language model.
PCFG syntactic surprisal can be supplied; it is not approximated with lexical surprisal.
The GLMM integrates random intercepts with Gauss–Hermite quadrature, rather than reproducing lme4.

## Balancing and evaluation

`--balancing none under smote cluster` compares unbalanced, random undersampling,
SMOTE and cluster undersampling. All fitted preprocessing and sampling use only training-fold rows.
SMOTE operates on continuous features, never text or token IDs. Singleton classes use exact replication,
recorded in the sampling report. Cluster sampling selects real examples nearest training-only centroids.
For BERT, balanced variants refit the final classifier on frozen representations from the training fold
following encoder fine-tuning; the report identifies this two-stage procedure.

Ten folds hold out whole conversations, keeping both speaker sides together. Every model/condition uses
the same manifest. There is no tuning on held-out folds. Hyperparameters/epochs must be fixed in advance.
Reports include:

- Per-tag exact and relaxed span precision, recall and F1; micro/macro F1.
- Mean and sample standard deviation across folds.
- Separate **restoration-only** scores for inserted material.
- Word-aligned BIO confusion counts and tag-aware WER (edit distance over word/tag pairs).
- Predictions, class-balancing diagnostics and data provenance per fold.

Spans are maximal BIO runs. Exact matching requires the same type and all correctly word-aligned tokens
of a reference span. Relaxed matching requires the same type and at least one correctly aligned token.
Both use maximum one-to-one matching, so two fragments cannot claim the same reference span twice.

The decoder supports learned insertion templates and relative copy pointers for repeated/repaired words.
It can copy a new input word without having seen that literal word as a restoration class.
Other inserted words/phrases come from the training-fold template vocabulary. This is a finite-template
restoration model, not unrestricted text generation. Out-of-window copy pointers abstain and are counted.

## Train and restore text

```bash
.venv/bin/python scripts/restoration.py train \
  --data data/restoration/examples.jsonl.gz --condition B \
  --model multinomial --representations tfidf --out ckpt/restoration/text_model

.venv/bin/python scripts/restoration.py predict \
  --model-dir ckpt/restoration/text_model --text 'i think we should go' \
  --out results/restoration/prediction.jsonl
```

Outputs contain restored text, BIO tags, insertion flags, observed-token indices and provenance.
Predictions from a single transcript are statistical predictions; its own canonical phones do not reveal
omitted words. Load only checkpoints produced by this project, since model metadata uses pickle.

For fine-tuning, use `--model bert`. `--tiny` creates a randomly initialized tiny BERT for engineering
tests; its results are explicitly marked and are not pretrained-model results.
Inputs exceeding the BERT position limit raise an error; they are never silently truncated.

## Validation and data contract

```bash
.venv/bin/python -m pytest -q
```

See [the data schema and methodology notes](docs/restoration-methodology.md) for supplying your own
transcripts, parses and targets. Corpus statistics remain available with:

```bash
.venv/bin/python scripts/switchboard_stats.py
```
