# GloVe frozen versus task-fine-tuned embeddings

All **3,085,832 lexical examples** from **2,438 conversations** are evaluated exactly once per
mode/task, using the same ten conversation-separated folds as the hashed logistic baseline.
There is no sample/window limit. Inputs are four surrounding fluent context words, with the
target word withheld. Source annotations define candidate contexts, as in the baseline.
No timing, phone or audio features are used.

Initialization: pretrained 50-dimensional GloVe Wikipedia/Gigaword vectors from the
[gensim-data release](https://github.com/piskvorky/gensim-data/releases/tag/glove-wiki-gigaword-50).
The archive matches the published checksum; source/license/hashes are recorded in
[embedding_sources.json](../../../data/embedding_sources.json).

A 64-unit MLP receives the concatenated positional vectors. Both modes start from identical
encoder and head weights per fold/task. The fixed recipe uses two complete training epochs,
batches of 8,192, AdamW, learning rate 0.001 and training-fold inverse-frequency class weights.
Static vocabulary fitting is confined to each training fold; unseen test words map to UNK.
No test-fold selection of hyperparameters or epochs is performed.

| Model | Binary balanced-test accuracy | Binary positive F1, natural prevalence | Ten-class accuracy | Ten-class macro F1 |
|---|---:|---:|---:|---:|
| Frozen GloVe + MLP | 73.20% | 47.06% | 32.44% | 16.99% |
| Fine-tuned GloVe + MLP | 75.54% | 49.98% | 33.58% | 17.82% |

Fine-tuning improves these GloVe models, but this fixed two-epoch recipe does not outperform
the hashed logistic baseline (binary balanced accuracy 76.27%, ten-class macro F1 24.86%).
The ten-class class-weighted model has low natural-prevalence accuracy; the table retains it
alongside macro F1. These are initial fixed-recipe comparisons, not convergence or best-model claims.

## Ten-class F1

| Class | Frozen GloVe | Fine-tuned GloVe |
|---|---:|---:|
| O | 47.63% | 48.89% |
| FILLER_HESITATION | 8.65% | 10.02% |
| FILLER_BACKCHANNEL | 70.05% | 70.08% |
| FILLER_DM | 25.05% | 26.20% |
| REPETITION | 8.24% | 10.96% |
| REPAIR | 2.87% | 3.30% |
| FALSE_START | 1.79% | 2.17% |
| PARTIAL_WORD | 4.13% | 4.97% |
| PROLONGATION | 1.46% | 1.60% |
| MISPRONUNCIATION | 0.05% | 0.06% |

Full precision/recall/F1/support, fold standard deviations and pooled confusion matrices are in
[per_class.csv](per_class.csv), [comparison.csv](comparison.csv) and [summary.json](summary.json).
[Verification](verification.json) checks source counts, fold coverage, identical initializations,
frozen-weight invariance and actual embedding updates.

## Usable final models

After CV, the same recipe is refitted on all corpus rows for each task/mode. These models and
adapted vectors are local, ignored Git artifacts; no training-set performance is presented as
held-out evaluation. Their paths and training provenance are in [final_models.json](final_models.json).
Fine-tuned binary and ten-class embedding tables are exported as `vectors.vec` beside each final
model. `load_checkpoint()` restores the encoder, classifier and static vocabulary mapping.

## Reproduce

```bash
.venv/bin/python scripts/classify_embeddings.py   --embedding static --vectors .cache/vectors/glove-wiki-gigaword-50.txt   --tasks binary multiclass --fold-manifest results/classification/full/settings.json   --epochs 2 --batch-size 8192 --hidden 64   --out results/classification/glove50 --resume   --fit-final --checkpoints ckpt/classification_glove50
```

The command saves final checkpoint weights; exported text vectors can be recreated from
`load_checkpoint(path)[0].encoder.weight` and `load_checkpoint(path)[1].words`.
Transformer and SentenceTransformer fine-tuning are implemented and tested, but their full-corpus
benchmarks are not included here. See [coverage](../../../docs/method-embedding-coverage.md).
