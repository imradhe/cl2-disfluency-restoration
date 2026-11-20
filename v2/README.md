# CL2 Project — v2

First step: audit the local Switchboard transcript release and inventory its
annotations and the existing project's disfluency labels.

Run from the project root:

```bash
.venv/bin/python v2/switchboard_stats.py
```

Read [the corpus report](reports/switchboard_stats.md). Tables and complete token
form counts are also saved as CSV and JSON in `reports/`.

The script reads the shared raw archive and existing labelled data without
copying the corpus or modifying v1. Repetitions, repairs, false starts and
prolongations in the existing data are heuristic labels, not native gold
annotations in the ISIP/MS-State transcript release. Counts are tokens rather
than episodes.

Raw corpus files, generated datasets, tokenizer caches, and model weights are
excluded from Git. Restore the source corpus using `data/MANIFEST.md` and run
`scripts/build_processed.py` before building v2 data. On a fresh checkout, run
`v2/build_data.py` without `HF_HUB_OFFLINE=1` once to download the standard BERT
tokenizer; a trained v1 checkpoint is not required.

## BERT restoration experiment

```bash
HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false .venv/bin/python v2/build_data.py
HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false .venv/bin/python v2/train.py --epochs 6
HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false .venv/bin/python v2/calibrate.py
.venv/bin/python v2/summarize.py
.venv/bin/python -m pytest -q v2/test_restoration.py
.venv/bin/python v2/predict.py --text 'the cat' --phones 'ah m dh ax k ae t'
```

The compact BERT uses two encoder layers, hidden size 128, and four attention
heads. Its weights start from random initialization; it reuses BERT's cached
WordPiece tokenizer. It takes flat text and the complete phone stream, predicts
insertions between words, and renders the restored transcript. Auxiliary phone
labels are used only for training losses, never as model inputs.

Training uses all training conversations, with every insertion-type disfluency
removed from the text. Phone streams are dictionary-derived and keep the
removed tokens, with simulated corruption on 70% of training windows. This
experiment does not evaluate an audio phone recognizer.

The best development checkpoint is saved in `checkpoints/bert_compact/`.
The full result report is written to `reports/restoration_results.md`, metrics
to `reports/restoration_results.json`, and restored test transcripts to
`reports/test_{clean,noisy,no_phones}_predictions.jsonl.gz`.

Calibration selects a decoder stop offset by development transcript error rate.
It also reports a second operating point selected by development restoration F1.
The default `model.pt` uses the former; `predict.py --weights model_f1_decoder.pt`
uses the latter. Test scores do not select either offset.

Gold references retain every insertion token, including gaps exceeding the
decoder's eight-token capacity. Partial words retain their spoken fragments.
Reported word error rate compares actual restored text to the reference;
typed restored-token F1 also requires the correct insertion gap and type.

## Pretrained BERT on Colab

`colab/v2_pretrained_bert_colab.ipynb` launches full `bert-base-uncased`
fine-tuning on a Colab GPU. Upload the accompanying
`colab/switchboard_v2_colab.zip` when prompted. Generate both with
`.venv/bin/python v2/make_colab.py` after changing any bundled source code.

The GPU run loads the pretrained 12-layer, hidden-size-768 encoder, reuses the
fixed v2 conversation split, and derives pronunciation guidance from the flat
input words. Both checkpoint selection and decoder selection use development
disfluency F1. Reported test scores are disfluency precision, recall, and F1;
the run does not select or report WER. Results are downloaded as
`v2_pretrained_bert_results.zip`. The full training command is in
`v2/train_colab.py`; `predict.py` also supports these pretrained checkpoints.

Colab notebook: https://colab.research.google.com/drive/1Ggn5l8IchTkdxOjfBNkehDqVPjXMAigH

Install and validate the downloaded result archive locally:

```bash
.venv/bin/python v2/install_colab.py v2_pretrained_bert_results.zip
```

The pretrained run's report is `reports/pretrained_colab_f1.md`, with exact
metrics in `reports/pretrained_colab_f1.json` and evenly spaced restored test
examples in `reports/pretrained_colab_examples.md`. Installed weights remain
under `checkpoints/pretrained_colab/`, outside Git.
