#!/usr/bin/env bash
# Bundle code + gap datasets for Colab. Upload the resulting zip to Drive: MyDrive/disfluency/colab_bundle.zip
# Usage: scripts/package_for_colab.sh
set -euo pipefail
cd "$(dirname "$0")/.."
OUT="colab_bundle.zip"
rm -f "$OUT"
zip -q -r "$OUT" src scripts configs notebooks \
  data/gaps/train.jsonl.gz data/gaps/dev.jsonl.gz data/gaps/test.jsonl.gz \
  data/gaps/restoration_vocab.json data/gaps/gap_stats.json \
  -x "*/__pycache__/*" "*.pyc"
ls -lh "$OUT"
echo "Upload $OUT to Google Drive at MyDrive/disfluency/ and open notebooks/train_colab.ipynb in Colab (GPU runtime)."
