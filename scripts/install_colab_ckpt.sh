#!/usr/bin/env bash
# Install the zip produced by notebooks/train_colab.ipynb into ckpt/colab_main and sanity-check it.
# Usage: scripts/install_colab_ckpt.sh ~/Downloads/main.zip
set -euo pipefail
ZIP="${1:?usage: install_colab_ckpt.sh path/to/main.zip [dest_name]}"
DEST="${2:-colab_main}"
cd "$(dirname "$0")/.."
rm -rf ckpt/_colab_tmp && mkdir -p ckpt/_colab_tmp
unzip -q "$ZIP" -d ckpt/_colab_tmp
rm -rf "ckpt/$DEST" && mv ckpt/_colab_tmp/main "ckpt/$DEST" && rmdir ckpt/_colab_tmp
if cmp -s data/gaps/restoration_vocab.json "ckpt/$DEST/restoration_vocab.json"; then
  echo "OK: Colab restoration vocab is identical to the local build (datasets reproduced)"
else
  echo "WARNING: Colab restoration vocab differs from the local one; evaluate with the Colab vocab!" >&2
fi
ls -lh "ckpt/$DEST"
