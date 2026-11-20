#!/usr/bin/env python
"""Generate notebooks/train_colab.ipynb: a self-contained Colab notebook (no Drive needed).

The notebook embeds this repo's code (src/, the three scripts it needs, configs/), downloads the
public Switchboard release (OpenSLR SLR5) inside Colab, rebuilds the datasets with the same
scripts used locally, trains the sequence-level restoration BERT (disfluency-removed transcript + G2P
phones + recognized phones -> restored transcript) on the GPU and downloads a zip with the (fp16)
weights + metrics.

Run:  python scripts/make_colab_notebook.py
"""
from __future__ import annotations

import base64
import io
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ["build_processed.py", "build_seq.py", "train_seq.py"]
TAR_SHA256 = "7bbc4dd5edc82b000078a028cc0a7f705e764da17cdf5027fe7888907578b106"


def bundle() -> str:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted((ROOT / "src").rglob("*.py")):
            z.write(p, p.relative_to(ROOT))
        for s in SCRIPTS:
            z.write(ROOT / "scripts" / s, f"scripts/{s}")
        for p in sorted((ROOT / "configs").glob("*.yaml")) + [ROOT / "configs/noise_profile.json"]:
            z.write(p, p.relative_to(ROOT))                # the fitted recognizer-error profile ships with the code
    return base64.b64encode(buf.getvalue()).decode()


def code(src: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": src.splitlines(True)}


def md(src: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": src.splitlines(True)}


def main() -> None:
    b64 = bundle()
    cells = [
        md("# Disfluency restoration BERT: Colab training (self-contained)\n\n"
           "Runtime: *Runtime > Change runtime type > T4 GPU*. No Drive access is needed: the code is embedded, "
           "Switchboard is downloaded from OpenSLR, datasets are rebuilt, and the result is downloaded as a zip. "
           "Inputs per example: a window of the disfluency-removed transcript, the G2P phones of its words and the "
           "(noisy) phone sequence of what was really said; target: the disfluencies to restore before every word."),
        code("# 1) unpack the project code\n"
             "import base64, io, zipfile, os\n"
             f"B64 = '{b64}'\n"
             "os.makedirs('/content/proj', exist_ok=True)\n"
             "zipfile.ZipFile(io.BytesIO(base64.b64decode(B64))).extractall('/content/proj')\n"
             "print(sorted(os.listdir('/content/proj')))"),
        code("# 2) dependencies (torch is preinstalled on Colab; pin transformers to the version used locally)\n"
             "!pip -q install transformers==5.22.2 scikit-learn pyyaml\n"
             "import torch\n"
             "print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU: change the runtime type')"),
        code("# 3) download the free Switchboard (ISIP / Mississippi State) transcripts + lexicon from OpenSLR SLR5\n"
             "import hashlib, subprocess, os\n"
             "os.makedirs('/content/swb', exist_ok=True)\n"
             "tar = '/content/swb/switchboard_word_alignments.tar.gz'\n"
             "for base in ['https://openslr.trmal.net/resources/5', 'https://openslr.elda.org/resources/5',\n"
             "             'https://openslr.magicdatatech.com/resources/5', 'https://www.openslr.org/resources/5']:\n"
             "    r = subprocess.run(['curl', '-L', '--fail', '-sS', '-m', '1500', '-o', tar, base + '/switchboard_word_alignments.tar.gz'])\n"
             "    if r.returncode == 0 and os.path.getsize(tar) > 40_000_000:\n"
             "        print('downloaded from', base); break\n"
             "else:\n"
             "    raise SystemExit('could not download Switchboard from any mirror')\n"
             "sha = hashlib.sha256(open(tar, 'rb').read()).hexdigest()\n"
             f"assert sha == '{TAR_SHA256}', sha\n"
             "print('checksum OK', sha[:16])\n"
             "!cd /content/swb && tar -xzf switchboard_word_alignments.tar.gz && ls swb_ms98_transcriptions | head -3"),
        code("# 4) rebuild the labelled corpus and the window datasets (same scripts as the local run)\n"
             "!cd /content/proj && python -u scripts/build_processed.py --root /content/swb/swb_ms98_transcriptions --out data/processed 2>&1 | tail -25\n"
             "!cd /content/proj && python -u scripts/build_seq.py --lex /content/swb/swb_ms98_transcriptions/sw-ms98-dict.text 2>&1 | tail -12"),
        code("# 5) start training as a DETACHED background process (fp16 autocast; 3 epochs over ~150k windows of ~230 tokens;\n"
             "# expect roughly 30-60 min per epoch on a T4/L4).\n"
             "# If the page loses its connection the process keeps running; cell 5b just watches its log.\n"
             "import os, subprocess, time\n"
             "os.makedirs('/content/ckpt/main', exist_ok=True)\n"
             "LOG = '/content/ckpt/main/train.log'\n"
             "env = dict(os.environ, PYTHONUNBUFFERED='1')\n"
             "proc = subprocess.Popen(\n"
             "    ['python', '-u', 'scripts/train_seq.py', '--epochs', '3', '--batch', '32', '--device', 'cuda', '--amp', '--save',\n"
             "     '--log-every', '100', '--noise-profile', 'configs/noise_profile.json', '--out', '/content/ckpt/main'],\n"
             "    cwd='/content/proj', env=env, stdout=open(LOG, 'w'), stderr=subprocess.STDOUT, start_new_session=True)\n"
             "print('started pid', proc.pid, 'at', time.strftime('%H:%M:%S'))\n"
             "!uptime -p"),
        code("# 5b) watch the log until training finishes (re-run this cell any time after a reconnect)\n"
             "import subprocess, time\n"
             "LOG = '/content/ckpt/main/train.log'\n"
             "alive = lambda: subprocess.run(['pgrep', '-f', 'scripts/train_seq[.]py'], capture_output=True).returncode == 0\n"
             "last = ''\n"
             "while alive():\n"
             "    cur = [l for l in open(LOG).read().splitlines() if l.strip()][-1:]\n"
             "    cur = cur[0][:170] if cur else ''\n"
             "    if cur != last:\n"
             "        print(time.strftime('%H:%M:%S'), cur, flush=True); last = cur\n"
             "    time.sleep(20)\n"
             "print('training process finished')\n"
             "print(''.join(open(LOG).readlines()[-18:]))"),
        code("# 6) shrink weights to fp16, zip and download (weights + tokenizer + metrics + log)\n"
             "import torch\n"
             "sd = torch.load('/content/ckpt/main/model.pt', map_location='cpu')\n"
             "sd = {k: (v.half() if v.is_floating_point() else v) for k, v in sd.items()}\n"
             "torch.save(sd, '/content/ckpt/main/model.pt')\n"
             "!cp /content/proj/data/seq/seq_stats.json /content/ckpt/main/\n"
             "!cd /content/ckpt && zip -q -r main.zip main && ls -lh main.zip\n"
             "from google.colab import files\n"
             "files.download('/content/ckpt/main.zip')"),
        code("import json\n"
             "m = json.load(open('/content/ckpt/main/dev_metrics.json'))[-1]\n"
             "print('slot', m['slot'], '| exact-unit', m['exact'], '| macro_f1', m['macro_f1'], '| restoration acc', m['restoration_acc_on_type_matches'])\n"
             "for t, v in m['per_type'].items():\n"
             "    print(f\"{t:20s} P={v['p']:.3f} R={v['r']:.3f} F1={v['f1']:.3f} n={v['support']}\")\n"
             "print(m['word_tags'])"),
    ]
    nb = {
        "cells": cells,
        "metadata": {"accelerator": "GPU", "colab": {"provenance": []},
                     "kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"}},
        "nbformat": 4, "nbformat_minor": 5,
    }
    out = ROOT / "notebooks" / "train_colab.ipynb"
    out.write_text(json.dumps(nb, indent=1))
    print(f"wrote {out} ({out.stat().st_size / 1024:.0f} KB; embedded code {len(b64) / 1024:.0f} KB base64)")


if __name__ == "__main__":
    main()
