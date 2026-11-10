#!/usr/bin/env python
"""Generate notebooks/train_colab.ipynb: a self-contained Colab notebook (no Drive needed).

The notebook embeds this repo's code (src/, the three scripts it needs, configs/), downloads the
public Switchboard release (OpenSLR SLR5) inside Colab, rebuilds the datasets with the same
scripts used locally, trains on the GPU and downloads a zip with the (fp16) weights + metrics.

Run:  python scripts/make_colab_notebook.py
"""
from __future__ import annotations

import base64
import io
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ["build_processed.py", "build_gaps.py", "train.py"]
TAR_SHA256 = "7bbc4dd5edc82b000078a028cc0a7f705e764da17cdf5027fe7888907578b106"


def bundle() -> str:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted((ROOT / "src").rglob("*.py")):
            z.write(p, p.relative_to(ROOT))
        for s in SCRIPTS:
            z.write(ROOT / "scripts" / s, f"scripts/{s}")
        for p in sorted((ROOT / "configs").glob("*.yaml")):
            z.write(p, p.relative_to(ROOT))
    return base64.b64encode(buf.getvalue()).decode()


def code(src: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": src.splitlines(True)}


def md(src: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": src.splitlines(True)}


def main() -> None:
    b64 = bundle()
    cells = [
        md("# Disfluency gap model: Colab training (self-contained)\n\n"
           "Runtime: *Runtime > Change runtime type > T4 GPU*. No Drive access is needed: the code is embedded, "
           "Switchboard is downloaded from OpenSLR, datasets are rebuilt, and the result is downloaded as a zip."),
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
        code("# 4) rebuild the labelled corpus and gap datasets (same scripts as the local run)\n"
             "!cd /content/proj && python -u scripts/build_processed.py --root /content/swb/swb_ms98_transcriptions --out data/processed 2>&1 | tail -25\n"
             "!cd /content/proj && python -u scripts/build_gaps.py 2>&1 | tail -12"),
        code("# 5) start training as a DETACHED background process (fp16 autocast; 3 epochs over ~262k examples).\n"
             "# If the page loses its connection the process keeps running; cell 5b just watches its log.\n"
             "import os, subprocess, time\n"
             "os.makedirs('/content/ckpt/main', exist_ok=True)\n"
             "LOG = '/content/ckpt/main/train.log'\n"
             "env = dict(os.environ, PYTHONUNBUFFERED='1')\n"
             "proc = subprocess.Popen(\n"
             "    ['python', '-u', 'scripts/train.py', '--epochs', '3', '--batch', '64', '--device', 'cuda', '--amp', '--save',\n"
             "     '--log-every', '100', '--out', '/content/ckpt/main'],\n"
             "    cwd='/content/proj', env=env, stdout=open(LOG, 'w'), stderr=subprocess.STDOUT, start_new_session=True)\n"
             "print('started pid', proc.pid, 'at', time.strftime('%H:%M:%S'))\n"
             "!uptime -p"),
        code("# 5b) watch the log until training finishes (re-run this cell any time after a reconnect)\n"
             "import subprocess, time\n"
             "LOG = '/content/ckpt/main/train.log'\n"
             "alive = lambda: subprocess.run(['pgrep', '-f', 'scripts/train[.]py'], capture_output=True).returncode == 0\n"
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
             "!cp /content/proj/data/gaps/restoration_vocab.json /content/proj/data/gaps/gap_stats.json /content/ckpt/main/\n"
             "!cd /content/ckpt && zip -q -r main.zip main && ls -lh main.zip\n"
             "from google.colab import files\n"
             "files.download('/content/ckpt/main.zip')"),
        code("import json\n"
             "m = json.load(open('/content/ckpt/main/dev_metrics.json'))[-1]\n"
             "print({k: round(m[k], 4) for k in ['unit_type_acc', 'unit_type_macro_f1', 'restoration_acc', 'restoration_acc_known_targets']})\n"
             "for t, v in m['per_class'].items():\n"
             "    print(f\"{t:20s} P={v['p']:.3f} R={v['r']:.3f} F1={v['f1']:.3f} n={v['support']}\")"),
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
