#!/usr/bin/env python3
"""Package the fixed v2 split and GPU code into a Colab upload bundle/notebook."""
import hashlib
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def cell(source, kind="code"):
    result = dict(cell_type=kind, metadata={}, source=source.splitlines(True))
    if kind == "code":
        result.update(execution_count=None, outputs=[])
    return result


def main():
    folder = ROOT / "v2/colab"
    folder.mkdir(parents=True, exist_ok=True)
    bundle = folder / "switchboard_v2_colab.zip"
    paths = sorted((ROOT / "src").rglob("*.py"))
    paths += [ROOT / "v2" / name for name in ["restoration.py", "train.py", "train_colab.py", "predict.py"]]
    paths += [p for p in (ROOT / "v2/data").rglob("*") if p.is_file()]
    paths = [p for p in paths if "__pycache__" not in p.parts]
    with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as z:
        for p in paths:
            z.write(p, p.relative_to(ROOT))
        z.write(ROOT / "data/raw/switchboard_ms98/sw-ms98-dict.text", "v2/data/sw-ms98-dict.text")
    sha = hashlib.sha256(bundle.read_bytes()).hexdigest()
    save = folder / "v2_pretrained_bert_colab.ipynb"
    cells = [
        cell("# V2: pretrained BERT disfluency restoration\n\nFine-tune **bert-base-uncased** on the fixed v2 Switchboard split. **Disfluency F1** selects the checkpoint and decoder and is the reported metric.\n\nSelect **Runtime → Change runtime type → GPU** (L4 if available, otherwise T4). Run all cells. When prompted, upload `switchboard_v2_colab.zip` alongside this notebook. Three epochs over all training windows; no compact-model weights are reused. Checkpoints/logs stay in the Colab VM until downloaded; closing the page does not guarantee the VM will remain allocated.\n\nInputs: flattened transcript with dictionary pronunciation guidance + full spoken phone sequence. Phones are dictionary-derived with simulated errors, not measured recognizer output.", "markdown"),
        cell("# 1. Verify GPU\nimport torch\nassert torch.cuda.is_available(), 'Select Runtime > Change runtime type > GPU, then Run all again.'\nprint('GPU:', torch.cuda.get_device_name(0))\nprint('Memory GiB:', round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1))"),
        cell("# 2. Upload the fixed dataset and source bundle\nfrom google.colab import files\nfrom pathlib import Path\nimport hashlib, zipfile\narchive = Path('/content/switchboard_v2_colab.zip')\nif not archive.exists():\n    uploaded = files.upload()\n    assert 'switchboard_v2_colab.zip' in uploaded, 'Upload switchboard_v2_colab.zip'\n"
             f"assert hashlib.sha256(archive.read_bytes()).hexdigest() == '{sha}', 'Bundle checksum mismatch'\n"
             "PROJECT = Path('/content/cl2_v2')\nPROJECT.mkdir(exist_ok=True)\nwith zipfile.ZipFile(archive) as z:\n    z.extractall(PROJECT)\nprint('Verified bundle and unpacked fixed training/dev/test data.')"),
        cell("# 3. Install model dependencies\nimport subprocess, sys\nsubprocess.check_call([sys.executable, '-m', 'pip', '-q', 'install', 'transformers==5.22.2', 'jiwer', 'pyyaml'])\nprint('Dependencies ready.')"),
        cell("# 4. Train in a detached process and watch its log\n# Re-run this cell after a page reconnect to watch the same process.\nimport os, subprocess, sys, time, json\nPROJECT = Path('/content/cl2_v2')\nOUT = PROJECT / 'v2/checkpoints/pretrained_colab'\nOUT.mkdir(parents=True, exist_ok=True)\nLOG = OUT / 'training.log'\nPID = OUT / 'training.pid'\nDONE = OUT / 'disfluency_f1_results.json'\n\ndef alive():\n    if not PID.exists(): return False\n    try:\n        command = Path('/proc/' + PID.read_text().strip() + '/cmdline').read_bytes()\n        return b'v2/train_colab.py' in command\n    except (FileNotFoundError, ProcessLookupError):\n        return False\n\nif not DONE.exists() and not alive():\n    cmd = [sys.executable, '-u', 'v2/train_colab.py', '--epochs', '3']\n    if (OUT / 'latest.pt').exists(): cmd.append('--resume')\n    proc = subprocess.Popen(cmd, cwd=PROJECT, env=dict(os.environ, PYTHONUNBUFFERED='1', TOKENIZERS_PARALLELISM='false'),\n                            stdout=open(LOG, 'a'), stderr=subprocess.STDOUT, start_new_session=True)\n    PID.write_text(str(proc.pid))\n    print('Started GPU training, PID', proc.pid, flush=True)\nelse:\n    print('Existing run:', 'completed' if DONE.exists() and not alive() else 'running', flush=True)\nlast = ''\nwhile alive():\n    lines = LOG.read_text(errors='replace').splitlines() if LOG.exists() else []\n    current = '\\n'.join(lines[-3:])\n    if current != last:\n        print(current, flush=True); last = current\n    time.sleep(15)\nprint('\\n'.join(LOG.read_text(errors='replace').splitlines()[-20:]))\nassert DONE.exists() and (OUT / 'disfluency_f1_report.md').exists(), 'Run did not finish successfully; inspect training.log.'"),
        cell("# 5. Display disfluency F1 results\nimport json\nfrom IPython.display import display, Markdown\nresults = json.loads((OUT / 'disfluency_f1_results.json').read_text())\ndisplay(Markdown((OUT / 'disfluency_f1_report.md').read_text()))\nprint('Best epoch:', results['best_epoch'], '| STOP offset:', results['stop_offset'])"),
        cell("# 6. Download weights, predictions, metrics, and the report\nimport zipfile\nfrom google.colab import files\nresult_zip = Path('/content/v2_pretrained_bert_results.zip')\nwith zipfile.ZipFile(result_zip, 'w', zipfile.ZIP_DEFLATED) as z:\n    for p in sorted(OUT.rglob('*')):\n        if p.is_file() and p.name not in {'latest.pt', 'training.pid'}:\n            z.write(p, Path('pretrained_colab') / p.relative_to(OUT))\nprint('Result archive MiB:', round(result_zip.stat().st_size/2**20, 1))\nfiles.download(str(result_zip))"),
    ]
    notebook = dict(cells=cells, metadata=dict(accelerator="GPU", colab=dict(provenance=[]),
                    kernelspec=dict(display_name="Python 3", name="python3"), language_info=dict(name="python")), nbformat=4, nbformat_minor=5)
    save.write_text(json.dumps(notebook, indent=1))
    (folder / "bundle_manifest.json").write_text(json.dumps(dict(bundle=bundle.name, sha256=sha, bytes=bundle.stat().st_size), indent=2)+"\n")
    print(json.dumps(dict(notebook=str(save), bundle=str(bundle), bundle_mib=bundle.stat().st_size/2**20, sha256=sha), indent=2))


if __name__ == "__main__":
    main()
