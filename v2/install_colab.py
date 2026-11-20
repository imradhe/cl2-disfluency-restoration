#!/usr/bin/env python3
"""Install the completed Colab archive and compare held-out disfluency F1."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import shutil
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from restoration import ROOT, render_target, save_json


def counts_by_conversation(path):
    grouped = defaultdict(lambda: np.zeros(3, dtype=np.int64))
    n = 0
    with gzip.open(path, "rt") as f:
        for line in f:
            row = json.loads(line)
            conv = re.match(r"sw(\d+)[AB]", row["id"]).group(1)
            words = row["flat_transcript"].split()
            for i, (gold, pred) in enumerate(zip(row["gold_slots"], row["predicted_slots"])):
                if gold is None:
                    continue
                def counter(units):
                    return Counter((u.split("|", 1)[0], render_target(u.split("|", 1)[1], words[:i], words[i:])) for u in units)
                g, p = counter(gold), counter(pred)
                tp = sum(v for key, v in (g & p).items() if key[1] != "<unk>")
                grouped[conv] += [tp, len(pred), len(gold)]
            n += 1
    return grouped, n


def micro_f1(counts):
    denominator = counts[..., 1]+counts[..., 2]
    return np.divide(2*counts[..., 0], denominator, out=np.zeros_like(denominator, dtype=float), where=denominator != 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("archive", type=Path)
    ap.add_argument("--dest", type=Path, default=ROOT / "v2/checkpoints/pretrained_colab")
    a = ap.parse_args()
    if a.dest.exists():
        raise SystemExit(f"Destination already exists: {a.dest}. Choose a new --dest to preserve it.")
    a.dest.mkdir(parents=True)
    with zipfile.ZipFile(a.archive) as z:
        assert z.testzip() is None, "Corrupt result archive"
        for name in z.namelist():
            parts = Path(name).parts
            assert parts[0] == "pretrained_colab" and ".." not in parts, name
            relative = Path(*parts[1:])
            dest = a.dest / relative
            if name.endswith("/"):
                dest.mkdir(parents=True, exist_ok=True)
            else:
                dest.parent.mkdir(parents=True, exist_ok=True)
                with z.open(name) as source, dest.open("wb") as target:
                    shutil.copyfileobj(source, target)
    report = json.loads((a.dest / "disfluency_f1_results.json").read_text())
    config = json.loads((a.dest / "bert_config.json").read_text())
    stats = json.loads((ROOT / "v2/data/stats.json").read_text())
    assert report["training"]["architecture"] == "pretrained"
    assert config["num_hidden_layers"] == 12 and config["hidden_size"] == 768
    assert report["training"]["train_windows"] == stats["splits"]["train"]["windows"]
    bootstrap, groups = {}, {}
    for name in ("test_clean", "test_noisy", "test_no_phones"):
        grouped, n = counts_by_conversation(a.dest / f"{name}_predictions.jsonl.gz")
        metrics = report["evaluation"][name]
        assert n == metrics["windows"] == stats["splits"]["test"]["windows"]
        total = np.stack(list(grouped.values())).sum(axis=0)
        counts = metrics["typed_restored_token"]
        assert list(total) == [counts["tp"], counts["predicted"], counts["support"]]
        assert abs(float(micro_f1(total))-counts["f1"]) < 1e-12
        values = np.stack([grouped[k] for k in sorted(grouped)])
        rng = np.random.default_rng(2026)
        indices = rng.integers(0, len(values), (10000, len(values)))
        samples = micro_f1(values[indices].sum(axis=1))
        bootstrap[name] = dict(f1=counts["f1"], paired_95_percent_ci=np.quantile(samples, [0.025, 0.975]).tolist(),
                               conversations=len(grouped), resamples=10000)
        groups[name] = grouped
    previous, _ = counts_by_conversation(ROOT / "v2/reports/test_noisy_f1_decoder_predictions.jsonl.gz")
    current = groups["test_noisy"]
    assert set(previous) == set(current)
    keys = sorted(current)
    old, new = np.stack([previous[k] for k in keys]), np.stack([current[k] for k in keys])
    assert np.array_equal(old[:, 2], new[:, 2])
    indices = np.random.default_rng(2026).integers(0, len(keys), (10000, len(keys)))
    delta = micro_f1(new[indices].sum(axis=1))-micro_f1(old[indices].sum(axis=1))
    comparison = dict(compact_baseline_f1=float(micro_f1(old.sum(axis=0))), pretrained_f1=float(micro_f1(new.sum(axis=0))),
                      difference=float(micro_f1(new.sum(axis=0))-micro_f1(old.sum(axis=0))),
                      paired_95_percent_ci=np.quantile(delta, [0.025, 0.975]).tolist())
    report.update(conversation_bootstrap=bootstrap, compact_baseline_comparison=comparison,
                  checkpoint_sha256=hashlib.sha256((a.dest / "model.pt").read_bytes()).hexdigest(),
                  result_archive_sha256=hashlib.sha256(a.archive.read_bytes()).hexdigest())
    save_json(ROOT / "v2/reports/pretrained_colab_f1.json", report)
    text = (a.dest / "disfluency_f1_report.md").read_text()
    text = text.replace("## Test results", "Overall disfluency F1 is micro-averaged over inserted tokens; it excludes unchanged fluent words.\n\n## Test results")
    text += "\n## Word-tag disfluency scores\n\nThese scores classify existing words and are separate from insertion-restoration F1.\n\n| Type | Test gold tokens | Precision | Recall | F1 |\n|---|---:|---:|---:|---:|\n"
    for typ, x in report["evaluation"]["test_noisy"]["word_tags"].items():
        text += f"| {typ} | {x['support']:,} | {x['precision']:.4f} | {x['recall']:.4f} | {x['f1']:.4f} |\n"
    text += "\n## Comparison with the compact baseline\n\nBoth rows below use decoder thresholds selected by noisy-development disfluency F1 and evaluate the same held-out noisy-phone windows. The new run changes both encoder initialization/size and pronunciation guidance; the comparison does not isolate those factors.\n\n"
    text += "| Model | Test disfluency F1 |\n|---|---:|\n"
    text += f"| Compact model trained from scratch | {comparison['compact_baseline_f1']:.4f} |\n| Pretrained BERT with pronunciation guidance | {comparison['pretrained_f1']:.4f} |\n"
    lo, hi = comparison["paired_95_percent_ci"]
    text += f"\nPaired F1 difference: {comparison['difference']:+.4f}; 95% conversation-bootstrap CI [{lo:+.4f}, {hi:+.4f}]. Bootstrap uses 10,000 resamples of whole conversations, seed 2026. It does not capture training-seed or annotation uncertainty.\n"
    text += "\n| Condition | Disfluency F1 | 95% conversation-bootstrap CI |\n|---|---:|---:|\n"
    for name, b in bootstrap.items():
        lo, hi = b["paired_95_percent_ci"]
        text += f"| {name} | {b['f1']:.4f} | [{lo:.4f}, {hi:.4f}] |\n"
    (ROOT / "v2/reports/pretrained_colab_f1.md").write_text(text)
    chosen = set(np.linspace(0, report["evaluation"]["test_noisy"]["windows"]-1, 12).astype(int).tolist())
    examples = ["# Pretrained BERT: restored test examples", "",
                "Twelve evenly spaced noisy-phone test windows. Examples are selected by index, not by whether restoration succeeded.", ""]
    transcripts = defaultdict(lambda: dict(flat=[], reference=[], restored=[], windows=0))
    with gzip.open(a.dest / "test_noisy_predictions.jsonl.gz", "rt") as f:
        for i, line in enumerate(f):
            row = json.loads(line)
            side = transcripts[row["id"].split(":", 1)[0]]
            side["flat"].append(row["flat_transcript"])
            side["reference"].append(row["reference"])
            side["restored"].append(row["restored"])
            side["windows"] += 1
            if i in chosen:
                examples += [f"## {row['id']}", "", f"Flat: {row['flat_transcript']}", "",
                             f"Reference: {row['reference']}", "", f"Restored: {row['restored']}", ""]
    (ROOT / "v2/reports/pretrained_colab_examples.md").write_text("\n".join(examples))
    with gzip.open(ROOT / "v2/reports/pretrained_restored_transcripts.jsonl.gz", "wt") as f:
        for sid, item in transcripts.items():
            f.write(json.dumps(dict(sid=sid, windows=item["windows"],
                                    flat_transcript=" ".join(item["flat"]), reference=" ".join(item["reference"]),
                                    restored_transcript=" ".join(item["restored"])))+"\n")
    status = ROOT / "v2/colab/run.json"
    run = json.loads(status.read_text())
    run.update(status="completed_and_downloaded", checkpoint=str(a.dest), best_epoch=report["best_epoch"],
               test_clean_disfluency_f1=report["evaluation"]["test_clean"]["typed_restored_token"]["f1"],
               test_noisy_disfluency_f1=report["evaluation"]["test_noisy"]["typed_restored_token"]["f1"],
               checkpoint_sha256=report["checkpoint_sha256"])
    save_json(status, run)
    print(json.dumps(dict(comparison=comparison, bootstrap=bootstrap, checkpoint=str(a.dest)), indent=2))


if __name__ == "__main__":
    main()
