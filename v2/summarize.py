#!/usr/bin/env python3
"""Add paired conversation bootstrap intervals and inspectable text examples."""
from __future__ import annotations

import gzip
import hashlib
import json
import re
from collections import defaultdict

import numpy as np
from jiwer import process_words

from restoration import ROOT, save_json


def main():
    report_path = ROOT / "v2/reports/restoration_results.json"
    report = json.loads(report_path.read_text())
    bootstrap = {}
    selected = []
    for condition in ("test_clean", "test_noisy", "test_noisy_f1_decoder"):
        groups = defaultdict(lambda: np.zeros(3))
        rows = []
        with gzip.open(ROOT / "v2/reports" / f"{condition}_predictions.jsonl.gz", "rt") as f:
            for line in f:
                row = json.loads(line)
                conversation = re.match(r"sw(\d+)[AB]", row["id"]).group(1)
                base = process_words(row["reference"], row["flat_transcript"])
                hyp = process_words(row["reference"], row["restored"])
                b = base.substitutions + base.insertions + base.deletions
                h = hyp.substitutions + hyp.insertions + hyp.deletions
                groups[conversation] += [len(row["reference"].split()), b, h]
                if condition == "test_noisy":
                    rows.append((row, b, h))
        values = np.stack(list(groups.values()))
        rng = np.random.default_rng(2026)
        samples = values[rng.integers(0, len(values), size=(10000, len(values)))].sum(axis=1)
        gains = (samples[:, 1] - samples[:, 2])/samples[:, 0]
        lo, hi = np.quantile(gains, [0.025, 0.975])
        total = values.sum(axis=0)
        metrics = report["evaluation"][condition]
        assert int(total[0]) == metrics["reference_tokens"]
        assert int(total[1]) == metrics["flat_edit_errors"]
        assert int(total[2]) == metrics["restored_edit_errors"]
        bootstrap[condition] = dict(conversations=len(values), resamples=10000,
                                    wer_reduction=float((total[1]-total[2])/total[0]),
                                    paired_95_percent_ci=[float(lo), float(hi)])
        if rows:
            # Fixed, evenly spaced examples, rather than selecting only successes.
            selected = [rows[i] for i in np.linspace(0, len(rows)-1, 12).astype(int)]
    report["paired_conversation_bootstrap"] = bootstrap
    weights = ROOT / "v2/checkpoints/bert_compact/model.pt"
    report["checkpoint_sha256"] = hashlib.sha256(weights.read_bytes()).hexdigest()
    save_json(report_path, report)
    path = ROOT / "v2/reports/restoration_results.md"
    text = path.read_text().split("\n## Paired uncertainty estimates")[0] + "\n## Paired uncertainty estimates\n\n"
    text += "95% percentile bootstrap intervals resample whole held-out conversations (10,000 resamples, seed 2026). Positive reductions mean lower WER than the unchanged flat transcript. These quantify uncertainty across conversations within this labelled transcript release, not uncertainty from training seeds, annotation errors, or real recognizer output.\n\n"
    text += "| Test condition | WER reduction, percentage points | 95% CI |\n|---|---:|---:|\n"
    for name, b in bootstrap.items():
        lo, hi = b["paired_95_percent_ci"]
        text += f"| {name} | {100*b['wer_reduction']:.3f} | [{100*lo:.3f}, {100*hi:.3f}] |\n"
    path.write_text(text)
    examples = ["# Restored test examples", "",
                "Twelve evenly spaced windows from the noisy-phone test condition, using the development-WER-selected decoder. These include successes and failures. Both full references and model outputs are shown.", ""]
    for row, b, h in selected:
        examples += [f"## {row['id']}", "", f"Flat: {row['flat_transcript']}", "", f"Reference: {row['reference']}", "",
                     f"Restored: {row['restored']}", "", f"Edit errors: flat {b}; restored {h}.", ""]
    (ROOT / "v2/reports/restored_examples.md").write_text("\n".join(examples))
    print(json.dumps(bootstrap, indent=2))


if __name__ == "__main__":
    main()
