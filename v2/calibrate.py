#!/usr/bin/env python3
"""Select a decoder STOP offset on development data, then freeze and score test.

Run after train.py finishes. Saves the offset into model weights so predict.py
uses exactly the reported decoder. No test score enters model selection.
"""
from __future__ import annotations

import json
import random
import shutil
import time

import torch
from transformers import AutoTokenizer

from restoration import ROOT, RestorationBert, save_json
from train import evaluate, read_examples, write_report


def main():
    start = time.monotonic()
    torch.set_num_threads(4)
    checkpoint = ROOT / "v2/checkpoints/bert_compact"
    result_path = ROOT / "v2/reports/restoration_results.json"
    report = json.loads(result_path.read_text())
    args = report["training"]
    tokenizer = AutoTokenizer.from_pretrained(checkpoint / "tokenizer", local_files_only=True)
    targets = json.loads((checkpoint / "target_vocab.json").read_text())
    model = RestorationBert(len(tokenizer), len(targets), hidden=args["hidden"], layers=args["layers"])
    raw_weights = checkpoint / "uncalibrated_model.pt"
    if not raw_weights.exists():
        shutil.copy2(checkpoint / "model.pt", raw_weights)
    model.load_state_dict(torch.load(raw_weights, map_location="cpu", weights_only=True))
    device = torch.device("cpu")
    dev = read_examples(ROOT / "v2/data/dev.jsonl.gz")
    subset = random.Random(args["seed"]).sample(dev, min(args["dev_windows"], len(dev)))
    original = model.type_head.bias[0].detach().clone()
    trials = []
    for offset in [-1.0, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0]:
        with torch.no_grad():
            model.type_head.bias[0].copy_(original + offset)
        metrics = evaluate(model, subset, tokenizer, targets, device, args["batch"], "noisy")
        trials.append(dict(stop_logit_offset=offset, **metrics))
        print(f"dev offset={offset:.1f} F1={metrics['typed_restored_token']['f1']:.4f} WER={metrics['restored_wer']:.4f}", flush=True)
    best_f1 = max(trials, key=lambda m: (m["typed_restored_token"]["f1"], -m["restored_wer"]))
    best = min(trials, key=lambda m: (m["restored_wer"], -m["typed_restored_token"]["f1"]))
    offset = best["stop_logit_offset"]
    with torch.no_grad():
        model.type_head.bias[0].copy_(original + offset)
    torch.save(model.state_dict(), checkpoint / "model.pt")
    calibration = dict(selection="lowest transcript WER on the same fixed noisy development subset; typed restored-token F1 breaks ties",
                       selected_stop_logit_offset=offset,
                       best_f1_stop_logit_offset=best_f1["stop_logit_offset"], development_trials=trials)
    save_json(checkpoint / "decoder_calibration.json", calibration)
    save_json(ROOT / "v2/reports/decoder_calibration.json", calibration)
    uncalibrated_report = ROOT / "v2/reports/restoration_uncalibrated_results.json"
    if not uncalibrated_report.exists():
        shutil.copy2(result_path, uncalibrated_report)
    report["uncalibrated_evaluation"] = report["evaluation"]
    report["decoder_calibration"] = calibration
    report["evaluation"] = {}
    test = read_examples(ROOT / "v2/data/test.jsonl.gz")
    for name, examples, condition in [("dev_noisy", dev, "noisy"), ("test_clean", test, "clean"), ("test_noisy", test, "noisy"), ("test_no_phones", test, "no_phones")]:
        prediction_path = ROOT / "v2/reports" / f"{name}_predictions.jsonl.gz"
        metrics = evaluate(model, examples, tokenizer, targets, device, args["batch"], condition, prediction_path)
        report["evaluation"][name] = metrics
        print(f"{name}: F1={metrics['typed_restored_token']['f1']:.4f} WER={metrics['restored_wer']:.4f} baseline={metrics['flat_wer']:.4f}", flush=True)
    with torch.no_grad():
        model.type_head.bias[0].copy_(original + best_f1["stop_logit_offset"])
    torch.save(model.state_dict(), checkpoint / "model_f1_decoder.pt")
    report["evaluation"]["test_noisy_f1_decoder"] = evaluate(
        model, test, tokenizer, targets, device, args["batch"], "noisy",
        ROOT / "v2/reports/test_noisy_f1_decoder_predictions.jsonl.gz")
    report["runtime_seconds"] += time.monotonic()-start
    save_json(result_path, report)
    write_report(report)
    report_md = ROOT / "v2/reports/restoration_results.md"
    text = report_md.read_text()
    text += ("\n## Decoder calibration\n\n"
             f"After epoch selection, the NONE/STOP class received a logit offset of {offset:+.1f}, selected solely by minimum transcript WER on the same fixed development subset. "
             f"A second decoder, with offset {best_f1['stop_logit_offset']:+.1f}, was selected by maximum typed restored-token F1 and is reported as `test_noisy_f1_decoder`. "
             "The candidate offsets were -1, 0, 0.5, 1, 1.5, 2, and 3. Each offset was frozen before its test evaluation. The default model weights use the minimum-WER decoder; `model_f1_decoder.pt` stores the F1 decoder. "
             "Both raw and calibrated evaluations are retained in the JSON results. Calibration does not retrain the encoder.\n")
    text += "\nPer-type restored-token F1 at the two development-selected operating points:\n\n| Type | Test gold tokens | WER-selected decoder F1 | F1-selected decoder F1 |\n|---|---:|---:|---:|\n"
    for typ, metric in report["evaluation"]["test_noisy"]["per_type"].items():
        other = report["evaluation"]["test_noisy_f1_decoder"]["per_type"].get(typ, {"f1": 0.0})
        text += f"| {typ} | {metric['support']:,} | {metric['f1']:.4f} | {other['f1']:.4f} |\n"
    text = text.replace("## Held-out results", "The default saved model uses the development-WER-selected decoder. A separate F1-selected decoder is reported alongside it; both use the same trained encoder.\n\n## Held-out results")
    report_md.write_text(text)


if __name__ == "__main__":
    main()
