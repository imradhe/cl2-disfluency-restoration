#!/usr/bin/env python3
"""Fine-tune pretrained BERT on a Colab GPU; select/report disfluency F1 only."""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from restoration import ROOT, RestorationBert, RestorationDataset, collate, save_json
from train import evaluate, read_examples
from phones.lexicon import Lexicon


def add_pronunciations(examples, lexicon):
    cache = {}
    missing = total = 0
    for ex in examples:
        expected = []
        for word in ex["words"]:
            if word not in cache:
                pronunciations = lexicon.get(word)
                cache[word] = list(pronunciations[0]) if pronunciations else []
            expected.append(cache[word])
            total += 1
            missing += not cache[word]
        ex["expected"] = expected
    return dict(flat_words=total, words_without_dictionary_pronunciation=missing)


def f1_evaluate(model, examples, tokenizer, targets, device, batch, amp_dtype, condition="noisy", save_path=None):
    # The common evaluator preserves the exact same token/slot scoring as v2.
    # Its transcript diagnostics are omitted from this experiment's outputs.
    amp = torch.autocast("cuda", dtype=amp_dtype) if device.type == "cuda" else contextlib.nullcontext()
    with amp:
        metrics = evaluate(model, examples, tokenizer, targets, device, batch, condition, save_path)
    keys = ("condition", "windows", "conversations", "reference_tokens", "typed_restored_token", "type_only",
            "slot_detection", "per_type", "word_tags", "overflow_gold_units_counted_as_misses")
    return {k: metrics[k] for k in keys}


def write_report(report, out):
    a = report["training"]
    lines = ["# Pretrained BERT: disfluency restoration F1", "",
             f"Fine-tuned `{a['bert']}` ({a['parameters']:,} total parameters) on {a['train_windows']:,} training windows for {a['epochs']} epochs using {a['gpu']}. Pretrained encoder weights were loaded; restoration heads and new phone-token embeddings were initialized for this task.", "",
             "Input: flat words, each followed by dictionary-derived pronunciation guidance, plus the full unsegmented spoken-phone sequence in the second BERT segment. Pronunciation guidance is derived only from the flat input, not from gold alignments or missing words. The encoder learns alignment using auxiliary training-only phone-role and word-pointer losses. No gold alignment or disfluency label is an input.", "",
             "The source dataset and conversation split are identical to the compact baseline. Phones are dictionary-derived; noisy conditions simulate recognizer errors. This does not establish performance on actual recognizer output. Existing project labels include heuristic annotations. Counts are restored tokens, not disfluency episodes.", "",
             "Epoch and decoder STOP offset are selected exclusively by typed restored-token F1 on a fixed noisy development subset. All held-out gold insertions are scored, including those beyond the eight-token-per-gap decoder capacity. A true positive requires the correct gap, disfluency type, and rendered token; matching is a multiset within each gap. Original insertion order is preserved in saved text predictions but is not enforced by this F1 metric.", "",
             "## Test results", "", "| Condition | Precision | Recall | Disfluency F1 | Gold disfluent tokens |", "|---|---:|---:|---:|---:|"]
    for name, m in report["evaluation"].items():
        x = m["typed_restored_token"]
        lines.append(f"| {name} | {x['precision']:.4f} | {x['recall']:.4f} | {x['f1']:.4f} | {x['support']:,} |")
    lines += ["", "## Per-type disfluency F1", "", "| Type | Gold tokens | Clean-phone F1 | Noisy-phone precision | Recall | F1 |", "|---|---:|---:|---:|---:|---:|"]
    noisy = report["evaluation"]["test_noisy"]["per_type"]
    clean = report["evaluation"]["test_clean"]["per_type"]
    for typ, x in noisy.items():
        lines.append(f"| {typ} | {x['support']:,} | {clean.get(typ, {}).get('f1', 0):.4f} | {x['precision']:.4f} | {x['recall']:.4f} | {x['f1']:.4f} |")
    lines += ["", "Prolongation and mispronunciation word-tag scores are available separately in JSON. These labels are not inserted-token scores. Phone IDs contain no duration information, and the existing prolongation labels used corpus-wide duration statistics, so acoustic prolongation performance cannot be established here.", "",
              f"Selected epoch: {report['best_epoch']}. Selected STOP-logit offset: {report['stop_offset']:+.2f}. Runtime including evaluation: {report['runtime_seconds']/60:.1f} minutes. Checkpoint, tokenizer, model configuration, target vocabulary, pronunciation dictionary, full predictions, and development history are included with this report.", ""]
    (out / "disfluency_f1_report.md").write_text("\n".join(lines))


def save_weights(model, path):
    torch.save({k: v.detach().cpu().half() if v.is_floating_point() else v.detach().cpu() for k, v in model.state_dict().items()}, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=ROOT / "v2/data")
    ap.add_argument("--out", type=Path, default=ROOT / "v2/checkpoints/pretrained_colab")
    ap.add_argument("--bert", default="bert-base-uncased")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=0, help="0: auto-select 16/32 for available GPU memory")
    ap.add_argument("--accum", type=int, default=0, help="0: target effective batch size 32")
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--head-lr", type=float, default=3e-4)
    ap.add_argument("--dev-windows", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("A Colab GPU is required. Select Runtime > Change runtime type > GPU, then rerun.")
    torch.set_num_threads(2)
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed)
    device = torch.device("cuda")
    memory = torch.cuda.get_device_properties(0).total_memory/2**30
    a.batch = a.batch or (16 if memory < 20 else 32)
    a.accum = a.accum or max(1, 32//a.batch)
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    a.out.mkdir(parents=True, exist_ok=True)
    targets = json.loads((a.data / "target_vocab.json").read_text())
    tokenizer = AutoTokenizer.from_pretrained(a.data / "tokenizer", local_files_only=True)
    lexicon = Lexicon.from_file(a.data / "sw-ms98-dict.text")
    train = read_examples(a.data / "train.jsonl.gz")
    dev = read_examples(a.data / "dev.jsonl.gz")
    coverage = {"train": add_pronunciations(train, lexicon), "dev": add_pronunciations(dev, lexicon)}
    subset = random.Random(a.seed).sample(dev, min(a.dev_windows, len(dev)))
    ds = RestorationDataset(train, tokenizer, targets, noise_p=0.7, seed=a.seed)
    loader = DataLoader(ds, batch_size=a.batch, shuffle=True, collate_fn=collate, pin_memory=True)
    print("Loading pretrained encoder...", flush=True)
    model = RestorationBert(len(tokenizer), len(targets), pretrained=a.bert).to(device)
    body = [p for n, p in model.named_parameters() if n.startswith("bert.")]
    heads = [p for n, p in model.named_parameters() if not n.startswith("bert.")]
    optimizer = torch.optim.AdamW([{"params": body, "lr": a.lr}, {"params": heads, "lr": a.head_lr}], weight_decay=0.01)
    total = math.ceil(len(loader)/a.accum)*a.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.06*total), total)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_dtype == torch.float16)
    args = {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()}
    args.update(architecture="pretrained", gpu=torch.cuda.get_device_name(0), gpu_memory_gb=memory,
                precision=str(amp_dtype), parameters=sum(p.numel() for p in model.parameters()),
                train_windows=len(train), train_conversations=len({e["conv"] for e in train}),
                initialization="pretrained encoder; new restoration heads", pronunciation_guidance=True,
                training_noise_probability=0.7, model_selection_metric="typed_restored_token.f1")
    print(json.dumps(args, indent=2), flush=True)
    save_json(a.out / "args.json", args)
    model.bert.config.to_json_file(a.out / "bert_config.json")
    tokenizer.save_pretrained(a.out / "tokenizer")
    save_json(a.out / "target_vocab.json", targets)
    save_json(a.out / "pronunciations.json", {k: v[0] for k, v in lexicon._lower.items()})
    epoch0, step, best, history = 0, 0, -1.0, []
    if a.resume:
        state = torch.load(a.out / "latest.pt", map_location=device, weights_only=False)
        model.load_state_dict(state["model"]); optimizer.load_state_dict(state["optimizer"]); scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        epoch0, step, best, history = state["epoch"], state["step"], state["best"], state["history"]
    start, start_step = time.monotonic(), step
    for epoch in range(epoch0, a.epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        for idx, batch in enumerate(loader):
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            group_start = (idx//a.accum)*a.accum
            group_size = min(a.accum, len(loader)-group_start)
            with torch.autocast("cuda", dtype=amp_dtype):
                result = model(batch)
                loss = result["loss"]/group_size
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite training loss")
            scaler.scale(loss).backward()
            if (idx+1) % a.accum == 0 or idx+1 == len(loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer); scaler.update(); scheduler.step(); optimizer.zero_grad(set_to_none=True)
                step += 1
                if step % 100 == 0:
                    elapsed = time.monotonic()-start
                    eta = elapsed/max(1, step-start_step)*(total-step)/60
                    print(f"epoch={epoch+1} step={step}/{total} loss={float(result['loss'].detach()):.4f} eta_minutes={eta:.1f}", flush=True)
        m = f1_evaluate(model, subset, tokenizer, targets, device, a.batch, amp_dtype)
        m.update(epoch=epoch+1, step=step, elapsed_seconds=time.monotonic()-start)
        history.append(m)
        score = m["typed_restored_token"]["f1"]
        print(f"DEV epoch={epoch+1} disfluency_F1={score:.4f} P={m['typed_restored_token']['precision']:.4f} R={m['typed_restored_token']['recall']:.4f}", flush=True)
        if score > best:
            best = score
            save_weights(model, a.out / "model.pt")
            save_json(a.out / "best_dev.json", m)
        save_json(a.out / "history.json", history)
        torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(), scaler=scaler.state_dict(),
                        epoch=epoch+1, step=step, best=best, history=history), a.out / "latest.pt")
    model.load_state_dict(torch.load(a.out / "model.pt", map_location=device, weights_only=True))
    original = model.type_head.bias[0].detach().clone()
    trials = []
    for offset in [-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]:
        with torch.no_grad():
            model.type_head.bias[0].copy_(original+offset)
        metrics = f1_evaluate(model, subset, tokenizer, targets, device, a.batch, amp_dtype)
        trials.append(dict(stop_offset=offset, **metrics))
        print(f"DEV decoder_offset={offset:+.1f} disfluency_F1={metrics['typed_restored_token']['f1']:.4f}", flush=True)
    chosen = max(trials, key=lambda m: m["typed_restored_token"]["f1"])
    with torch.no_grad():
        model.type_head.bias[0].copy_(original+chosen["stop_offset"])
    save_weights(model, a.out / "model.pt")
    best_dev = json.loads((a.out / "best_dev.json").read_text())
    report = dict(training=args, dataset=json.loads((a.data / "stats.json").read_text()), coverage=coverage,
                  history=history, best_epoch=best_dev["epoch"], decoder_trials=trials,
                  stop_offset=chosen["stop_offset"], evaluation={})
    test = read_examples(a.data / "test.jsonl.gz")
    report["coverage"]["test"] = add_pronunciations(test, lexicon)
    for name, examples, condition in [("dev_noisy", dev, "noisy"), ("test_clean", test, "clean"), ("test_noisy", test, "noisy"), ("test_no_phones", test, "no_phones")]:
        m = f1_evaluate(model, examples, tokenizer, targets, device, a.batch, amp_dtype, condition, a.out / f"{name}_predictions.jsonl.gz")
        report["evaluation"][name] = m
        x = m["typed_restored_token"]
        print(f"{name}: disfluency_F1={x['f1']:.4f} P={x['precision']:.4f} R={x['recall']:.4f}", flush=True)
        save_json(a.out / "disfluency_f1_results.json", report)
    report["runtime_seconds"] = time.monotonic()-start
    save_json(a.out / "disfluency_f1_results.json", report)
    write_report(report, a.out)
    print("TRAINING_AND_TEST_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
