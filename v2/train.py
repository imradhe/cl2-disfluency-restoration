#!/usr/bin/env python3
"""Train a fresh compact BERT and evaluate restored text on held-out conversations."""
from __future__ import annotations

import argparse
import gzip
import json
import math
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from jiwer import process_words
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from restoration import (K, ROOT, TYPES, WTAGS, RestorationBert, RestorationDataset,
                         collate, render_target, restore, save_json)


def read_examples(path):
    with gzip.open(path, "rt") as f:
        return [json.loads(line) for line in f]


def prf(tp, predicted, support):
    p, r = tp/predicted if predicted else 0.0, tp/support if support else 0.0
    return dict(tp=tp, predicted=predicted, support=support, precision=p, recall=r,
                f1=2*p*r/(p+r) if p+r else 0.0)


@torch.no_grad()
def evaluate(model, examples, tokenizer, targets, device, batch_size=64, condition="clean", save_path=None):
    model.eval()
    if condition == "no_phones":
        examples_in = [{**ex, "phones": [], "roles": [], "pointers": []} for ex in examples]
    else:
        examples_in = examples
    ds = RestorationDataset(examples_in, tokenizer, targets, noise_p=1.0 if condition == "noisy" else 0.0, seed=2026)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate)
    counts = defaultdict(Counter)
    type_counts = defaultdict(Counter)
    word_counts = defaultdict(Counter)
    reference_n = baseline_errors = restored_errors = exact_windows = baseline_exact = slots_tp = slots_p = slots_g = overflow = 0
    processed = 0
    saved = gzip.open(save_path, "wt") if save_path else None
    try:
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(batch, loss=False)
            labels, surfaces = out["predicted_types"].cpu().tolist(), out["predicted_targets"].cpu().tolist()
            wtags = out["word_logits"].argmax(-1).cpu().tolist()
            si = wi = 0
            # batch_size examples; the final minibatch can be shorter.
            for ex in examples[processed:processed+len(batch["input_ids"])]:
                ns, nw = len(ex["slots"]), len(ex["words"])
                prediction = []
                for i in range(ns):
                    units = []
                    for typ, target in zip(labels[si+i], surfaces[si+i]):
                        if typ == 0:
                            break
                        units.append(TYPES[typ] + "|" + targets[target])
                    prediction.append(units)
                for s, (gold, pred) in enumerate(zip(ex["slots"], prediction)):
                    if gold is None:
                        # Boundary gaps belong to the following window.
                        prediction[s] = []
                        continue
                    overflow += max(0, len(gold)-K)
                    gl = Counter(u.split("|", 1)[0] for u in gold)
                    pl = Counter(u.split("|", 1)[0] for u in pred)
                    gs = Counter((u.split("|", 1)[0], render_target(u.split("|", 1)[1], ex["words"][:s], ex["words"][s:])) for u in gold)
                    ps = Counter((u.split("|", 1)[0], render_target(u.split("|", 1)[1], ex["words"][:s], ex["words"][s:])) for u in pred)
                    for typ in set(gl) | set(pl):
                        type_counts[typ].update(tp=min(gl[typ], pl[typ]), predicted=pl[typ], support=gl[typ])
                        correct = sum(min(v, ps[key]) for key, v in gs.items() if key[0] == typ and key[1] != "<unk>")
                        counts[typ].update(tp=correct, predicted=pl[typ], support=gl[typ])
                    slots_tp += bool(gold) and bool(pred)
                    slots_p += bool(pred)
                    slots_g += bool(gold)
                for typ in (1, 2):
                    g = [i for i, x in enumerate(ex["wtags"]) if x == typ]
                    p = [i for i, x in enumerate(wtags[wi:wi+nw]) if x == typ]
                    word_counts[WTAGS[typ]].update(tp=len(set(g)&set(p)), predicted=len(p), support=len(g))
                restored = restore(ex["words"], prediction)
                ref, base, hyp = " ".join(ex["reference"]), " ".join(ex["words"]), " ".join(restored)
                baseline = process_words(ref, base)
                result = process_words(ref, hyp)
                reference_n += len(ex["reference"])
                baseline_errors += baseline.substitutions + baseline.deletions + baseline.insertions
                restored_errors += result.substitutions + result.deletions + result.insertions
                exact_windows += hyp == ref
                baseline_exact += base == ref
                if saved:
                    saved.write(json.dumps(dict(id=ex["id"], flat_transcript=base, phones=examples_in[processed]["phones"] if condition != "noisy" else None,
                                                reference=ref, restored=hyp, predicted_slots=prediction, gold_slots=ex["slots"])) + "\n")
                si, wi, processed = si+ns, wi+nw, processed+1
    finally:
        if saved:
            saved.close()
    def aggregate(counters):
        c = sum(counters.values(), Counter())
        return prf(c["tp"], c["predicted"], c["support"])
    return dict(condition=condition, windows=len(examples), conversations=len({e["conv"] for e in examples}),
                reference_tokens=reference_n, flat_wer=baseline_errors/reference_n,
                restored_wer=restored_errors/reference_n, flat_edit_errors=baseline_errors, restored_edit_errors=restored_errors,
                window_exact_match=exact_windows/len(examples), flat_window_exact_match=baseline_exact/len(examples),
                typed_restored_token=aggregate(counts), type_only=aggregate(type_counts),
                slot_detection=prf(slots_tp, slots_p, slots_g),
                per_type={typ: prf(c["tp"], c["predicted"], c["support"]) for typ, c in sorted(counts.items())},
                word_tags={typ: prf(c["tp"], c["predicted"], c["support"]) for typ, c in word_counts.items()},
                overflow_gold_units_counted_as_misses=overflow)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=ROOT / "v2/data")
    ap.add_argument("--out", type=Path, default=ROOT / "v2/checkpoints/bert_compact")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--lr", type=float, default=0.0005)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--dev-windows", type=int, default=1000)
    ap.add_argument("--benchmark-steps", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    a.out.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(a.data / "tokenizer", local_files_only=True)
    targets = json.loads((a.data / "target_vocab.json").read_text())
    print("Loading data...", flush=True)
    train = read_examples(a.data / "train.jsonl.gz")
    dev = read_examples(a.data / "dev.jsonl.gz")
    subset = random.Random(a.seed).sample(dev, min(a.dev_windows, len(dev)))
    ds = RestorationDataset(train, tokenizer, targets, noise_p=0.7, seed=a.seed)
    loader = DataLoader(ds, batch_size=a.batch, shuffle=True, collate_fn=collate)
    model = RestorationBert(len(tokenizer), len(targets), hidden=a.hidden, layers=a.layers).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    total = len(loader)*a.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.05*total), total)
    args = {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()}
    args.update(device=str(device), parameters=sum(p.numel() for p in model.parameters()),
                initialization="random; BERT WordPiece tokenizer only is reused", train_windows=len(train),
                train_conversations=len({e["conv"] for e in train}), training_noise_probability=0.7)
    print(json.dumps(args, indent=2), flush=True)
    start_ep, step, history, best = 0, 0, [], -1.0
    if a.resume:
        state = torch.load(a.out / "latest.pt", map_location=device, weights_only=False)
        model.load_state_dict(state["model"]); optimizer.load_state_dict(state["optimizer"]); scheduler.load_state_dict(state["scheduler"])
        start_ep, step, history, best = state["epoch"], state["step"], state["history"], state["best"]
    if not a.benchmark_steps:
        save_json(a.out / "args.json", args)
        model.bert.config.to_json_file(a.out / "bert_config.json")
        tokenizer.save_pretrained(a.out / "tokenizer")
        save_json(a.out / "target_vocab.json", targets)
    start = time.monotonic()
    for epoch in range(start_ep, a.epochs):
        model.train()
        losses, nloss = Counter(), 0
        for batch in loader:
            b = {k: v.to(device) for k, v in batch.items()}
            out = model(b)
            if not torch.isfinite(out["loss"]):
                raise RuntimeError("Nonfinite loss")
            out["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True)
            step += 1
            nloss += 1
            for name in ("loss", "type_loss", "target_loss", "role_loss", "pointer_loss"):
                losses[name] += float(out[name].detach())
            if step % 100 == 0:
                print(f"epoch={epoch+1} step={step}/{total} loss={losses['loss']/nloss:.4f} seconds/step={(time.monotonic()-start)/(step - (state['step'] if a.resume else 0)):.3f}", flush=True)
            if a.benchmark_steps and step >= a.benchmark_steps:
                elapsed = time.monotonic()-start
                print(json.dumps(dict(benchmark_steps=step, seconds=elapsed, seconds_per_step=elapsed/step,
                                      estimated_train_minutes=elapsed/step*total/60)), flush=True)
                return
        metrics = evaluate(model, subset, tokenizer, targets, device, a.batch, condition="noisy")
        metrics.update(epoch=epoch+1, step=step, mean_losses={k: v/nloss for k, v in losses.items()}, elapsed_seconds=time.monotonic()-start)
        history.append(metrics)
        score = metrics["typed_restored_token"]["f1"]
        print(f"DEV epoch={epoch+1} typed_token_f1={score:.4f} restored_wer={metrics['restored_wer']:.4f} flat_wer={metrics['flat_wer']:.4f}", flush=True)
        if score > best:
            best = score
            torch.save(model.state_dict(), a.out / "model.pt")
            save_json(a.out / "best_dev.json", metrics)
        save_json(a.out / "history.json", history)
        torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                        epoch=epoch+1, step=step, history=history, best=best), a.out / "latest.pt")

    model.load_state_dict(torch.load(a.out / "model.pt", map_location=device, weights_only=True))
    test = read_examples(a.data / "test.jsonl.gz")
    report = dict(training=args, best_dev=json.loads((a.out / "best_dev.json").read_text()),
                  dataset=json.loads((a.data / "stats.json").read_text()), history=history, evaluation={})
    for name, examples, condition in [("dev_noisy", dev, "noisy"), ("test_clean", test, "clean"), ("test_noisy", test, "noisy"), ("test_no_phones", test, "no_phones")]:
        predictions = ROOT / "v2/reports" / f"{name}_predictions.jsonl.gz"
        result = evaluate(model, examples, tokenizer, targets, device, a.batch, condition, predictions)
        report["evaluation"][name] = result
        save_json(ROOT / "v2/reports/restoration_results.json", report)
        print(f"{name}: F1={result['typed_restored_token']['f1']:.4f} WER={result['restored_wer']:.4f} baseline={result['flat_wer']:.4f}", flush=True)
    report["runtime_seconds"] = time.monotonic()-start
    save_json(ROOT / "v2/reports/restoration_results.json", report)
    write_report(report)


def write_report(report):
    a = report["training"]
    lines = ["# V2 BERT disfluency restoration results", "",
             f"A fresh, compact BERT encoder ({a['layers']} layers, hidden size {a['hidden']}, {a['parameters']:,} parameters) was trained for {a['epochs']} epochs on {a['train_windows']:,} windows from {a['train_conversations']:,} Switchboard conversations. All model weights were randomly initialized; the cached BERT WordPiece tokenizer was reused. This is a BERT architecture trained for this task, not fine-tuning of pretrained BERT-base.", "",
             "## Task and data", "",
             "Input: `[CLS] flat transcript [END] [SEP] full phone sequence [SEP]`. No word-to-phone boundaries, gold slot/type labels, or auxiliary pointers enter the encoder. The flat transcript has every insertion-type disfluency removed. Phones retain the removed words. The model predicts insertion types and surface words/copy pointers at each transcript gap; rendering those insertions produces a restored transcript. The word tagger also predicts prolongation and mispronunciation labels.", "",
             "Phone sequences come from the MS98 pronunciation dictionary, not an audio recognizer. 70% of training windows receive simulated phone corruption (8% deletion; 14% substitution, using the existing simulator's mutually exclusive probabilities; 4% insertion). Each noisy evaluation uses a fixed random seed. Clean phones are an oracle condition. Removing phones at test time is an input ablation of the same trained model, not a separately trained text-only baseline.", "",
             "Train/dev/test split by conversation ID: test 4010–4153; dev 4519–4936; train remainder. Target vocabulary is built from training only. The best epoch is selected using typed restored-token F1 on a fixed random subset of development windows with simulated noise. Test conversations are first evaluated after training and model selection.", "",
             "| Split | Conversations | Windows | Flat words | Reference words |", "|---|---:|---:|---:|---:|"]
    for name, s in report["dataset"]["splits"].items():
        lines.append(f"| {name} | {s['conversations']:,} | {s['windows']:,} | {s['flat_words']:,} | {s['reference_words']:,} |")
    lines += ["", "## Held-out results", "", "| Condition | Typed restored-token precision | Recall | F1 | Restored WER | Flat baseline WER | Exact windows |", "|---|---:|---:|---:|---:|---:|---:|"]
    for name, m in report["evaluation"].items():
        f = m["typed_restored_token"]
        lines.append(f"| {name} | {f['precision']:.4f} | {f['recall']:.4f} | {f['f1']:.4f} | {m['restored_wer']:.4f} | {m['flat_wer']:.4f} | {m['window_exact_match']:.4f} |")
    lines += ["", "## Per-type test results with simulated phone noise", "", "| Type | Gold tokens | Predicted tokens | Precision | Recall | F1 |", "|---|---:|---:|---:|---:|---:|"]
    m = report["evaluation"]["test_noisy"]
    for typ, f in m["per_type"].items():
        lines.append(f"| {typ} | {f['support']:,} | {f['predicted']:,} | {f['precision']:.4f} | {f['recall']:.4f} | {f['f1']:.4f} |")
    lines += ["", "Prolongation and mispronunciation word-tag results:", "", "| Type | Gold tokens | Precision | Recall | F1 |", "|---|---:|---:|---:|---:|"]
    for typ, f in m["word_tags"].items():
        lines.append(f"| {typ} | {f['support']:,} | {f['precision']:.4f} | {f['recall']:.4f} | {f['f1']:.4f} |")
    lines += ["", "## Interpretation and limits", "",
              "Typed restored-token scoring requires the insertion gap, type, and rendered token to match, using multiset matching within a gap; it does not enforce the ordering of multiple insertions within the same gap. Corpus WER and exact-window match assess the actual restored text, including insertion order. WER aggregates edit counts over all windows, rather than averaging per-window WER. Disfluency F1 counts only inserted disfluencies; it excludes unchanged fluent words.", "",
              f"The decoder supports at most {K} inserted tokens per gap. All gold tokens, including overflow, remain in scoring. Test overflow: {m['overflow_gold_units_counted_as_misses']:,} tokens. Rare unseen surface targets map to UNK at training/inference but are scored against their actual held-out surfaces. Word-internal prolongation is reported as a tag; phone identities without durations do not identify acoustic lengthening. Mispronunciation tags do not themselves replace a word's surface, so any unresolved spelling substitutions still count as transcript errors.", "",
              "Training and evaluation use the existing project's labels, including inferred repetitions/repairs/false starts/discourse markers. They measure agreement with those rules, not independently validated human judgements. Existing duration-based labels were derived before this split and used corpus-wide word-duration statistics; prolongation tag results therefore are not a strictly train-only label-generation evaluation. Conversation separation does not guarantee unique-speaker separation because speaker metadata are unavailable here.", "",
              "Nonspeech noise and laughter are excluded from text restoration; no fabricated noise phones are added. Speech through laughter remains ordinary speech. The reference preserves partial-word fragments with edge hyphens, rather than replacing them with their intended complete words. Oversize windows and sides lacking any flat words are excluded and counted in `data/stats.json`. Window boundaries use input lengths; no phone stream is silently truncated.", "",
              f"Device: {a['device']}; measured run time including final evaluation: {report['runtime_seconds']/60:.2f} minutes. Weights: `v2/checkpoints/bert_compact/model.pt`; training history/config and target vocabulary are saved alongside them. Complete predictions and machine-readable results are in `v2/reports/`.", ""]
    (ROOT / "v2/reports/restoration_results.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
