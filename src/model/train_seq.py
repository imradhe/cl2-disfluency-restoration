"""Train / evaluate the sequence-level restoration BERT (src/model/seq.py).

Local smoke run (MPS):
  PYTHONPATH=src .venv/bin/python -I -m model.train_seq --max-train 3000 --max-dev 300 --epochs 1 --save
Full run (Colab GPU): same command without --max-*, plus --amp --device cuda.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from phones.channel import ProfileNoise, load_profile, save_profile
from phones.noise import PhoneNoise

from .bert_seq import BertSeq
from .dataset import load_jsonl
from .seq import SEQ_SPECIAL, WTAGS, SeqDataset, collate_seq, predict_seq, slot_prf

ROOT = Path(__file__).resolve().parents[2]


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")


def build_tokenizer(name: str):
    tok = AutoTokenizer.from_pretrained(name)
    tok.add_special_tokens({"additional_special_tokens": SEQ_SPECIAL})
    return tok


def word_tag_prf(gold: list[list[int]], pred: list[list[int]]) -> dict:
    out = {}
    for c, name in enumerate(WTAGS[1:], 1):
        tp = sum(g == c and p == c for gw, pw in zip(gold, pred) for g, p in zip(gw, pw))
        n_p = sum(p == c for pw in pred for p in pw)
        n_g = sum(g == c for gw in gold for g in gw)
        pr, rc = tp / n_p if n_p else 0.0, tp / n_g if n_g else 0.0
        out[name] = {"support": n_g, "predicted": n_p, "p": round(pr, 4), "r": round(rc, 4),
                     "f1": round(2 * pr * rc / (pr + rc), 4) if pr + rc else 0.0}
    return out


def evaluate(model, examples: list[dict], tok, vocab: list[str], dev: torch.device,
             biases: tuple[float, ...] = (0.0,), noise: PhoneNoise | None = None) -> dict:
    preds = predict_seq(model, examples, tok, vocab, dev, none_biases=biases, noise=noise or PhoneNoise(seed=1))
    res = {}
    for b, pr in preds.items():
        m = slot_prf([e["slots"] for e in examples], [p["slots"] for p in pr])
        m["word_tags"] = word_tag_prf([e["wtags"] for e in examples], [p["wtags"] for p in pr])
        res[str(b)] = m
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data/seq"))
    ap.add_argument("--bert", default="bert-base-uncased")
    ap.add_argument("--out", default=str(ROOT / "ckpt/seq0"))
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--max-train", type=int, default=0)
    ap.add_argument("--max-dev", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--aux-weight", type=float, default=0.5,
                    help="weight of the auxiliary per-phone losses (role / word pointer); 0 = off (ablation)")
    ap.add_argument("--no-noise", action="store_true", help="ablation: train on clean recognized phones")
    ap.add_argument("--noise-profile", default="",
                    help="recognizer-error profile (scripts/calibrate_noise.py); empty = iid PhoneNoise only")
    ap.add_argument("--profile-p", type=float, default=0.7,
                    help="share of training windows that get the fitted channel (the rest get iid PhoneNoise)")
    ap.add_argument("--dev-profile-n", type=int, default=3000, help="dev windows also scored through the fitted channel")
    ap.add_argument("--amp", action="store_true", help="fp16 autocast + grad scaling (CUDA only)")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--save", action="store_true", help="write weights, tokenizer and unit vocab to --out")
    a = ap.parse_args()

    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    dev = pick_device(a.device)
    data = Path(a.data)
    vocab = json.loads((data / "unit_vocab.json").read_text())
    tok = build_tokenizer(a.bert)
    train_ex = load_jsonl(data / "train.jsonl.gz", 0)
    if a.max_train:
        random.Random(a.seed).shuffle(train_ex)
        train_ex = train_ex[:a.max_train]
    dev_ex = load_jsonl(data / "dev.jsonl.gz", a.max_dev)
    prof = load_profile(a.noise_profile) if a.noise_profile else None
    train_noise = None if a.no_noise else (ProfileNoise(prof, a.profile_p, seed=a.seed) if prof else PhoneNoise(seed=a.seed))
    tr = SeqDataset(train_ex, tok, vocab, noise=train_noise)
    tl = DataLoader(tr, batch_size=a.batch, shuffle=True, collate_fn=collate_seq)

    model = BertSeq(a.bert, len(tok), len(vocab), aux_weight=a.aux_weight).to(dev)
    heads = [p for n, p in model.named_parameters() if not n.startswith("bert.")]
    body = [p for n, p in model.named_parameters() if n.startswith("bert.")]
    opt = torch.optim.AdamW([{"params": body, "lr": a.lr}, {"params": heads, "lr": a.head_lr}], weight_decay=0.01)
    total = len(tl) * a.epochs
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * total), total)
    print(f"device={dev} train={len(tr)} dev={len(dev_ex)} steps={total} unit_classes={len(vocab)}", flush=True)

    use_amp = a.amp and dev.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    step, t0, history = 0, time.time(), []
    for ep in range(a.epochs):
        model.train()
        for batch in tl:
            b = {k: v.to(dev) for k, v in batch.items()}
            with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=use_amp):
                out = model(b)
            scaler.scale(out["loss"]).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step(); opt.zero_grad(set_to_none=True)
            step += 1
            if step % a.log_every == 0:
                print(f"ep{ep} step {step}/{total} loss={float(out['loss']):.3f} slot={float(out['slot_loss']):.3f} "
                      f"word={float(out['word_loss']):.3f} role={float(out['role_loss']):.3f} ptr={float(out['ptr_loss']):.3f} "
                      f"{(time.time() - t0) / step:.2f}s/step", flush=True)
        m = evaluate(model, dev_ex, tok, vocab, dev)["0.0"]
        m["epoch"] = ep + 1
        if prof:                                        # same dev windows through the fitted recognizer channel
            mp = evaluate(model, dev_ex[:a.dev_profile_n], tok, vocab, dev, noise=ProfileNoise(prof, 1.0, seed=1))["0.0"]
            m["profile_noise"] = {k: mp[k] for k in ("slot", "exact", "macro_f1", "per_type", "word_tags")}
            print(f"== dev ep{ep + 1} (fitted-channel noise, {min(len(dev_ex), a.dev_profile_n)} windows): "
                  f"slot F1={mp['slot']['f1']:.3f} (P={mp['slot']['p']:.3f} R={mp['slot']['r']:.3f}) "
                  f"exact-unit F1={mp['exact']['f1']:.3f} macroF1(types)={mp['macro_f1']:.3f}", flush=True)
        history.append(m)
        print(f"== dev ep{ep + 1}: slot F1={m['slot']['f1']:.3f} (P={m['slot']['p']:.3f} R={m['slot']['r']:.3f}) "
              f"exact-unit F1={m['exact']['f1']:.3f} macroF1(types)={m['macro_f1']:.3f} "
              f"restoration_acc={m['restoration_acc_on_type_matches']:.3f}", flush=True)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "dev_metrics.json").write_text(json.dumps(history, indent=2))
    (out / "args.json").write_text(json.dumps(vars(a), indent=2))
    if a.save:
        torch.save(model.state_dict(), out / "model.pt")
        tok.save_pretrained(out / "tokenizer")
        (out / "unit_vocab.json").write_text(json.dumps(vocab))
        if prof:
            save_profile(prof, out / "noise_profile.json")
    print("final per-type (dev):")
    for t, v in history[-1]["per_type"].items():
        print(f"  {t:20s} P={v['p']:.3f} R={v['r']:.3f} F1={v['f1']:.3f} n={v['support']}")
    print("word tags:", json.dumps(history[-1]["word_tags"]))


if __name__ == "__main__":
    main()
