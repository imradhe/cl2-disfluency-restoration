"""Train / evaluate the BERT gap model.

Local smoke run (MPS):
  PYTHONPATH=src .venv/bin/python -I -m model.train --max-train 4000 --max-dev 1000 --epochs 1
Full run (Colab GPU): same command without --max-*.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from phones.noise import PhoneNoise

from .bert_gap import BertGap
from .dataset import IGNORE, SPECIAL_TOKENS, TAG_TYPES, TAGS, GapDataset, collate, load_jsonl

ROOT = Path(__file__).resolve().parents[2]


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def to_device(batch: dict, dev: torch.device) -> dict:
    return {k: v.to(dev) for k, v in batch.items()}


def build_tokenizer(name: str):
    tok = AutoTokenizer.from_pretrained(name)
    tok.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})
    return tok


@torch.no_grad()
def evaluate(model: BertGap, loader: DataLoader, dev: torch.device, rest_vocab: list[str]) -> dict:
    model.eval()
    unit_gold, unit_pred = [], []        # unit-level type labels
    rest_hit, rest_n, rest_hit_known, rest_n_known = 0, 0, 0, 0
    per_label_rest: dict[str, list[int]] = {}
    unk = rest_vocab.index("UNK")
    loss_sum, n_batches = 0.0, 0
    for batch in loader:
        b = to_device(batch, dev)
        out = model(b)
        loss_sum += float(out["loss"])
        n_batches += 1
        pred_tag = out["tag_logits"].argmax(-1).cpu()
        tags = batch["tags"]
        ub, us, ue = batch["unit_batch"], batch["unit_start"], batch["unit_end"]
        for k in range(len(ub)):
            g = tags[ub[k], us[k]:ue[k]]
            p = pred_tag[ub[k], us[k]:ue[k]]
            gold_t = TAGS[int(g[0])].split("-", 1)[-1] if int(g[0]) != 0 else "O"
            counts = np.bincount([TAG_TYPES.index(TAGS[int(x)].split("-", 1)[-1]) if int(x) != 0 else 0 for x in p],
                                 minlength=len(TAG_TYPES))
            unit_gold.append(gold_t)
            unit_pred.append(TAG_TYPES[int(counts.argmax())])
        if len(ub):
            pr = out["rest_logits"].argmax(-1).cpu()
            gt = batch["unit_target"]
            hit = (pr == gt)
            rest_hit += int(hit.sum()); rest_n += len(gt)
            known = gt != unk
            rest_hit_known += int((hit & known).sum()); rest_n_known += int(known.sum())
            for k in range(len(gt)):
                lab = unit_gold[-len(gt) + k]
                s = per_label_rest.setdefault(lab, [0, 0])
                s[0] += int(hit[k]); s[1] += 1
    p, r, f, sup = precision_recall_fscore_support(unit_gold, unit_pred, labels=TAG_TYPES, zero_division=0)
    per_class = {t: {"p": round(float(p[i]), 4), "r": round(float(r[i]), 4), "f1": round(float(f[i]), 4),
                     "support": int(sup[i])} for i, t in enumerate(TAG_TYPES)}
    present = [t for t in TAG_TYPES if per_class[t]["support"] > 0]
    return {
        "loss": loss_sum / max(1, n_batches),
        "unit_type_acc": float(np.mean([g == q for g, q in zip(unit_gold, unit_pred)])) if unit_gold else 0.0,
        "unit_type_macro_f1": float(np.mean([per_class[t]["f1"] for t in present])) if present else 0.0,
        "restoration_acc": rest_hit / max(1, rest_n),
        "restoration_acc_known_targets": rest_hit_known / max(1, rest_n_known),
        "restoration_acc_by_label": {k: round(v[0] / max(1, v[1]), 4) for k, v in per_label_rest.items()},
        "per_class": per_class,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data/gaps"))
    ap.add_argument("--bert", default="bert-base-uncased")
    ap.add_argument("--out", default=str(ROOT / "ckpt/run0"))
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--max-train", type=int, default=0)
    ap.add_argument("--max-dev", type=int, default=0)
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--no-noise", action="store_true", help="ablation: train on clean phones")
    ap.add_argument("--no-phones", action="store_true", help="ablation: text-only BERT")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--save", action="store_true", help="write model weights to --out")
    a = ap.parse_args()

    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    dev = pick_device(a.device)
    data = Path(a.data)
    rest_vocab = json.loads((data / "restoration_vocab.json").read_text())
    tok = build_tokenizer(a.bert)

    train_ex = load_jsonl(data / "train.jsonl.gz", a.max_train)
    dev_ex = load_jsonl(data / "dev.jsonl.gz", a.max_dev)
    if a.max_train:
        random.Random(a.seed).shuffle(train_ex)
    noise = None if a.no_noise else PhoneNoise(seed=a.seed)
    tr = GapDataset(train_ex, tok, rest_vocab, a.max_len, noise=noise, use_phones=not a.no_phones, seed=a.seed)
    dv = GapDataset(dev_ex, tok, rest_vocab, a.max_len, noise=PhoneNoise(seed=1), use_phones=not a.no_phones)
    tl = DataLoader(tr, batch_size=a.batch, shuffle=True, collate_fn=collate)
    dl = DataLoader(dv, batch_size=a.batch * 2, shuffle=False, collate_fn=collate)

    model = BertGap(a.bert, len(tok), len(TAGS), len(rest_vocab)).to(dev)
    heads = [p for n, p in model.named_parameters() if not n.startswith("bert.")]
    body = [p for n, p in model.named_parameters() if n.startswith("bert.")]
    opt = torch.optim.AdamW([{"params": body, "lr": a.lr}, {"params": heads, "lr": a.head_lr}], weight_decay=0.01)
    total = len(tl) * a.epochs
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * total), total)
    print(f"device={dev} train={len(tr)} dev={len(dv)} steps={total} tags={len(TAGS)} rest_classes={len(rest_vocab)}", flush=True)

    step, t0 = 0, time.time()
    history = []
    for ep in range(a.epochs):
        model.train()
        for batch in tl:
            b = to_device(batch, dev)
            out = model(b)
            out["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            step += 1
            if step % a.log_every == 0:
                print(f"ep{ep} step {step}/{total} loss={float(out['loss']):.3f} "
                      f"tag={float(out['tag_loss']):.3f} rest={float(out['rest_loss']):.3f} "
                      f"{(time.time() - t0) / step:.2f}s/step", flush=True)
        m = evaluate(model, dl, dev, rest_vocab)
        m["epoch"] = ep + 1
        history.append(m)
        print(f"== dev ep{ep + 1}: loss={m['loss']:.3f} type_acc={m['unit_type_acc']:.3f} "
              f"macroF1={m['unit_type_macro_f1']:.3f} rest_acc={m['restoration_acc']:.3f} "
              f"rest_acc_known={m['restoration_acc_known_targets']:.3f}", flush=True)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "dev_metrics.json").write_text(json.dumps(history, indent=2))
    (out / "args.json").write_text(json.dumps(vars(a), indent=2))
    if a.save:
        torch.save(model.state_dict(), out / "model.pt")
        tok.save_pretrained(out / "tokenizer")
    print("final per-class (dev):")
    for t, v in history[-1]["per_class"].items():
        print(f"  {t:20s} P={v['p']:.3f} R={v['r']:.3f} F1={v['f1']:.3f} n={v['support']}")


if __name__ == "__main__":
    main()
