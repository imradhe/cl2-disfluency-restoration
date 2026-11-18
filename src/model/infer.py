"""Inference for the BERT gap model: find units from phone tags, then restore them.

Pass 1 tags every phone (BIO + type); consecutive phones form predicted units.
Pass 2 re-runs the model with those predicted spans to classify each unit's restoration.
Unlike ``train.evaluate`` (which is handed the gold unit spans), this needs no oracle.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import torch

from .bert_gap import BertGap
from .dataset import TAGS, collate


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")


def load_gap_model(ckpt: str | Path, rest_vocab_path: str | Path | None = None, bert: str = "bert-base-uncased",
                   dev: torch.device | None = None):
    """Load a trained checkpoint directory -> ``(model, tokenizer, rest_vocab, device)``.

    ``rest_vocab_path`` defaults to ``<ckpt>/restoration_vocab.json`` (saved by the Colab runs).
    """
    from transformers import AutoTokenizer

    ckpt, dev = Path(ckpt), dev or pick_device()
    vocab_path = Path(rest_vocab_path) if rest_vocab_path else ckpt / "restoration_vocab.json"
    tok = AutoTokenizer.from_pretrained(ckpt / "tokenizer")
    rest_vocab = json.loads(vocab_path.read_text())
    model = BertGap(bert, len(tok), len(TAGS), len(rest_vocab)).to(dev)
    model.load_state_dict(torch.load(ckpt / "model.pt", map_location=dev))
    model.eval()
    return model, tok, rest_vocab, dev


def decode_units(tags: list[str]) -> list[tuple[int, int, str]]:
    """Group per-phone BIO tags into (start, end, type) units; 'O' phones form no unit."""
    units: list[tuple[int, int, str]] = []
    cur: list | None = None
    for i, t in enumerate(tags):
        if t == "O":
            cur = None
            continue
        pref, typ = t.split("-", 1)
        if cur is None or pref == "B" or cur[2] != typ:
            cur = [i, i + 1, typ]
            units.append(cur)          # type: ignore[arg-type]
        else:
            cur[1] = i + 1
    return [(s, e, t) for s, e, t in units]


def _to_device(batch: dict, dev: torch.device) -> dict:
    return {k: v.to(dev) for k, v in batch.items()}


@torch.no_grad()
def predict(model, items: list[dict], dev: torch.device, rest_vocab: list[str], bs: int = 64) -> list[list[dict]]:
    """For each dataset item return its predicted units: [{s, e, type, restoration}] (phone-relative)."""
    model.eval()
    results: list[list[dict]] = []
    for lo in range(0, len(items), bs):
        chunk = items[lo:lo + bs]
        b1 = collate([{**it, "spans": [], "targets": []} for it in chunk])
        out1 = model(_to_device(b1, dev), compute_loss=False)
        pred = out1["tag_logits"].argmax(-1).cpu()
        per_item: list[list[tuple[int, int, str]]] = []
        spans: list[list[tuple[int, int]]] = []
        for k, it in enumerate(chunk):
            tags = [TAGS[int(pred[k, p])] for p in range(it["ph_start"], it["ph_end"])]
            units = decode_units(tags)
            per_item.append(units)
            spans.append([(it["ph_start"] + s, it["ph_start"] + e) for s, e, _ in units])
        b2 = collate([{**it, "spans": sp, "targets": [0] * len(sp)} for it, sp in zip(chunk, spans)])
        rest = []
        if b2["unit_batch"].numel():
            rest = model(_to_device(b2, dev), compute_loss=False)["rest_logits"].argmax(-1).cpu().tolist()
        j = 0
        for it, units in zip(chunk, per_item):
            res = []
            for s, e, typ in units:
                res.append({"s": s, "e": e, "type": typ, "restoration": rest_vocab[rest[j]]})
                j += 1
            results.append(res)
    return results


def unit_prf(gold: list[list[tuple]], pred: list[list[dict]], overlap: float = 0.5) -> dict:
    """Per-type detection P/R/F1, strict (same span + type) and lenient (>=overlap IoU + type).

    ``gold``: per item a list of (s, e, label, target) in phone-relative coordinates.
    Also returns restoration accuracy over strictly matched units.
    """
    tp_s: dict = defaultdict(int); tp_l: dict = defaultdict(int)
    n_gold: dict = defaultdict(int); n_pred: dict = defaultdict(int)
    rest_ok = rest_n = 0
    for g_items, p_items in zip(gold, pred):
        g_items = [g for g in g_items if g[2] != "O"]
        for g in g_items:
            n_gold[g[2]] += 1
        for p in p_items:
            n_pred[p["type"]] += 1
        used_s: set = set(); used_l: set = set()
        for pi, p in enumerate(p_items):
            for gi, g in enumerate(g_items):
                if g[2] != p["type"]:
                    continue
                inter = max(0, min(g[1], p["e"]) - max(g[0], p["s"]))
                union = max(g[1], p["e"]) - min(g[0], p["s"])
                if gi not in used_s and (g[0], g[1]) == (p["s"], p["e"]):
                    used_s.add(gi); tp_s[g[2]] += 1
                    rest_n += 1; rest_ok += int(g[3] == p["restoration"])
                if gi not in used_l and union and inter / union >= overlap:
                    used_l.add(gi); tp_l[g[2]] += 1
                    break
    types = sorted(set(n_gold) | set(n_pred))
    def prf(tp, t):
        p = tp[t] / n_pred[t] if n_pred[t] else 0.0
        r = tp[t] / n_gold[t] if n_gold[t] else 0.0
        return round(p, 4), round(r, 4), round(2 * p * r / (p + r), 4) if p + r else 0.0
    table = {}
    for t in types:
        ps, rs, fs = prf(tp_s, t); pl, rl, fl = prf(tp_l, t)
        table[t] = {"support": n_gold[t], "predicted": n_pred[t],
                    "strict": {"p": ps, "r": rs, "f1": fs}, "lenient": {"p": pl, "r": rl, "f1": fl}}
    present = [t for t in types if n_gold[t]]
    return {
        "per_type": table,
        "macro_f1_strict": round(sum(table[t]["strict"]["f1"] for t in present) / max(1, len(present)), 4),
        "macro_f1_lenient": round(sum(table[t]["lenient"]["f1"] for t in present) / max(1, len(present)), 4),
        "restoration_acc_on_strict_matches": round(rest_ok / max(1, rest_n), 4),
        "strict_matches": rest_n,
    }
