"""Tokenization + batching of gap / word-span examples for the BERT gap model."""
from __future__ import annotations

import gzip
import json
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset

from phones.lexicon import SWB_PHONES
from phones.noise import PhoneNoise

from .gaps import COPY_L, COPY_R, GAP_LABELS, SPAN_LABELS

PHONE_TOKENS = [f"<ph:{p}>" for p in SWB_PHONES]
SPECIAL_TOKENS = (
    ["[GAP]", "[/GAP]", "[WSPAN]", "[/WSPAN]"]
    + [f"[G{k}]" for k in range(8)]        # gap duration buckets
    + [f"[Z{k}]" for k in range(6)]        # word-span duration z-score buckets
    + PHONE_TOKENS
)
GAP_BOUNDS = (0.1, 0.2, 0.3, 0.45, 0.7, 1.0, 1.5)

TAG_TYPES = ["O"] + GAP_LABELS + ["NOISE"] + SPAN_LABELS
TAGS = ["O"] + [f"{p}-{t}" for t in TAG_TYPES[1:] for p in ("B", "I")]
TAG2ID = {t: i for i, t in enumerate(TAGS)}
IGNORE = -100


def gap_bucket(dur: float) -> int:
    return sum(dur >= b for b in GAP_BOUNDS)


def tag_for(label: str, first: bool) -> str:
    return "O" if label == "O" else f"{'B' if first else 'I'}-{label}"


def load_jsonl(path: str | Path, limit: int = 0) -> list[dict]:
    out = []
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for n, line in enumerate(fh):
            if limit and n >= limit:
                break
            out.append(json.loads(line))
    return out


class GapDataset(Dataset):
    def __init__(
        self, examples: list[dict], tokenizer, rest_vocab: list[str], max_len: int = 128,
        max_phones: int = 40, noise: PhoneNoise | None = None, use_phones: bool = True,
        seed: int = 0,
    ) -> None:
        self.ex, self.tok, self.max_len, self.max_phones = examples, tokenizer, max_len, max_phones
        self.rest2id = {t: i for i, t in enumerate(rest_vocab)}
        self.unk = self.rest2id["UNK"]
        self.noise, self.use_phones = noise, use_phones
        self.rng = random.Random(seed)
        self.sid = {t: tokenizer.convert_tokens_to_ids(t) for t in SPECIAL_TOKENS}
        self.words_cache: dict[str, list[int]] = {}

    def __len__(self) -> int:
        return len(self.ex)

    def _wp(self, words: list[str]) -> list[int]:
        ids: list[int] = []
        for w in words:
            c = self.words_cache.get(w)
            if c is None:
                c = self.tok.encode(w, add_special_tokens=False)
                self.words_cache[w] = c
            ids.extend(c)
        return ids

    def __getitem__(self, i: int) -> dict:
        ex = self.ex[i]
        phones = ex["phones"][: self.max_phones]
        units = ex["units"]
        if self.noise is not None and self.use_phones:
            # corrupt phones but keep unit spans valid by corrupting per unit
            new_ph, new_units = [], []
            for u in units:
                seg = self.noise.corrupt(ex["phones"][u["s"]:u["e"]])
                new_units.append({**u, "s": len(new_ph), "e": len(new_ph) + len(seg)})
                new_ph.extend(seg)
            phones, units = new_ph[: self.max_phones], new_units
        if not self.use_phones:
            phones = ["ax"] * min(len(phones), 1)       # keep a single uninformative slot
            units = [{**units[0], "s": 0, "e": 1}] if units else units

        S = self.sid
        room = self.max_len - 6 - len(phones) - 2
        n_side = max(2, room // 2)
        left = self._wp(ex["left"])[-n_side:]
        right = self._wp(ex["right"])[:n_side]
        ph_ids = [self.tok.convert_tokens_to_ids(f"<ph:{p}>") for p in phones]

        if ex["kind"] == "gap":
            head = [S["[GAP]"], S[f"[G{gap_bucket(ex['dur'])}]"]]
            tail = [S["[/GAP]"]]
        else:
            head = [S["[WSPAN]"]] + self._wp([ex["word"]]) + [S[f"[Z{ex.get('zb', 0)}]"]]
            tail = [S["[/WSPAN]"]]

        ids = [self.tok.cls_token_id] + left + head + ph_ids + tail + right + [self.tok.sep_token_id]
        ph_start = 1 + len(left) + len(head)
        types = [0] * len(ids)
        for k in range(ph_start - 1, ph_start + len(ph_ids)):
            types[k] = 1

        tags = [IGNORE] * len(ids)
        spans, targets, gold = [], [], []
        for u in units:
            s, e = u["s"], min(u["e"], len(phones))
            if s >= e:
                continue
            for k in range(s, e):
                tags[ph_start + k] = TAG2ID[tag_for(u["label"], k == s)]
            spans.append((ph_start + s, ph_start + e))
            targets.append(self.rest2id.get(u["target"], self.unk))
            gold.append((s, e, u["label"], u["target"]))
        return {
            "ids": ids, "types": types, "tags": tags, "spans": spans, "targets": targets,
            "ctx_pos": 1 + len(left),            # index of [GAP] / [WSPAN]
            "ph_start": ph_start, "ph_end": ph_start + len(ph_ids),
            "gold_units": gold,                  # phone-relative (s, e, label, target)
        }


def collate(batch: list[dict]) -> dict[str, torch.Tensor]:
    T = max(len(b["ids"]) for b in batch)
    B = len(batch)
    ids = torch.zeros(B, T, dtype=torch.long)
    types = torch.zeros(B, T, dtype=torch.long)
    att = torch.zeros(B, T, dtype=torch.long)
    tags = torch.full((B, T), IGNORE, dtype=torch.long)
    ub, us, ue, ut = [], [], [], []
    ctx = torch.zeros(B, dtype=torch.long)
    for k, b in enumerate(batch):
        n = len(b["ids"])
        ids[k, :n] = torch.tensor(b["ids"])
        types[k, :n] = torch.tensor(b["types"])
        att[k, :n] = 1
        tags[k, :n] = torch.tensor(b["tags"])
        ctx[k] = b["ctx_pos"]
        for (s, e), t in zip(b["spans"], b["targets"]):
            ub.append(k), us.append(s), ue.append(e), ut.append(t)
    return {
        "input_ids": ids, "token_type_ids": types, "attention_mask": att, "tags": tags,
        "unit_batch": torch.tensor(ub, dtype=torch.long), "unit_start": torch.tensor(us, dtype=torch.long),
        "unit_end": torch.tensor(ue, dtype=torch.long), "unit_target": torch.tensor(ut, dtype=torch.long),
        "ctx_pos": ctx,
    }
