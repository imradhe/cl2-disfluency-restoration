"""Build gap / word-span training examples from labelled Switchboard sides.

A *skeleton* is the side's fluent word sequence (tokens labelled O, PROLONGATION or
MISPRONUNCIATION; mispronounced words are replaced by the intended word). Every
maximal run of disfluent tokens between two skeleton words is an **insertion gap**
whose phones are the concatenated lexicon phones of its tokens. Words that are
prolonged / mispronounced stay in the skeleton and become **word-span** examples.

Example schema (one JSON object per example)::

    {"id", "sid", "kind": "gap" | "wspan",
     "left": [...words], "right": [...words],
     "word": str            # wspan only: the (intended) word in the skeleton
     "phones": [...],
     "units": [{"s": int, "e": int, "label": str, "target": str}, ...],
     "dur": float,          # gap: summed token durations; wspan: word duration
     "zb": int}             # wspan: duration z-score bucket

Restoration targets (strings):  ``FIL:<w>`` closed filler/DM words, ``COPY_R<k>`` /
``COPY_L<k>`` the unit repeats / completes the k-th word right / left of the gap,
``W:<w>`` open-vocabulary word, ``SELF`` (word-span: the skeleton word itself),
``DROP`` (noise), ``UNK``.
"""
from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Iterable, Iterator

from phones.noise import PhoneNoise

GAP_LABELS = [
    "FILLER_HESITATION", "FILLER_BACKCHANNEL", "FILLER_DM",
    "REPETITION", "REPAIR", "FALSE_START", "PARTIAL_WORD",
]
SPAN_LABELS = ["PROLONGATION", "MISPRONUNCIATION"]
SKELETON_LABELS = {"O", *SPAN_LABELS}
FILLER_LABELS = {"FILLER_HESITATION", "FILLER_BACKCHANNEL", "FILLER_DM"}
COPY_R, COPY_L = 4, 8        # pointer range into right / left context


def split_of(conv: str) -> str:
    """Conventional Switchboard split by conversation id (Charniak & Johnson)."""
    n = int(conv)
    if 4010 <= n <= 4153:
        return "test"
    if 4519 <= n <= 4936:
        return "dev"
    return "train"


def z_bucket(z: float | None) -> int:
    if z is None:
        return 0
    return min(5, max(0, int(z) + 1)) if z >= 0 else 0


class WordDur:
    """Per-word mean/std log duration, for z-score features."""

    def __init__(self) -> None:
        self._xs: dict[str, list[float]] = defaultdict(list)
        self.mean: dict[str, float] = {}
        self.std: dict[str, float] = {}

    def add_side(self, toks: list[dict]) -> None:
        for t in toks:
            if t["kind"] == "WORD" and t["end"] - t["start"] > 0.02:
                self._xs[t["spoken"]].append(math.log(t["end"] - t["start"]))

    def finalize(self, min_n: int = 30) -> None:
        for w, xs in self._xs.items():
            if len(xs) >= min_n:
                m = sum(xs) / len(xs)
                v = sum((x - m) ** 2 for x in xs) / max(1, len(xs) - 1)
                self.mean[w], self.std[w] = m, max(math.sqrt(v), 0.15)
        self._xs = defaultdict(list)

    def z(self, word: str, dur: float) -> float | None:
        if word not in self.mean or dur <= 0.02:
            return None
        return (math.log(dur) - self.mean[word]) / self.std[word]


def _skeleton_word(t: dict) -> str:
    return t["intended"] if t["label"] == "MISPRONUNCIATION" and t["intended"] else t["spoken"]


def restore_target(label: str, text: str, intended: str, left: list[str], right: list[str]) -> str:
    """Choose the restoration class for one unit given its fluent context."""
    if label in FILLER_LABELS:
        return f"FIL:{text}"
    word = intended if (label == "PARTIAL_WORD" and intended) else text
    if label == "PARTIAL_WORD" and not intended:
        return "UNK"
    for k, w in enumerate(right[:COPY_R], 1):
        if w == word:
            return f"COPY_R{k}"
    for k, w in enumerate(reversed(left[-COPY_L:]), 1):
        if w == word:
            return f"COPY_L{k}"
    return f"W:{word}"


def build_side(
    side: dict, wd: WordDur, rng: random.Random, noise: PhoneNoise, ctx: int = 16,
    neg_span_ratio: float = 1.0, noise_gaps: bool = True,
) -> Iterator[dict]:
    toks = side["tokens"]
    sid = side["sid"]

    # --- pass 1: skeleton words + pending disfluent runs --------------------------------
    skel: list[str] = []
    skel_tok_idx: list[int] = []                 # index into toks for each skeleton word
    gaps: list[tuple[int, list[dict]]] = []      # (position in skeleton, unit tokens)
    noise_pos: list[int] = []                    # skeleton positions of NOISE tokens
    pending: list[dict] = []
    for t in toks:
        lab = t["label"]
        if lab == "NOISE":
            noise_pos.append(len(skel))
            continue
        if lab in SKELETON_LABELS:
            if pending:
                gaps.append((len(skel), pending))
                pending = []
            skel.append(_skeleton_word(t))
            skel_tok_idx.append(t["i"])
        else:
            pending.append(t)
    if pending:
        gaps.append((len(skel), pending))

    def window(pos: int) -> tuple[list[str], list[str]]:
        return skel[max(0, pos - ctx):pos], skel[pos:pos + ctx]

    # --- insertion gaps ---------------------------------------------------------------
    for g, (pos, units) in enumerate(gaps):
        left, right = window(pos)
        phones: list[str] = []
        spans: list[dict] = []
        for u in units:
            ph = u["phones"]
            if not ph:
                continue
            s = len(phones)
            phones.extend(ph)
            spans.append({
                "s": s, "e": len(phones), "label": u["label"],
                "target": restore_target(u["label"], u["spoken"], u["intended"], left, right),
            })
        if not spans:
            continue
        yield {
            "id": f"{sid}:g{g}", "sid": sid, "kind": "gap", "left": left, "right": right,
            "phones": phones, "units": spans,
            "dur": round(sum(u["end"] - u["start"] for u in units), 3),
        }

    # --- NOISE tokens heard as junk phones --------------------------------------------
    if noise_gaps:
        for n, pos in enumerate(noise_pos):
            if rng.random() > 0.5:
                continue
            left, right = window(pos)
            ph = noise.spurious()
            yield {
                "id": f"{sid}:n{n}", "sid": sid, "kind": "gap", "left": left, "right": right,
                "phones": ph, "units": [{"s": 0, "e": len(ph), "label": "NOISE", "target": "DROP"}],
                "dur": round(rng.uniform(0.1, 0.8), 3),
            }

    # --- word spans: positives (prolonged / mispronounced) + sampled fluent negatives ----
    positives, negatives = [], []
    for pos, ti in enumerate(skel_tok_idx):
        t = toks[ti]
        if not t["phones"]:
            continue
        (positives if t["label"] in SPAN_LABELS else negatives).append((pos, t))
    n_neg = int(round(len(positives) * neg_span_ratio))
    rng.shuffle(negatives)
    for k, (pos, t) in enumerate(positives + negatives[:n_neg]):
        left, right = skel[max(0, pos - ctx):pos], skel[pos + 1:pos + 1 + ctx]
        dur = t["end"] - t["start"]
        word = skel[pos]
        label = t["label"]
        yield {
            "id": f"{sid}:w{k}", "sid": sid, "kind": "wspan", "left": left, "right": right,
            "word": word, "phones": t["phones"],
            "units": [{"s": 0, "e": len(t["phones"]), "label": label, "target": "SELF"}],
            "dur": round(dur, 3), "zb": z_bucket(wd.z(word, dur)),
        }


def count_targets(ex: dict, fil: dict, words: dict) -> None:
    """Accumulate restoration-target frequencies of one example (call on train only)."""
    for u in ex["units"]:
        tg = u["target"]
        if tg.startswith("FIL:"):
            fil[tg] = fil.get(tg, 0) + 1
        elif tg.startswith("W:"):
            words[tg] = words.get(tg, 0) + 1


def make_restoration_vocab(fil: dict, words: dict, top_k_words: int = 4000) -> list[str]:
    """Class list for the restoration head, built from *training* examples only."""
    fixed = ["<pad>", "SELF", "UNK", "DROP"]
    fixed += [f"COPY_R{k}" for k in range(1, COPY_R + 1)]
    fixed += [f"COPY_L{k}" for k in range(1, COPY_L + 1)]
    fil_list = sorted(fil, key=lambda k: -fil[k])
    word_list = sorted(words, key=lambda k: -words[k])[:top_k_words]
    return fixed + fil_list + word_list
