#!/usr/bin/env python
"""Phase 4: turn labelled sides into gap / word-span examples, split by conversation.

Outputs in --out:  {train,dev,test}.jsonl.gz, restoration_vocab.json, gap_stats.json
Run:  PYTHONPATH=src .venv/bin/python -I scripts/build_gaps.py [--limit N]
"""
from __future__ import annotations

import argparse
import gzip
import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from model.gaps import (  # noqa: E402
    WordDur, build_side, count_targets, make_restoration_vocab, split_of,
)
from phones.lexicon import SWB_PHONES  # noqa: E402
from phones.noise import PhoneNoise  # noqa: E402


def read_sides(path: Path, limit: int = 0):
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for n, line in enumerate(fh):
            if limit and n >= limit:
                return
            yield json.loads(line)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", default=str(ROOT / "data/processed/swb_tokens.jsonl.gz"))
    ap.add_argument("--out", default=str(ROOT / "data/gaps"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--ctx", type=int, default=16)
    ap.add_argument("--neg-span-ratio", type=float, default=1.0)
    ap.add_argument("--top-k-words", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=13)
    a = ap.parse_args()

    inp, out = Path(a.inp), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(a.seed)
    noise = PhoneNoise(seed=a.seed)

    print("pass 1: word duration statistics (train split) ...", flush=True)
    wd = WordDur()
    for side in read_sides(inp, a.limit):
        if split_of(side["conv"]) == "train":
            wd.add_side(side["tokens"])
    wd.finalize()

    print("pass 2: building examples ...", flush=True)
    files = {s: gzip.open(out / f"{s}.jsonl.gz", "wt", encoding="utf-8") for s in ("train", "dev", "test")}
    fil: dict[str, int] = {}
    words: dict[str, int] = {}
    stats = {s: {"examples": 0, "sides": 0, "kinds": Counter(), "labels": Counter(), "phones": 0} for s in files}
    bad_phone: Counter = Counter()
    for side in read_sides(inp, a.limit):
        sp = split_of(side["conv"])
        stats[sp]["sides"] += 1
        for ex in build_side(side, wd, rng, noise, ctx=a.ctx, neg_span_ratio=a.neg_span_ratio):
            for p in ex["phones"]:
                if p not in SWB_PHONES:
                    bad_phone[p] += 1
            files[sp].write(json.dumps(ex, separators=(",", ":")) + "\n")
            st = stats[sp]
            st["examples"] += 1
            st["kinds"][ex["kind"]] += 1
            st["phones"] += len(ex["phones"])
            for u in ex["units"]:
                st["labels"][u["label"]] += 1
            if sp == "train":
                count_targets(ex, fil, words)
    for f in files.values():
        f.close()

    vocab = make_restoration_vocab(fil, words, a.top_k_words)
    (out / "restoration_vocab.json").write_text(json.dumps(vocab))
    summary = {s: {**{k: v for k, v in st.items() if k not in ("kinds", "labels")},
                   "kinds": dict(st["kinds"]), "labels": dict(st["labels"].most_common())}
               for s, st in stats.items()}
    summary["restoration_classes"] = len(vocab)
    summary["unknown_phones"] = dict(bad_phone)
    (out / "gap_stats.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
