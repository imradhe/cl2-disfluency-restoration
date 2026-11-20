#!/usr/bin/env python
"""Build the window datasets for the sequence-level restoration model (see src/model/seq.py).

Input: data/processed/swb_tokens.jsonl.gz (scripts/build_processed.py) + the sw-ms98 lexicon (G2P of the
transcript words). Output in --out: {train,dev,test}.jsonl.gz, unit_vocab.json, seq_stats.json.
Split by conversation id (Charniak & Johnson), as for the gap data. Train windows have random sizes and the
transcript keeps some disfluent tokens (--keep-p, as an ASR sometimes does); dev / test windows are fixed-size
and fully fluent, which is what the inference pipeline produces.

Run:  PYTHONPATH=src .venv/bin/python -I scripts/build_seq.py [--limit N]
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

from model.gaps import split_of  # noqa: E402
from model.seq import build_seq_side, count_units, make_unit_vocab  # noqa: E402
from phones.lexicon import SWB_PHONES, Lexicon  # noqa: E402
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
    ap.add_argument("--lex", default=str(ROOT / "data/raw/switchboard_ms98/sw-ms98-dict.text"))
    ap.add_argument("--out", default=str(ROOT / "data/seq"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--keep-p", type=float, default=0.3, help="train only: chance an ASR-keepable disfluent token stays in the transcript")
    ap.add_argument("--min-words", type=int, default=10)
    ap.add_argument("--max-words", type=int, default=24)
    ap.add_argument("--eval-words", type=int, default=20)
    ap.add_argument("--top-k-units", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=13)
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    lex = Lexicon.from_file(a.lex)
    rng, noise = random.Random(a.seed), PhoneNoise(seed=a.seed)
    files = {s: gzip.open(out / f"{s}.jsonl.gz", "wt", encoding="utf-8") for s in ("train", "dev", "test")}
    stats = {s: {"sides": 0, "windows": 0, "words": 0, "rec_phones": 0, "slots_with_units": 0,
                 "units_over_k": 0, "units": Counter()} for s in files}
    counts: Counter = Counter()
    bad_phone: Counter = Counter()
    for side in read_sides(Path(a.inp), a.limit):
        sp = split_of(side["conv"])
        train = sp == "train"
        sizes = (a.min_words, a.max_words) if train else (a.eval_words, a.eval_words)
        st = stats[sp]
        st["sides"] += 1
        for ex in build_seq_side(side, lex, rng, noise, keep_p=a.keep_p if train else 0.0, sizes=sizes):
            for p in ex["rec"] + [q for e in ex["exp"] for q in e]:
                if p not in SWB_PHONES:
                    bad_phone[p] += 1
            files[sp].write(json.dumps(ex, separators=(",", ":")) + "\n")
            st["windows"] += 1
            st["words"] += len(ex["words"])
            st["rec_phones"] += len(ex["rec"])
            for slot in ex["slots"]:
                if slot:
                    st["slots_with_units"] += 1
                    st["units_over_k"] += max(0, len(slot) - 4)
                    st["units"].update(u.split("|", 1)[0] for u in slot)
            if train:
                count_units(ex, counts)
    for f in files.values():
        f.close()

    vocab = make_unit_vocab(counts, a.top_k_units)
    (out / "unit_vocab.json").write_text(json.dumps(vocab))
    covered = sum(counts[u] for u in vocab if u in counts)
    summary = {s: {**{k: v for k, v in st.items() if k != "units"}, "units": dict(st["units"].most_common())}
               for s, st in stats.items()}
    summary["unit_classes"] = len(vocab)
    summary["train_unit_coverage"] = round(covered / max(1, sum(counts.values())), 4)
    summary["unknown_phones"] = dict(bad_phone)
    (out / "seq_stats.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
