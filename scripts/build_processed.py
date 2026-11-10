#!/usr/bin/env python
"""Phase 2: parse + label every Switchboard side.

Outputs (in --out):
  swb_tokens.jsonl.gz   one JSON object per speaker side, with labelled tokens + phones
  label_stats.json      counts per label / token kind, lexicon coverage
  spotcheck/<LABEL>.txt random context windows per label for manual precision checks

Run:  PYTHONPATH=src .venv/bin/python -I scripts/build_processed.py [--limit N]
"""
from __future__ import annotations

import argparse
import gzip
import json
import random
import sys
from collections import Counter
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from phones.lexicon import Lexicon  # noqa: E402
from swb.corpus import iter_sides  # noqa: E402
from swb.label import IN_WORD_LABELS, build_ltoks, collect_stats, label_side  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(ROOT / "data/raw/switchboard_ms98/extracted/swb_ms98_transcriptions"))
    ap.add_argument("--lex", default=None, help="lexicon path (default: <root>/sw-ms98-dict.text)")
    ap.add_argument("--cfg", default=str(ROOT / "configs/labels.yaml"))
    ap.add_argument("--out", default=str(ROOT / "data/processed"))
    ap.add_argument("--limit", type=int, default=0, help="only the first N sides (smoke test)")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--spot", type=int, default=40, help="spot-check samples per label")
    a = ap.parse_args()

    root = Path(a.root)
    cfg = yaml.safe_load(Path(a.cfg).read_text())
    lex = Lexicon.from_file(a.lex or root / "sw-ms98-dict.text")
    out = Path(a.out)
    (out / "spotcheck").mkdir(parents=True, exist_ok=True)
    rng = random.Random(a.seed)

    def sides():
        for n, s in enumerate(iter_sides(root)):
            if a.limit and n >= a.limit:
                return
            yield s

    print("pass 1: duration statistics ...", flush=True)
    stats = collect_stats(sides())
    print(f"  words with duration stats: {len(stats.mean)}", flush=True)

    label_counts: Counter = Counter()
    kind_counts: Counter = Counter()
    lex_hit: Counter = Counter()
    reservoir: dict[str, list[str]] = {}
    seen: Counter = Counter()
    n_sides = n_tok = 0

    print("pass 2: labelling ...", flush=True)
    with gzip.open(out / "swb_tokens.jsonl.gz", "wt", encoding="utf-8") as fh:
        for side in sides():
            toks = label_side(side, cfg, stats)
            recs = []
            for t in toks:
                tk = t.tok
                phones = lex.phones(tk) if tk.kind != "NOISE" else None
                label_counts[t.label] += 1
                kind_counts[tk.kind] += 1
                if tk.kind != "NOISE":
                    lex_hit[(tk.kind, phones is not None)] += 1
                recs.append({
                    "i": t.i, "raw": tk.raw, "kind": tk.kind, "spoken": tk.spoken,
                    "intended": tk.intended, "cut": tk.cut, "label": t.label, "role": t.role,
                    "prolonged": t.prolonged, "start": round(t.wt.start, 3), "end": round(t.wt.end, 3),
                    "pause_before": round(t.pause_before, 3), "utt": t.wt.utt_id,
                    "utt_first": t.utt_first, "utt_last": t.utt_last, "phones": phones,
                })
                # reservoir sampling of context windows for spot checks
                if t.label != "O":
                    seen[t.label] += 1
                    lo, hi = max(0, t.i - 6), min(len(toks), t.i + 7)
                    ctx = " ".join(
                        (f"<<{x.tok.raw}|{x.label}>>" if x.i == t.i else
                         (f"[{x.tok.raw}|{x.label}]" if x.label != "O" else x.tok.raw))
                        for x in toks[lo:hi]
                    )
                    line = f"{side.sid}\t{ctx}"
                    buf = reservoir.setdefault(t.label, [])
                    if len(buf) < a.spot:
                        buf.append(line)
                    else:
                        j = rng.randrange(seen[t.label])
                        if j < a.spot:
                            buf[j] = line
            fh.write(json.dumps({"sid": side.sid, "conv": side.conv, "side": side.side, "tokens": recs}) + "\n")
            n_sides += 1
            n_tok += len(recs)
            if n_sides % 500 == 0:
                print(f"  {n_sides} sides, {n_tok} tokens", flush=True)

    for label, lines in reservoir.items():
        (out / "spotcheck" / f"{label}.txt").write_text("\n".join(lines) + "\n")

    hit_rate = {}
    for kind in sorted({k for k, _ in lex_hit}):
        h, m = lex_hit[(kind, True)], lex_hit[(kind, False)]
        hit_rate[kind] = {"hit": h, "miss": m, "rate": round(h / max(1, h + m), 4)}
    summary = {
        "sides": n_sides, "tokens": n_tok, "labels": dict(label_counts.most_common()),
        "kinds": dict(kind_counts.most_common()), "lexicon_coverage": hit_rate,
    }
    (out / "label_stats.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
