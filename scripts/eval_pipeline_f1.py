#!/usr/bin/env python
"""End-to-end disfluency F1 of pipeline outputs (``scripts/run_pipeline.py --out DIR``) against the verbatim gold.

Each DIR is scored twice: *Whisper alone* (restored tokens left out; Whisper's own fillers and immediate repeats
still count, via the same rules) and *restored* (everything the model inserted). See ``pipeline/score.py`` for the
definition. ``--from-frac 0.6`` scores only the last 40% of the lecture, the part ``configs/noise_profile.json`` was
not fitted on.

Run: .venv/bin/python -I scripts/eval_pipeline_f1.py --dir results/pipeline/english_001_cal_norm --from-frac 0.6
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from nptel.recognized import split_tokens  # noqa: E402
from nptel.rsml import aligned_to_side  # noqa: E402
from pipeline.asr import norm_word  # noqa: E402
from pipeline.score import GOLD_TYPES, disfluency_f1, restoration_f1, system_tokens  # noqa: E402


def gold_tokens(ref: Path, from_frac: float) -> tuple[list[tuple[str, str]], float]:
    """(word, type or "O") of the verbatim gold from the start of the scored part, and that part's start time."""
    toks = aligned_to_side(ref, sid="gold")["tokens"]
    cut = split_tokens(toks, from_frac) if from_frac > 0 else 0
    gold = [(norm_word(t["spoken"]), t["label"] if t["label"] in GOLD_TYPES else "O")
            for t in toks[cut:] if t["label"] != "NOISE"]
    return [(w, l) for w, l in gold if w], (toks[cut]["start"] if from_frac > 0 else 0.0)


def row(name: str, r: dict) -> str:
    u, t = r["untyped"], r["typed"]
    return (f"{name:46s} {u['system']:>7d} {u['tp']:>8d}   {u['p']:.3f}  {u['r']:.3f}  {u['f1']:.3f}   |  typed F1 {t['f1']:.3f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", nargs="+", required=True, help="pipeline output directories (with restored.json)")
    ap.add_argument("--ref", default=str(ROOT / "data/nptel/raw/english_001/english_001.aligned.rsml"))
    ap.add_argument("--from-frac", type=float, default=0.0, help="score only the last (1 - from_frac) of the lecture")
    ap.add_argument("--drop-types", default="", help="comma-separated restored types to leave out of every restored row")
    ap.add_argument("--per-type", action="store_true", help="print the per-type table of the restored rows")
    ap.add_argument("--out", default="", help="write all results as JSON")
    a = ap.parse_args()

    gold, t_from = gold_tokens(Path(a.ref), a.from_frac)
    n_gold = {t: sum(l == t for _, l in gold) for t in GOLD_TYPES}
    print(f"gold: {len(gold)} words from t={t_from:.1f}s, disfluent tokens {sum(n_gold.values())}: "
          f"{ {k: v for k, v in n_gold.items() if v} }\n")
    print(f"{'system':46s} {'sys D':>7s} {'correct':>8s}   {'P':5s}  {'R':5s}  {'F1':5s}")
    drop = frozenset(t for t in a.drop_types.split(",") if t)
    results = {}
    for d in a.dir:
        j = json.loads((Path(d) / "restored.json").read_text())
        everything = frozenset(t["label"] for t in j["tokens"] if t["source"] != "asr")
        name = Path(d).name
        for label, dropped in ((f"{name}: Whisper alone", everything), (f"{name}: restored", drop)):
            sysd = system_tokens(j["tokens"], dropped, t_from=t_from)
            r = disfluency_f1(gold, sysd)
            results[label] = r
            print(row(label, r))
            if label.endswith("restored"):
                rr = restoration_f1(gold, sysd)
                results[label]["restoration_only"] = rr
                u = rr["untyped"]
                print(f"{'   the model alone (restored vs missing)':46s} {u['system']:>7d} {u['tp']:>8d}   {u['p']:.3f}  {u['r']:.3f}  {u['f1']:.3f}"
                      f"   |  typed F1 {rr['typed']['f1']:.3f}   ({rr['gold_missing_from_whisper']} gold disfluent tokens missing from Whisper)")
                if a.per_type:
                    for t in r["per_type"]:
                        v, w = r["per_type"][t], rr["per_type"].get(t, {"gold": 0, "system": 0, "tp": 0, "p": 0, "r": 0, "f1": 0})
                        print(f"      {t:20s} end-to-end: gold {v['gold']:>3d} sys {v['system']:>3d} ok {v['tp']:>3d} F1 {v['f1']:.3f}"
                              f"  | model alone: missing {w['gold']:>3d} restored {w['system']:>3d} ok {w['tp']:>3d} P {w['p']:.2f} R {w['r']:.2f} F1 {w['f1']:.2f}")
                    print(f"      correct by source: {r['correct_by_source']}   gold present but not tagged: {r['gold_present_but_untagged']}")
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps({"from_frac": a.from_frac, "gold_counts": n_gold, "results": results}, indent=2))


if __name__ == "__main__":
    main()
