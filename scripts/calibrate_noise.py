#!/usr/bin/env python
"""Fit the recognizer-error profile (src/phones/channel.py) on the first part of a verified NPTEL lecture.

For every utterance of the forced-aligned lecture the G2P phones of its gold tokens are aligned with the wav2vec2
phones recognized in the utterance's time span (scripts/nptel_phones.py output); the profile is fitted on the first
``--fit-frac`` of the lecture and written to configs/noise_profile.json. The remainder is never used here and is
what scripts/eval_nptel_seq.py --from-frac scores. As a check, the fitted channel is replayed on the held-out
utterances and its statistics are compared with the real ones.

Run: HF_HOME=.cache/hf HF_HUB_OFFLINE=1 .venv/bin/python -I scripts/calibrate_noise.py
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from nptel.recognized import load_recognized, recognized_phones, split_tokens  # noqa: E402
from nptel.rsml import aligned_to_side  # noqa: E402
from phones.channel import (  # noqa: E402
    HES, HES_REF, VOWELS, ProfileNoise, bucket, fit_profile, save_profile, stretch_outcomes,
)
from phones.g2p import G2P  # noqa: E402


def stretches(toks: list[dict], g2p: G2P, centres, rtoks) -> list[tuple[list[tuple[str, list[str]]], list[str]]]:
    """One (gold tokens [(label, G2P phones)], recognized phones of the utterance's time span) per utterance."""
    out = []
    for _, grp in itertools.groupby(toks, key=lambda t: t["utt"]):
        grp = list(grp)
        items = [(t["label"], list(g2p.phones(t["spoken"]))) for t in grp if t["label"] != "NOISE"]
        items = [(lab, ref) for lab, ref in items if ref]
        if items:
            out.append((items, recognized_phones(centres, rtoks, g2p.ipa, grp[0]["start"], grp[-1]["end"], tol=0.0)))
    return out


def summarize(rows: list[dict]) -> dict:
    tok_n, tok_empty, strays = Counter(), Counter(), 0
    hes_n = hes_empty = ref_n = n_tok = n_ins_all = 0
    err: Counter = Counter()
    hes_str: Counter = Counter()
    for row in rows:
        ref, out, ins = row["ref"], row["out"], row["ins"]
        if not ref:
            continue
        n_tok += 1
        strays += bool(ins[-1])
        n_ins_all += sum(map(len, ins))
        dropped = all(o is None for o in out)
        if row["label"] == HES and tuple(ref) == HES_REF:
            hes_n += 1
            if out[0] in VOWELS:
                hes_str[out[0]] += 1
            else:
                hes_empty += 1                      # nothing, or only a consonant (a neighbour's phone)
            continue
        b = bucket(len(ref))
        tok_n[b] += 1
        if dropped:
            tok_empty[b] += 1
            continue
        for k, r in enumerate(ref):
            ref_n += 1
            err["del"] += out[k] is None
            err["sub"] += out[k] is not None and out[k] != r
    res = {"tokens": n_tok,
           "token_dropped": {b: f"{tok_empty[b]}/{tok_n[b]} = {tok_empty[b] / max(1, tok_n[b]):.1%}" for b in ("1", "2", "3+")},
           "on_surviving_tokens": {k: round(v / max(1, ref_n), 4) for k, v in err.items()},
           "inserted_phones_per_100_tokens": round(100 * n_ins_all / max(1, n_tok), 1),
           "tokens_followed_by_stray_phones": round(strays / max(1, n_tok), 3),
           "hesitations": {"n": hes_n, "not_written_as_a_vowel": round(hes_empty / max(1, hes_n), 3),
                           "written_as": dict(hes_str.most_common(6))}}
    res["on_surviving_tokens"]["del+sub"] = round(sum(err.values()) / max(1, ref_n), 4)
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stem", default="english_001")
    ap.add_argument("--rsml-dir", default=str(ROOT / "data/nptel/raw"))
    ap.add_argument("--phones-dir", default=str(ROOT / "data/nptel/phones"))
    ap.add_argument("--lex", default=str(ROOT / "data/raw/switchboard_ms98/sw-ms98-dict.text"))
    ap.add_argument("--fit-frac", type=float, default=0.6, help="fit on the first part of the lecture, hold out the rest")
    ap.add_argument("--replays", type=int, default=10, help="times the fitted channel is replayed on the held-out part")
    ap.add_argument("--out", default=str(ROOT / "configs/noise_profile.json"))
    ap.add_argument("--report", default=str(ROOT / "results/noise_calibration.json"))
    a = ap.parse_args()

    g2p = G2P.from_files(a.lex, ROOT / "configs/phonemap.yaml")
    side = aligned_to_side(Path(a.rsml_dir) / a.stem / f"{a.stem}.aligned.rsml", sid=a.stem)
    toks = side["tokens"]
    cut = split_tokens(toks, a.fit_frac)
    fit, held = toks[:cut], toks[cut:]
    centres, rtoks = load_recognized(Path(a.phones_dir) / f"{a.stem}.jsonl")
    st_fit, st_held = stretches(fit, g2p, centres, rtoks), stretches(held, g2p, centres, rtoks)
    rows_fit = [r for items, hyp in st_fit for r in stretch_outcomes(items, hyp)]
    rows_held = [r for items, hyp in st_held for r in stretch_outcomes(items, hyp)]

    meta = {"lecture": a.stem, "fit_tokens": f"0..{cut} of {len(toks)}", "fit_seconds": [fit[0]["start"], fit[-1]["end"]],
            "recognizer": "facebook/wav2vec2-xlsr-53-espeak-cv-ft (greedy CTC), tol=0", "alignment": "per utterance"}
    prof = fit_profile(rows_fit, meta=meta)
    save_profile(prof, a.out)

    # replay check: the fitted channel applied to the held-out reference phones, aligned the same way as the real ones
    replay_rows = []
    for seed in range(a.replays):
        noise = ProfileNoise(prof, p_profile=1.0, seed=seed)
        for items, _ in st_held:
            flat = [p for _, ref in items for p in ref]
            groups = [t for t, (_, ref) in enumerate(items) for _ in ref]
            noisy, _src = noise.corrupt_idx(flat, groups, [lab for lab, _ in items])
            replay_rows += stretch_outcomes(items, noisy)

    report = {"meta": meta, "fit_real": summarize(rows_fit), "held_out_real": summarize(rows_held),
              "held_out_replayed_by_profile": summarize(replay_rows),
              "profile": {k: prof[k] for k in ("tok_del", "global", "orphan") if k in prof}
              | ({"hes": {k: v for k, v in prof["hes"].items()}} if "hes" in prof else {})}
    Path(a.report).parent.mkdir(parents=True, exist_ok=True)
    Path(a.report).write_text(json.dumps(report, indent=2))
    print(json.dumps({k: report[k] for k in ("fit_real", "held_out_real", "held_out_replayed_by_profile")}, indent=2))
    print("orphan:", prof.get("orphan"), "\ntok_del:", prof["tok_del"], "global:", prof["global"])
    print(f"\nwrote {a.out} and {a.report}")


if __name__ == "__main__":
    main()
