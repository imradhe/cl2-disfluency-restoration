#!/usr/bin/env python
"""Evaluate a sequence-model checkpoint on the verified NPTEL lectures, with gold segmentation.

Gold tags give, per window of 20 words, the disfluency-removed transcript (what an ideal ASR would write), its
G2P phones, and the units the model should restore. The recognized phone stream is either
  g2p        clean lexicon/espeak phones of every spoken token (+ the training-time phone noise, "g2p_noisy"), or
  recognized the real wav2vec2 output (data/nptel/phones) of the window's time span.
Also scores a Switchboard dev sample so the domain gap is visible. Reports slot-level metrics for several
``none_bias`` values (a threshold on "nothing to restore").

Run: HF_HOME=.cache/hf HF_HUB_OFFLINE=1 .venv/bin/python -I scripts/eval_nptel_seq.py --ckpt ckpt/seq_local10k
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from model.bert_seq import load_seq_model  # noqa: E402
from model.dataset import load_jsonl  # noqa: E402
from model.seq import build_seq_side, predict_seq, slot_prf  # noqa: E402
from model.train_seq import word_tag_prf  # noqa: E402
from nptel.recognized import load_recognized, recognized_phones, split_tokens  # noqa: E402
from nptel.rsml import aligned_to_side, parse_rsml, to_side  # noqa: E402
from phones.channel import ProfileNoise, load_profile  # noqa: E402
from phones.g2p import G2P  # noqa: E402
from phones.lexicon import Lexicon  # noqa: E402
from phones.noise import PhoneNoise  # noqa: E402


def nptel_windows(rsml_dir: Path, g2p: G2P, lex: Lexicon, phones_dir: Path | None, words: int,
                  from_frac: float = 0.0) -> tuple[list[dict], dict]:
    """Windows of every English, fully verified lecture. ``phones_dir`` None -> clean G2P phones of every token.
    ``from_frac`` > 0 keeps only the windows of the last part of each lecture (the part a noise profile was not
    fitted on, see scripts/calibrate_noise.py)."""
    out: list[dict] = []
    info = {"files": [], "gold_units": Counter(), "dropped_tags": Counter()}
    for p in sorted(rsml_dir.rglob("*.rsml")):
        if p.name.endswith(".aligned.rsml"):
            continue
        r = parse_rsml(p)
        if not (r.is_english and r.fully_verified):
            continue
        aligned = p.with_name(p.stem + ".aligned.rsml")
        side = aligned_to_side(aligned, sid=p.stem) if aligned.exists() else to_side(r)
        toks = side["tokens"]
        for t in toks:
            t["phones"] = g2p.phones(t["spoken"]) or None
        cut = split_tokens(toks, from_frac) if from_frac > 0 else 0
        part = dict(side, tokens=toks[cut:])             # token ``i`` / window ``span`` keep indexing the full ``toks``
        wins = list(build_seq_side(part, lex, random.Random(0), PhoneNoise(seed=0), keep_p=0.0, sizes=(words, words), noise_p=0.0))
        if phones_dir is not None:
            pf = phones_dir / f"{p.stem}.jsonl"
            if not pf.exists():
                raise SystemExit(f"no recognized phones for {p.stem}: run scripts/nptel_phones.py first")
            centres, rtoks = load_recognized(pf)
            for w in wins:
                lo, hi = w["span"]
                t0 = toks[lo]["start"]
                t1 = toks[hi]["start"] if hi < len(toks) else toks[hi - 1]["end"] + 1.0
                w["rec"] = recognized_phones(centres, rtoks, g2p.ipa, t0, t1, tol=0.0)
                for k in ("rec_role", "rec_ptr", "rec_tok", "tok_lab"):
                    w.pop(k, None)                # these described the G2P stream, not this one
        for w in wins:
            for slot in w["slots"]:
                info["gold_units"].update(u.split("|", 1)[0] for u in slot or [])
        info["dropped_tags"].update(side["dropped_tags"])
        info["files"].append({"file": p.name, "segments": len(r.segments), "minutes": round(r.duration / 60, 1), "windows": len(wins),
                              "from_token": cut})
        out += wins
    return out, info


def score(model, tok, vocab, dev, windows: list[dict], noise: PhoneNoise | None, biases: tuple[float, ...]) -> dict:
    preds = predict_seq(model, windows, tok, vocab, dev, none_biases=biases, noise=noise)
    res = {}
    for b, pr in preds.items():
        m = slot_prf([w["slots"] for w in windows], [p["slots"] for p in pr])
        m["word_tags"] = word_tag_prf([w["wtags"] for w in windows], [p["wtags"] for p in pr])
        m["n_windows"] = len(windows)
        res[str(b)] = m
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--bert", default="bert-base-uncased")
    ap.add_argument("--rsml-dir", default=str(ROOT / "data/nptel/raw"))
    ap.add_argument("--phones-dir", default=str(ROOT / "data/nptel/phones"))
    ap.add_argument("--lex", default=str(ROOT / "data/raw/switchboard_ms98/sw-ms98-dict.text"))
    ap.add_argument("--seq", default=str(ROOT / "data/seq"))
    ap.add_argument("--words", type=int, default=20, help="words per window")
    ap.add_argument("--dev-n", type=int, default=1500)
    ap.add_argument("--biases", default="0,1,2,3")
    ap.add_argument("--out", default=str(ROOT / "results"))
    ap.add_argument("--from-frac", type=float, default=0.0,
                    help="score only the last (1 - from_frac) of each lecture; 0.6 = the part configs/noise_profile.json "
                         "was NOT fitted on")
    ap.add_argument("--profile", default=str(ROOT / "configs/noise_profile.json"),
                    help="fitted recognizer-error profile; adds a 'g2p_profile' set (G2P phones through that channel)")
    ap.add_argument("--tag", default="", help="suffix of the output file name")
    a = ap.parse_args()
    biases = tuple(float(b) for b in a.biases.split(","))

    ckpt = Path(a.ckpt)
    model, tok, vocab, dev = load_seq_model(ckpt, bert=a.bert)
    lex = Lexicon.from_file(a.lex)
    g2p = G2P.from_files(a.lex, ROOT / "configs/phonemap.yaml")

    clean, info = nptel_windows(Path(a.rsml_dir), g2p, lex, None, a.words, a.from_frac)
    real, _ = nptel_windows(Path(a.rsml_dir), g2p, lex, Path(a.phones_dir), a.words, a.from_frac)
    print(f"NPTEL (from_frac={a.from_frac}): {info['files']}  gold units: {dict(info['gold_units'])}")
    sets = {
        "nptel_g2p_clean": (clean, None),
        "nptel_g2p_noisy": (clean, PhoneNoise(seed=5)),
        "nptel_recognized": (real, None),
        "switchboard_dev_noisy": (load_jsonl(Path(a.seq) / "dev.jsonl.gz", a.dev_n), PhoneNoise(seed=1)),
    }
    if Path(a.profile).exists():
        sets["nptel_g2p_profile"] = (clean, ProfileNoise(load_profile(a.profile), p_profile=1.0, seed=5))
    results = {}
    for name, (wins, noise) in sets.items():
        res = score(model, tok, vocab, dev, wins, noise, biases)
        results[name] = res
        print(f"\n=== {name}  ({len(wins)} windows)")
        for b, m in res.items():
            print(f"  none_bias={b:>4s}  slot P/R/F1={m['slot']['p']:.2f}/{m['slot']['r']:.2f}/{m['slot']['f1']:.2f}  "
                  f"exact-unit F1={m['exact']['f1']:.2f}  type macroF1={m['macro_f1']:.2f}  "
                  f"restoration acc on type matches={m['restoration_acc_on_type_matches']:.2f}")
        m = res[str(biases[0])]
        print(f"  {'type':20s}{'support':>8s}{'pred':>7s}   P/R/F1 (none_bias={biases[0]})")
        for t, v in m["per_type"].items():
            print(f"  {t:20s}{v['support']:8d}{v['predicted']:7d}   {v['p']:.2f}/{v['r']:.2f}/{v['f1']:.2f}")

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    dest = out / f"nptel_seq_{ckpt.name}{a.tag}.json"
    dest.write_text(json.dumps({"ckpt": str(ckpt), "nptel_files": info["files"], "gold_units": dict(info["gold_units"]),
                                "dropped_tags": dict(info["dropped_tags"]), "words_per_window": a.words,
                                "from_frac": a.from_frac,
                                "results": results}, indent=2))
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
