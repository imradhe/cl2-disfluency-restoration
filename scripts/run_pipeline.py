#!/usr/bin/env python
"""Phase 7: audio -> VAD -> Whisper + phone recognizer -> alignment -> gaps -> BERT -> restored transcript.

Run (offline once the models are cached):
  HF_HOME=.cache/hf .venv/bin/python -I scripts/run_pipeline.py \\
      --audio data/nptel/wav16k/english_001.wav --ckpt ckpt/colab_main --out results/pipeline/english_001

``--reference data/nptel/raw/english_001/english_001.aligned.rsml`` additionally prints a sanity check:
word error rate of the ASR transcript and of the restored transcript against the verbatim gold
transcript (disfluencies included) and against the fluent gold. It is not the Phase 8 evaluation.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("HF_HOME", str(ROOT / ".cache/hf"))

from nptel.rsml import aligned_to_side, parse_rsml, to_side  # noqa: E402
from pipeline.asr import DEFAULT_MODEL  # noqa: E402
from pipeline.extract import GapParams  # noqa: E402
from pipeline.run import Config, run  # noqa: E402


def gold_words(path: Path, limit_s: float) -> tuple[list[str], list[str]]:
    """(verbatim, fluent) gold word lists from an (aligned) rsml, optionally cut at ``limit_s`` seconds."""
    side = aligned_to_side(path) if path.name.endswith(".aligned.rsml") else to_side(parse_rsml(path))
    toks = [t for t in side["tokens"] if not limit_s or t["start"] < limit_s]
    verbatim = [t["spoken"] for t in toks]
    fluent = [t["spoken"] for t in toks if t["label"] in ("O", "PROLONGATION")]
    return verbatim, fluent


def wer(ref: list[str], hyp: str) -> float:
    import jiwer
    return jiwer.wer(" ".join(ref), hyp)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--audio", required=True, help="any ffmpeg-readable audio file (16 kHz mono wav is used as is)")
    ap.add_argument("--ckpt", required=True, help="BERT gap checkpoint dir (model.pt, tokenizer/, restoration_vocab.json)")
    ap.add_argument("--out", required=True, help="output / cache directory")
    ap.add_argument("--whisper", default=DEFAULT_MODEL)
    ap.add_argument("--lex", default=str(ROOT / "data/raw/switchboard_ms98/sw-ms98-dict.text"))
    ap.add_argument("--limit-s", type=float, default=0.0, help="only the first N seconds")
    ap.add_argument("--min-gap-phones", type=int, default=1)
    ap.add_argument("--max-gap-phones", type=int, default=30)
    ap.add_argument("--min-gap-prob", type=float, default=0.0, help="drop gaps whose mean phone posterior is lower")
    ap.add_argument("--min-gap-dur", type=float, default=GapParams.min_dur,
                    help="drop gaps whose estimated duration (s) is shorter")
    ap.add_argument("--keep-partial", action="store_true",
                    help="also insert PARTIAL_WORD restorations into the transcript (skipped by default)")
    ap.add_argument("--force", action="store_true", help="recompute cached VAD / ASR / phone stages")
    ap.add_argument("--reference", default=None, help="(aligned) .rsml of the same audio, for the WER sanity check")
    ap.add_argument("--samples", type=int, default=12, help="restored gaps to print")
    a = ap.parse_args()

    cfg = Config(ckpt=Path(a.ckpt), lexicon=Path(a.lex), phonemap=ROOT / "configs/phonemap.yaml", whisper=a.whisper,
                 limit_s=a.limit_s, force=a.force,
                 skip_types=frozenset() if a.keep_partial else frozenset({"PARTIAL_WORD"}),
                 gap=GapParams(min_phones=a.min_gap_phones, max_phones=a.max_gap_phones, min_prob=a.min_gap_prob,
                             min_dur=a.min_gap_dur))
    out = Path(a.out)
    summary, tokens = run(Path(a.audio), out, cfg)
    print(json.dumps(summary, indent=2))

    shown = 0
    print("\n--- restored gaps (left context [restored] right context) ---")
    data = json.loads((out / "restored.json").read_text())
    for g in data["gaps"]:
        if not g["pred"] or shown >= a.samples:
            continue
        pred = " + ".join(f"{u['type']}:{u['rest']}" for u in g["pred"])
        print(f"{' '.join(g['left'])} [[{' '.join(g['phones'])}]] {' '.join(g['right'])}\n      -> {pred}")
        shown += 1

    if a.reference:
        verb, flu = gold_words(Path(a.reference), a.limit_s)
        asr = (out / "asr.txt").read_text().strip()
        res = (out / "restored.txt").read_text().strip()
        print(f"\n--- WER sanity check ({len(verb)} verbatim / {len(flu)} fluent gold words) ---")
        print(f"ASR vs verbatim gold:       {wer(verb, asr):.3f}")
        print(f"restored vs verbatim gold:  {wer(verb, res):.3f}")
        print(f"ASR vs fluent gold:         {wer(flu, asr):.3f}")
        print(f"restored vs fluent gold:    {wer(flu, res):.3f}")


if __name__ == "__main__":
    main()
