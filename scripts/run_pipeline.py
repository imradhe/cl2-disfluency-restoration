#!/usr/bin/env python
"""Phase 7: audio -> VAD -> Whisper (transcript -> G2P) + phone recognizer -> BERT -> restored transcript.

Run (offline once the models are cached):
  HF_HOME=.cache/hf .venv/bin/python -I scripts/run_pipeline.py \\
      --audio data/nptel/wav16k/english_001.wav --ckpt ckpt/seq_colab --out results/pipeline/english_001

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
    ap.add_argument("--ckpt", required=True, help="sequence-model checkpoint dir (model.pt, tokenizer/, unit_vocab.json)")
    ap.add_argument("--out", required=True, help="output / cache directory")
    ap.add_argument("--whisper", default=DEFAULT_MODEL)
    ap.add_argument("--lex", default=str(ROOT / "data/raw/switchboard_ms98/sw-ms98-dict.text"))
    ap.add_argument("--limit-s", type=float, default=0.0, help="only the first N seconds")
    ap.add_argument("--max-words", type=int, default=20, help="words per BERT window")
    ap.add_argument("--none-bias", type=float, default=0.0,
                    help="subtracted from the 'nothing to restore' logit: > 0 restores more, < 0 less")
    ap.add_argument("--skip-types", default="", help="comma-separated disfluency types to detect but not insert, e.g. PARTIAL_WORD")
    ap.add_argument("--no-norm-fillers", action="store_true",
                    help="keep Whisper's filler spelling \"ah\" (default: respell it \"uh\", the spelling the model is trained on)")
    ap.add_argument("--force", action="store_true", help="recompute cached VAD / ASR / phone stages")
    ap.add_argument("--reference", default=None, help="(aligned) .rsml of the same audio, for the WER sanity check")
    ap.add_argument("--samples", type=int, default=12, help="restored tokens to print in context")
    a = ap.parse_args()

    cfg = Config(ckpt=Path(a.ckpt), lexicon=Path(a.lex), phonemap=ROOT / "configs/phonemap.yaml", whisper=a.whisper,
                 max_words=a.max_words, none_bias=a.none_bias, limit_s=a.limit_s, force=a.force,
                 norm_fillers=not a.no_norm_fillers,
                 skip_types=frozenset(t for t in a.skip_types.split(",") if t))
    out = Path(a.out)
    summary, tokens = run(Path(a.audio), out, cfg)
    print(json.dumps(summary, indent=2))

    print("\n--- restored tokens in context ---")
    toks = json.loads((out / "restored.json").read_text())["tokens"]
    shown = 0
    for k, t in enumerate(toks):
        if t["source"] != "restored" or shown >= a.samples:
            continue
        ctx = " ".join(f"[{x['label']}:{x['text']}]" if x["source"] == "restored" else x["text"]
                       for x in toks[max(0, k - 6):k + 7])
        print(f"  {ctx}")
        shown += 1

    if a.reference:
        verb, flu = gold_words(Path(a.reference), a.limit_s)
        raw = (out / "asr_raw.txt").read_text().strip()
        asr = (out / "asr.txt").read_text().strip()
        res = (out / "restored.txt").read_text().strip()
        print(f"\n--- WER sanity check ({len(verb)} verbatim / {len(flu)} fluent gold words) ---")
        print(f"Whisper as written vs verbatim gold:  {wer(verb, raw):.3f}")
        if asr != raw:
            print(f"Whisper, fillers respelled:           {wer(verb, asr):.3f}")
        print(f"restored vs verbatim gold:            {wer(verb, res):.3f}")
        print(f"Whisper as written vs fluent gold:    {wer(flu, raw):.3f}")
        print(f"restored vs fluent gold:              {wer(flu, res):.3f}")


if __name__ == "__main__":
    main()
