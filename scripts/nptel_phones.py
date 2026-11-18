#!/usr/bin/env python
"""Phase 6: recognize phones on NPTEL audio with wav2vec2-xlsr-53-espeak-cv-ft (CTC).

For each annotated segment of an aligned .rsml (plus padding) the recognizer is run and the
greedy CTC output is collapsed into phone tokens with start/end times (20 ms frames) and a
mean posterior. Output: data/nptel/phones/<stem>.jsonl, one line per segment:
  {"seg": 3, "t0": 12.3, "t1": 18.9, "tokens": [{"tok": "ə", "start": 12.46, "end": 12.50, "p": 0.93}, ...]}

Run: HF_HOME=.cache/hf HF_HUB_OFFLINE=1 .venv/bin/python -I scripts/nptel_phones.py \
        --audio data/nptel/raw/english_001/english_001.wav --aligned data/nptel/raw/english_001/english_001.aligned.rsml
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import wave
from pathlib import Path

import numpy as np
import torch
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForCTC

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from nptel.rsml import parse_aligned  # noqa: E402
from pipeline.phonerec import MODEL, collapse  # noqa: E402


def to_16k_mono(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-ac", "1", "-ar", "16000",
                    "-sample_fmt", "s16", str(dst)], check=True)


def read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as w:
        assert w.getframerate() == 16000 and w.getnchannels() == 1 and w.getsampwidth() == 2
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    return pcm.astype(np.float32) / 32768.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True)
    ap.add_argument("--aligned", required=True, help="aligned .rsml (segment boundaries)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--pad", type=float, default=0.3)
    ap.add_argument("--limit", type=int, default=0, help="only the first N segments (smoke test)")
    a = ap.parse_args()

    audio = Path(a.audio)
    stem = audio.stem
    wav16 = ROOT / "data/nptel/wav16k" / f"{stem}.wav"
    t0 = time.time()
    to_16k_mono(audio, wav16)
    x = read_wav(wav16)
    print(f"audio {len(x) / 16000:.1f}s @16k mono ({time.time() - t0:.1f}s)", flush=True)

    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    fe = Wav2Vec2FeatureExtractor.from_pretrained(MODEL)
    model = Wav2Vec2ForCTC.from_pretrained(MODEL).to(dev).eval()
    vocab_map = json.loads((Path(model.config._name_or_path) / "vocab.json").read_text()) \
        if (Path(model.config._name_or_path) / "vocab.json").exists() else None
    if vocab_map is None:
        from huggingface_hub import hf_hub_download
        vocab_map = json.loads(Path(hf_hub_download(MODEL, "vocab.json")).read_text())
    vocab = [None] * (max(vocab_map.values()) + 1)
    for tok, i in vocab_map.items():
        vocab[i] = tok

    segs = parse_aligned(a.aligned)
    if a.limit:
        segs = segs[:a.limit]
    out_path = Path(a.out) if a.out else ROOT / "data/nptel/phones" / f"{stem}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_tok = 0
    with open(out_path, "w") as fh, torch.no_grad():
        for k, aseg in enumerate(segs):
            s0 = max(0.0, aseg.seg.start - a.pad)
            s1 = min(len(x) / 16000, aseg.seg.end + a.pad)
            chunk = x[int(s0 * 16000):int(s1 * 16000)]
            inp = fe(chunk, sampling_rate=16000, return_tensors="pt")
            logits = model(inp.input_values.to(dev)).logits[0].float()
            probs = logits.softmax(-1)
            ids = probs.argmax(-1).cpu().numpy()
            pmax = probs.max(-1).values.cpu().numpy()
            toks = collapse(ids, pmax, vocab, s0)
            n_tok += len(toks)
            fh.write(json.dumps({"seg": aseg.seg.idx, "t0": round(s0, 3), "t1": round(s1, 3), "tokens": toks}) + "\n")
            if (k + 1) % 50 == 0:
                print(f"  {k + 1}/{len(segs)} segments, {n_tok} phone tokens, {time.time() - t0:.0f}s", flush=True)
    print(f"wrote {out_path}: {len(segs)} segments, {n_tok} phone tokens, total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
