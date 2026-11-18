"""Audio loading and voice-activity segmentation (the "VAD" stage of the pipeline).

The lecture is cut into speech *chunks* (<= ``max_len`` s, joined across short pauses).
Every later stage (ASR, phone recognition, alignment) works chunk by chunk, so a chunk is
the unit within which expected and recognized phones are aligned.
"""
from __future__ import annotations

import subprocess
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SR = 16000


@dataclass(frozen=True)
class Chunk:
    idx: int
    start: float
    end: float

    def samples(self, x: np.ndarray) -> np.ndarray:
        return x[int(self.start * SR):int(self.end * SR)]


def to_16k_mono(src: str | Path, dst: str | Path) -> Path:
    """Convert any audio file readable by ffmpeg to 16 kHz mono 16-bit PCM wav (no-op if present)."""
    src, dst = Path(src), Path(dst)
    if dst.exists():
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-ac", "1", "-ar", str(SR),
                    "-sample_fmt", "s16", str(dst)], check=True)
    return dst


def read_wav(path: str | Path) -> np.ndarray:
    with wave.open(str(path), "rb") as w:
        if not (w.getframerate() == SR and w.getnchannels() == 1 and w.getsampwidth() == 2):
            raise ValueError(f"{path}: expected 16 kHz mono 16-bit wav (use to_16k_mono)")
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    return pcm.astype(np.float32) / 32768.0


def chunk_speech(spans: list[tuple[float, float]], total: float, max_len: float = 28.0,
                 max_gap: float = 0.6, pad: float = 0.15) -> list[Chunk]:
    """Turn VAD speech spans into non-overlapping chunks.

    Consecutive spans are merged while the pause between them is <= ``max_gap`` and the merged
    chunk stays <= ``max_len``; a single span longer than ``max_len`` is cut into equal pieces.
    Each chunk is then padded by up to ``pad`` s on both sides (never into a neighbour, never
    outside ``[0, total]``).
    """
    pieces: list[tuple[float, float]] = []
    for s, e in sorted(spans):
        if e - s > max_len:
            n = int(np.ceil((e - s) / max_len))
            step = (e - s) / n
            pieces += [(s + k * step, s + (k + 1) * step) for k in range(n)]
        else:
            pieces.append((s, e))
    merged: list[list[float]] = []
    for s, e in pieces:
        if merged and s - merged[-1][1] <= max_gap and e - merged[-1][0] <= max_len:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    out: list[Chunk] = []
    for k, (s, e) in enumerate(merged):
        lo = merged[k - 1][1] if k else 0.0
        hi = merged[k + 1][0] if k + 1 < len(merged) else total
        ps = s - min(pad, max(0.0, (s - lo) / 2 if k else s - lo))
        pe = e + min(pad, max(0.0, (hi - e) / 2 if k + 1 < len(merged) else hi - e))
        out.append(Chunk(k, round(max(0.0, ps), 3), round(min(total, pe), 3)))
    return out


def vad_spans(x: np.ndarray, threshold: float = 0.4, min_silence_ms: int = 200,
              min_speech_ms: int = 100, max_len: float = 28.0) -> list[tuple[float, float]]:
    """Speech spans in seconds from Silero VAD.

    The threshold is a little below Silero's default and the minimum speech length short, so
    that breathy, short fillers ("uh") are not discarded before they reach the recognizers.
    """
    import torch
    from silero_vad import get_speech_timestamps, load_silero_vad

    ts = get_speech_timestamps(
        torch.from_numpy(x), load_silero_vad(), sampling_rate=SR, threshold=threshold,
        min_silence_duration_ms=min_silence_ms, min_speech_duration_ms=min_speech_ms,
        max_speech_duration_s=max_len, speech_pad_ms=30, return_seconds=True,
    )
    return [(float(t["start"]), float(t["end"])) for t in ts]
