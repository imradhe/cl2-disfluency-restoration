"""ASR stage: fluent-ish word transcript with word timestamps (Whisper via mlx-whisper).

Whisper is deliberately used *without* a prompt: it tends to drop fillers, repetitions and
false starts, which is exactly the gap that the phone evidence + BERT restoration fills.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

DEFAULT_MODEL = "mlx-community/whisper-large-v3-turbo"


@dataclass
class Word:
    text: str
    start: float
    end: float
    prob: float = 1.0

    def to_json(self) -> dict:
        return {"text": self.text, "start": round(self.start, 3), "end": round(self.end, 3), "prob": round(self.prob, 3)}


_EDGE = re.compile(r"^[^\w']+|[^\w']+$")


def norm_word(raw: str) -> str:
    """Lower-case and strip edge punctuation (inner apostrophes / hyphens are kept)."""
    w = _EDGE.sub("", raw.strip().lower()).strip("'")
    return w


def words_from_segments(segments: list[dict], offset: float = 0.0) -> list[Word]:
    """Flatten Whisper ``segments[*].words`` into normalised :class:`Word` s (times shifted by ``offset``)."""
    out: list[Word] = []
    for seg in segments:
        for w in seg.get("words", []):
            text = norm_word(w["word"])
            if not text:
                continue
            start, end = offset + float(w["start"]), offset + float(w["end"])
            out.append(Word(text, start, max(end, start), float(w.get("probability", 1.0))))
    return out


# Whisper writes a filled pause as "ah" (45 times, never "uh", in the 32-minute NPTEL lecture whose gold writes it
# "uh" 170 times and never "ah"). The restoration model is trained on Switchboard spelling ("uh" / "um"), so a lone
# "ah" is respelled before windowing. Anything else is left alone: there is no evidence for other mappings.
FILLER_SPELLING = {"ah": "uh"}


def normalize_fillers(words: list[Word], spelling: dict[str, str] | None = None) -> tuple[list[Word], int]:
    """New words with Whisper's filler spellings replaced, and the number of words changed (input untouched)."""
    spelling = FILLER_SPELLING if spelling is None else spelling
    out, n = [], 0
    for w in words:
        s = spelling.get(w.text)
        if s is None:
            out.append(w)
        else:
            out.append(Word(s, w.start, w.end, w.prob))
            n += 1
    return out, n


class MlxWhisper:
    def __init__(self, repo: str = DEFAULT_MODEL) -> None:
        self.repo = repo

    def transcribe(self, audio: np.ndarray, offset: float = 0.0) -> list[Word]:
        import mlx_whisper

        res = mlx_whisper.transcribe(
            audio, path_or_hf_repo=self.repo, word_timestamps=True, language="en",
            condition_on_previous_text=False, temperature=0.0, verbose=None,
        )
        return words_from_segments(res["segments"], offset)
