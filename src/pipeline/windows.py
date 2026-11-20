"""Cut a chunk's ASR words and recognized phones into the windows the sequence model reads.

A window holds consecutive ASR words, their G2P phones, and the recognized phones that were heard while those
words (and the pause before the first of them) were spoken. This is the only place where time is used, and only
to *split* the phone stream between windows: the model itself receives no times and no alignment.
"""
from __future__ import annotations

from dataclasses import dataclass

from model.seq import BUDGET, word_cost

from .asr import Word
from .phonerec import RecPhone


@dataclass
class Window:
    words: list[Word]
    exp: list[list[str]]          # G2P phones of every word
    rec: list[str]                # recognized phones of the window (Switchboard phone set)
    t0: float                     # time span covered (start of the pause before the first word ... end of the window)
    t1: float
    final: bool                   # last window of its chunk: the slot after its last word is predicted too

    def example(self, sid: str, k: int) -> dict:
        n = len(self.words)
        return {"id": f"{sid}:w{k}", "sid": sid, "words": [w.text for w in self.words], "exp": self.exp,
                "rec": self.rec, "slots": [[] for _ in range(n + 1)], "wtags": [0] * n}


def _end(costs: list[float], pauses: list[float], a: int, max_words: int, budget: float) -> int:
    """End of the window starting at ``a``: as many words as fit, then moved back (by up to 5 words) to the
    longest pause so that a window rarely ends in the middle of a phrase."""
    b, c = a, 0.0
    while b < len(costs) and b - a < max_words and (b == a or c + costs[b] <= budget):
        c += costs[b]
        b += 1
    if b < len(costs) and b - a > 6:
        b = max(range(b - 5, b + 1), key=lambda k: pauses[k])
    return b


def make_windows(words: list[Word], rec: list[RecPhone], g2p, t_start: float, t_end: float,
                 max_words: int = 20, budget: float = BUDGET) -> list[Window]:
    """Windows of one VAD chunk (``t_start`` / ``t_end``: the chunk's span). A chunk without words but with
    recognized phones (a lone filler) gives one empty window, so the model can still restore it."""
    exp = [g2p.phones(w.text) for w in words]
    costs = [word_cost(len(e)) for e in exp]
    # pauses[k]: silence between word k-1 and word k (index 0 unused)
    pauses = [0.0] + [words[k].start - words[k - 1].end for k in range(1, len(words))]
    spans: list[tuple[int, int]] = []
    a = 0
    while a < len(words):
        b = _end(costs, pauses, a, max_words, budget)
        spans.append((a, b))
        a = b
    if not spans:
        return [Window([], [], [r.ph for r in rec], t_start, t_end, True)] if rec else []

    # a recognized phone belongs to the window whose time span holds its centre; the cut between two windows
    # lies midway between the last word of one and the first of the next
    cuts = [(words[b - 1].end + words[b].start) / 2 for _, b in spans[:-1]]
    bounds = [t_start] + cuts + [t_end]
    out: list[Window] = []
    for k, (a, b) in enumerate(spans):
        lo, hi = bounds[k], bounds[k + 1]
        ph = [r.ph for r in rec if lo <= (r.start + r.end) / 2 < hi or (k == len(spans) - 1 and (r.start + r.end) / 2 >= lo)]
        out.append(Window(words[a:b], exp[a:b], ph, lo, hi, k == len(spans) - 1))
    return out
