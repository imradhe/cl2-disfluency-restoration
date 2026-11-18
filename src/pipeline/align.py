"""Align the phones the ASR words *should* contain against the phones the recognizer *heard*.

Recognized phones that no expected phone accounts for are insertions. An insertion run that sits
on a word boundary is a **gap**: speech the ASR transcript does not contain (a filler, a repeated
word, a false start ...), later classified and restored by the BERT gap model.

The alignment is a global edit-distance alignment (Needleman-Wunsch) whose costs encode what is
known about the two sources:

* substitutions are cheap between acoustically confusable phones (``phones.noise``) and dearer
  between unrelated ones, so a missed phone is not mistaken for a filler;
* a match between an expected and a recognized phone is penalised by their time difference
  (expected phone times are interpolated inside Whisper's word spans), which keeps a repeated
  word from sliding onto the wrong copy;
* inserting *inside* a word costs extra, so ties between "gap before the word" and "gap in the
  middle of the word" resolve to the word boundary.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from phones.lexicon import SWB_PHONES
from phones.noise import CONFUSION_GROUPS

VOWELS = {"aa", "ae", "ah", "ao", "aw", "ax", "ay", "eh", "er", "ey", "ih", "iy", "ow", "oy", "uh", "uw"}
_PH = {p: i for i, p in enumerate(SWB_PHONES)}


@dataclass(frozen=True)
class AlignParams:
    c_del: float = 1.0           # an expected phone the recognizer missed
    c_ins: float = 1.0           # a recognized phone no expected phone explains
    c_ins_inside: float = 0.5    # extra cost of an insertion inside a word
    c_sub_near: float = 0.4      # confusable pair (p/b, ih/iy ...)
    c_sub_class: float = 1.0     # vowel/vowel or consonant/consonant
    c_sub_far: float = 1.6       # vowel vs consonant (still < c_ins + c_del)
    w_time: float = 1.0          # cost per second of time mismatch ...
    time_cap: float = 1.0        # ... capped at this many seconds


def _sub_table(p: AlignParams) -> np.ndarray:
    """(n+1) x (n+1) substitution costs; the extra index is any phone outside the 42-phone set."""
    n = len(SWB_PHONES)
    near = {(a, b) for g in CONFUSION_GROUPS for a in g for b in g if a != b}
    tab = np.full((n + 1, n + 1), p.c_sub_far)
    for a in SWB_PHONES:
        for b in SWB_PHONES:
            if a == b:
                c = 0.0
            elif (a, b) in near:
                c = p.c_sub_near
            elif (a in VOWELS) == (b in VOWELS):
                c = p.c_sub_class
            else:
                c = p.c_sub_far
            tab[_PH[a], _PH[b]] = c
    tab[n, n] = p.c_sub_class
    return tab


def fit_cost(variant: list[str], window: list[str], params: AlignParams = AlignParams()) -> float:
    """Cheapest alignment of the whole ``variant`` against any contiguous stretch of ``window``
    (phones of the window outside that stretch are free): how well a pronunciation is supported
    by what was heard around the word. Time is ignored."""
    n, m = len(variant), len(window)
    if n == 0:
        return 0.0
    if m == 0:
        return n * params.c_del
    other = len(SWB_PHONES)
    sub = _sub_table(params)[np.ix_([_PH.get(p, other) for p in variant], [_PH.get(p, other) for p in window])]
    prev = np.zeros(m + 1)                            # row 0: the stretch may start anywhere
    for i in range(1, n + 1):
        base = np.empty(m + 1)
        base[0] = prev[0] + params.c_del
        base[1:] = np.minimum(prev[:-1] + sub[i - 1], prev[1:] + params.c_del)
        j = np.arange(m + 1)
        prev = j * params.c_ins + np.minimum.accumulate(base - j * params.c_ins)
    return float(prev.min())                          # ... and end anywhere


def align_phones(exp_ph: list[str], exp_word: list[int], exp_t: list[float],
                 rec_ph: list[str], rec_t: list[float], params: AlignParams = AlignParams(),
                 ) -> list[tuple[str, int | None, int | None]]:
    """Edit-distance alignment. Returns ops ``("M"|"S", i, j)``, ``("D", i, None)``, ``("I", None, j)``
    in order, where ``i`` indexes the expected and ``j`` the recognized phones."""
    n, m = len(exp_ph), len(rec_ph)
    if n == 0 or m == 0:
        return [("D", i, None) for i in range(n)] + [("I", None, j) for j in range(m)]
    other = len(SWB_PHONES)
    ei = np.array([_PH.get(p, other) for p in exp_ph])
    ri = np.array([_PH.get(p, other) for p in rec_ph])
    sub = _sub_table(params)[np.ix_(ei, ri)]
    dt = np.abs(np.asarray(exp_t)[:, None] - np.asarray(rec_t)[None, :])
    sub = sub + params.w_time * np.minimum(dt, params.time_cap)

    # c_ins[i]: cost of inserting a recognized phone after i expected phones have been consumed
    c_ins = np.full(n + 1, params.c_ins)
    for i in range(1, n):
        if exp_word[i - 1] == exp_word[i]:
            c_ins[i] += params.c_ins_inside

    D = np.zeros((n + 1, m + 1))
    D[0, :] = np.arange(m + 1) * c_ins[0]
    j_idx = np.arange(m + 1)
    for i in range(1, n + 1):
        base = np.empty(m + 1)
        base[0] = D[i - 1, 0] + params.c_del
        base[1:] = np.minimum(D[i - 1, :-1] + sub[i - 1], D[i - 1, 1:] + params.c_del)
        c = c_ins[i]
        D[i] = j_idx * c + np.minimum.accumulate(base - j_idx * c)

    ops: list[tuple[str, int | None, int | None]] = []
    i, j, eps = n, m, 1e-9
    while i > 0 or j > 0:
        if i > 0 and j > 0 and abs(D[i, j] - (D[i - 1, j - 1] + sub[i - 1, j - 1])) < eps:
            ops.append(("M" if exp_ph[i - 1] == rec_ph[j - 1] else "S", i - 1, j - 1))
            i, j = i - 1, j - 1
        elif j > 0 and abs(D[i, j] - (D[i, j - 1] + c_ins[i])) < eps:
            ops.append(("I", None, j - 1))
            j -= 1
        else:
            ops.append(("D", i - 1, None))
            i -= 1
    ops.reverse()
    return ops


def insertion_slots(ops: list[tuple[str, int | None, int | None]], exp_word: list[int], n_words: int,
                    ) -> tuple[dict[int, list[int]], int]:
    """Group inserted recognized phones by word boundary.

    Returns ``({slot: [rec phone indices]}, n_inside)``: slot ``k`` means "before word ``k``" (``n_words``
    = after the last word). Insertions that sit between two phones of the *same* word are not gaps;
    they are counted in ``n_inside`` and dropped.
    """
    slots: dict[int, list[int]] = {}
    n_inside = 0
    prev_word: int | None = None
    pending: list[int] = []

    def flush(next_word: int | None) -> None:
        nonlocal n_inside
        if not pending:
            return
        if prev_word is not None and next_word is not None and prev_word == next_word:
            n_inside += len(pending)
        else:
            slots.setdefault(next_word if next_word is not None else n_words, []).extend(pending)
        pending.clear()

    for kind, i, j in ops:
        if kind == "I":
            pending.append(j)                       # type: ignore[arg-type]
        else:
            w = exp_word[i]                         # type: ignore[index]
            flush(w)
            prev_word = w
    flush(None)
    return slots, n_inside
