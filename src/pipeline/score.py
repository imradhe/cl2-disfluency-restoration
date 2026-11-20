"""End-to-end disfluency F1 of a pipeline output against the verbatim gold transcript.

The output transcript (Whisper words plus the tokens the model restored) is aligned word by word to the gold
verbatim tokens. The *system's disfluent tokens* are
  * every restored token (the model says which type it is), and
  * the disfluencies Whisper already wrote, found by two transparent rules that every system gets alike:
    lexical fillers (uh, um, ...) and an immediate repeat of the next word (``the the`` tags the first ``the``).
A system token is **correct** when it is aligned to a gold disfluent token with the same word. Then
    precision = correct / system disfluent tokens,  recall = correct / gold disfluent tokens.
``typed`` additionally requires the type to agree (repetition, filler, ...). Whisper alone (no restoration) is the
baseline that the same rules score, so the numbers show what the model adds.
"""
from __future__ import annotations

from collections import Counter

import numpy as np

from .asr import norm_word

GOLD_TYPES = ("FILLER_HESITATION", "FILLER_BACKCHANNEL", "FILLER_DM", "REPETITION", "REPAIR", "FALSE_START", "PARTIAL_WORD")
RULE_FILLERS = frozenset({"uh", "um", "hm", "hmm", "mm", "mhm", "er", "erm"})


def align_idx(a: list[str], b: list[str]) -> list[tuple[int | None, int | None]]:
    """Levenshtein alignment of two word lists as index pairs: (i, j) aligned (equal or substituted), (i, None) a[i]
    unmatched, (None, j) b[j] unmatched. Ties prefer aligning, then skipping ``a``. Vectorised row by row."""
    vocab: dict[str, int] = {}
    A = np.array([vocab.setdefault(x, len(vocab)) for x in a], dtype=np.int64)
    B = np.array([vocab.setdefault(x, len(vocab)) for x in b], dtype=np.int64)
    n, m = len(A), len(B)
    D = np.zeros((n + 1, m + 1), dtype=np.int32)
    D[0] = np.arange(m + 1)
    K = np.arange(m + 1)
    for i in range(1, n + 1):
        E = np.empty(m + 1, dtype=np.int32)
        E[0] = i
        E[1:] = np.minimum(D[i - 1, :-1] + (B != A[i - 1]), D[i - 1, 1:] + 1)
        D[i] = np.minimum.accumulate(E - K) + K                 # allows any number of insertions (left moves) in the row
    ops: list[tuple[int | None, int | None]] = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and D[i, j] == D[i - 1, j - 1] + (A[i - 1] != B[j - 1]):
            ops.append((i - 1, j - 1)); i -= 1; j -= 1
        elif i > 0 and D[i, j] == D[i - 1, j] + 1:
            ops.append((i - 1, None)); i -= 1
        else:
            ops.append((None, j - 1)); j -= 1
    return ops[::-1]


def rule_tags(words: list[str]) -> list[str | None]:
    """Disfluency type of every Whisper word by the two rules (None = fluent)."""
    tags: list[str | None] = [None] * len(words)
    for i, w in enumerate(words):
        if w in RULE_FILLERS:
            tags[i] = "FILLER_HESITATION"
        elif i + 1 < len(words) and words[i + 1] == w:
            tags[i] = "REPETITION"
    return tags


def system_tokens(tokens: list[dict], drop_types: frozenset[str] = frozenset(), rules: bool = True,
                  t_from: float = 0.0) -> list[tuple[str, str | None, str]]:
    """(word, disfluency type or None, source) of every output token in ``restored.json`` order. Restored tokens of
    ``drop_types`` are left out (as if they had not been inserted); ``rules=False`` leaves Whisper's own words
    untagged; ``t_from`` keeps only tokens starting at or after that time."""
    toks = [t for t in tokens if t["start"] >= t_from and not (t["source"] != "asr" and t["label"] in drop_types)]
    asr_idx = [k for k, t in enumerate(toks) if t["source"] == "asr"]
    asr_tags = dict(zip(asr_idx, rule_tags([norm_word(toks[k]["text"]) for k in asr_idx]))) if rules else {}
    out = []
    for k, t in enumerate(toks):
        w = norm_word(t["text"])
        if w:
            out.append((w, asr_tags.get(k) if t["source"] == "asr" else t["label"], "rule" if t["source"] == "asr" else "restored"))
    return out


def prf(tp: int, n_sys: int, n_gold: int) -> dict:
    p, r = (tp / n_sys if n_sys else 0.0), (tp / n_gold if n_gold else 0.0)
    return {"tp": tp, "system": n_sys, "gold": n_gold, "p": round(p, 4), "r": round(r, 4),
            "f1": round(2 * p * r / (p + r), 4) if p + r else 0.0}


def restoration_f1(gold: list[tuple[str, str]], system: list[tuple[str, str | None, str]]) -> dict:
    """The model's own contribution: only the *restored* tokens are scored, against the gold disfluent tokens that
    Whisper's words do not contain (no equal word aligned to them once the restored tokens are removed).
    precision = restored tokens aligned to such a gold token with the same word / restored tokens;
    recall = the same / gold disfluent tokens missing from Whisper's words. ``typed`` also needs the same type."""
    gw = [w for w, _ in gold]
    asr = [s for s in system if s[2] != "restored"]
    aw = [w for w, _, _ in asr]
    present = {gi for si, gi in align_idx(aw, gw) if si is not None and gi is not None and aw[si] == gw[gi]}
    missing = {gi for gi, (_, l) in enumerate(gold) if l in GOLD_TYPES and gi not in present}
    sw = [w for w, _, _ in system]
    tp_any, tp_typed, n_sys = Counter(), Counter(), Counter(s[1] for s in system if s[2] == "restored")
    for si, gi in align_idx(sw, gw):
        if si is None or gi is None or system[si][2] != "restored" or sw[si] != gw[gi] or gi not in missing:
            continue
        tp_any[gold[gi][1]] += 1
        tp_typed[gold[gi][1]] += system[si][1] == gold[gi][1]
    n_miss = Counter(gold[gi][1] for gi in missing)
    return {"untyped": prf(sum(tp_any.values()), sum(n_sys.values()), len(missing)),
            "typed": prf(sum(tp_typed.values()), sum(n_sys.values()), len(missing)),
            "per_type": {t: prf(tp_typed[t], n_sys[t], n_miss[t]) for t in GOLD_TYPES if n_miss[t] or n_sys[t]},
            "gold_missing_from_whisper": len(missing)}


def disfluency_f1(gold: list[tuple[str, str]], system: list[tuple[str, str | None, str]]) -> dict:
    """``gold``: (word, label) of the verbatim transcript, label "O" for fluent words and a type of
    ``GOLD_TYPES`` for disfluent ones (other labels count as fluent). ``system``: see :func:`system_tokens`."""
    gw = [w for w, _ in gold]
    sw = [w for w, _, _ in system]
    tp_any, tp_typed = Counter(), Counter()
    by_source: Counter = Counter()
    present_untagged = 0
    for si, gi in align_idx(sw, gw):
        if si is None or gi is None or sw[si] != gw[gi]:
            continue
        g_lab, (_, s_lab, src) = gold[gi][1], system[si]
        if g_lab not in GOLD_TYPES:
            continue
        if s_lab is None:
            present_untagged += 1                         # the word is in the output but the system did not call it a disfluency
            continue
        tp_any[g_lab] += 1
        by_source[src] += 1
        if s_lab == g_lab:
            tp_typed[g_lab] += 1
    n_gold, n_sys = Counter(l for _, l in gold if l in GOLD_TYPES), Counter(l for _, l, _ in system if l is not None)
    res = {"untyped": prf(sum(tp_any.values()), sum(n_sys.values()), sum(n_gold.values())),
           "typed": prf(sum(tp_typed.values()), sum(n_sys.values()), sum(n_gold.values())),
           "per_type": {t: prf(tp_typed[t], n_sys[t], n_gold[t]) for t in GOLD_TYPES if n_gold[t] or n_sys[t]},
           "correct_by_source": dict(by_source), "gold_present_but_untagged": present_untagged}
    return res
