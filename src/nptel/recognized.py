"""Recognized phones of an NPTEL lecture (scripts/nptel_phones.py output), cut by time."""
from __future__ import annotations

import bisect
import json
from pathlib import Path


def load_recognized(path: Path) -> tuple[list[float], list[dict]]:
    """Flatten the recognizer output of one lecture into time-sorted tokens (+ their centres)."""
    toks = []
    for line in path.read_text().splitlines():
        toks.extend(json.loads(line)["tokens"])
    toks.sort(key=lambda t: t["start"])
    # segments are padded and may overlap: drop duplicates at (almost) the same time
    uniq, last = [], None
    for t in toks:
        if last and t["tok"] == last["tok"] and abs(t["start"] - last["start"]) < 0.03:
            continue
        uniq.append(t); last = t
    return [(t["start"] + t["end"]) / 2 for t in uniq], uniq


def split_tokens(tokens: list[dict], frac: float) -> int:
    """Index of the first token of the held-out part: the first utterance start at or after ``frac`` of the tokens.
    The noise profile is fitted on ``tokens[:k]`` and the model is evaluated on ``tokens[k:]``."""
    k = int(len(tokens) * frac)
    while 0 < k < len(tokens) and not tokens[k].get("utt_first", True):
        k += 1
    return k


def recognized_phones(centres: list[float], toks: list[dict], ipa, t0: float, t1: float, tol: float = 0.03) -> list[str]:
    lo = bisect.bisect_left(centres, t0 - tol)
    hi = bisect.bisect_right(centres, t1 + tol)
    return ipa.map_sequence([t["tok"] for t in toks[lo:hi]])
