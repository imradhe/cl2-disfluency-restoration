"""Readers for the ISIP / Mississippi State Switchboard transcription release.

Layout (inside ``swb_ms98_transcriptions/``)::

    <NN>/<conv>/sw<conv>A-ms98-a-trans.text   utterance level
    <NN>/<conv>/sw<conv>A-ms98-a-word.text    word level (incl. [silence] intervals)
    sw-ms98-dict.text                          pronunciation lexicon

Line formats::

    trans: <utt-id> <start> <end> <text...>
    word : <utt-id> <start> <end> <token>
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from .markers import SILENCE, Tok, parse_token

_FILE_RE = re.compile(r"^sw(?P<conv>\d+)(?P<side>[AB])-ms98-a-(?P<kind>trans|word)\.text$")


@dataclass(frozen=True)
class Utt:
    utt_id: str
    start: float
    end: float
    text: str


@dataclass(frozen=True)
class WTok:
    """One aligned token of a speaker side."""
    utt_id: str
    start: float
    end: float
    tok: Tok

    @property
    def dur(self) -> float:
        return self.end - self.start


@dataclass
class Side:
    conv: str            # e.g. "2001"
    side: str            # "A" or "B"
    utts: list[Utt] = field(default_factory=list)
    words: list[WTok] = field(default_factory=list)

    @property
    def sid(self) -> str:
        return f"sw{self.conv}{self.side}"


def read_trans(path: Path) -> list[Utt]:
    out: list[Utt] = []
    with open(path, encoding="latin-1") as fh:
        for line in fh:
            parts = line.rstrip("\n").split(None, 3)
            if len(parts) < 3:
                continue
            text = parts[3] if len(parts) > 3 else ""
            out.append(Utt(parts[0], float(parts[1]), float(parts[2]), text.strip()))
    return out


def read_words(path: Path) -> list[WTok]:
    out: list[WTok] = []
    with open(path, encoding="latin-1") as fh:
        for line in fh:
            parts = line.rstrip("\n").split(None, 3)
            if len(parts) < 4:
                continue
            out.append(WTok(parts[0], float(parts[1]), float(parts[2]), parse_token(parts[3])))
    return out


def iter_sides(root: Path) -> Iterator[Side]:
    """Yield every speaker side found under ``root`` (the transcriptions dir), sorted."""
    pairs: dict[tuple[str, str], dict[str, Path]] = {}
    for p in sorted(root.rglob("sw*-ms98-a-*.text")):
        m = _FILE_RE.match(p.name)
        if m:
            pairs.setdefault((m["conv"], m["side"]), {})[m["kind"]] = p
    for (conv, side), files in sorted(pairs.items()):
        if "trans" not in files or "word" not in files:
            continue
        yield Side(conv, side, read_trans(files["trans"]), read_words(files["word"]))


def pause_before(words: list[WTok], i: int) -> float:
    """Silence (s) immediately before token ``i``: explicit [silence] entries plus gaps."""
    total, j = 0.0, i - 1
    prev_end = words[i].start
    while j >= 0 and words[j].tok.kind == SILENCE:
        total += words[j].dur
        prev_end = words[j].start
        j -= 1
    if j >= 0:
        total += max(0.0, prev_end - words[j].end)
    return total
