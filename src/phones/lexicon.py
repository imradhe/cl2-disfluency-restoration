"""Mississippi State (sw-ms98) pronunciation lexicon.

Each line is ``<key> <phone> <phone> ...`` where the key is the raw transcript
token. For partial words and mispronunciations the phones are those of the part
that was actually *spoken* (e.g. ``-[a]b[out]- b``). The phone set is a
42-symbol lower-case ARPAbet variant (``ax en el`` included, no stress marks).
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

from swb.markers import Tok

SWB_PHONES = (
    "aa ae ah ao aw ax ay b ch d dh eh el en er ey f g hh ih iy jh k l m n ng ow oy "
    "p r s sh t th uh uw v w y z zh"
).split()


class Lexicon:
    def __init__(self) -> None:
        self._exact: dict[str, list[list[str]]] = {}
        self._lower: dict[str, list[list[str]]] = {}

    @classmethod
    def from_file(cls, path: str | Path) -> "Lexicon":
        lex = cls()
        with open(path, encoding="latin-1") as fh:
            for line in fh:
                s = line.rstrip("\n")
                if not s.strip() or s.lstrip().startswith("#") or s.startswith(" file"):
                    continue
                parts = s.split()
                key, pron = parts[0], parts[1:]
                if not pron:
                    continue
                lex._exact.setdefault(key, []).append(pron)
                lex._lower.setdefault(key.lower(), []).append(pron)
        return lex

    def __len__(self) -> int:
        return len(self._exact)

    def get(self, key: str) -> list[list[str]]:
        """All pronunciations for a raw key (exact case first, then case-insensitive)."""
        return self._exact.get(key) or self._lower.get(key.lower()) or []

    def phones(self, tok: Tok) -> list[str] | None:
        """Canonical (first) pronunciation for a parsed token, or None if unknown."""
        prons = self.get(tok.lex_key)
        if not prons and tok.spoken:
            prons = self.get(tok.spoken)
        return prons[0] if prons else None

    def phone_inventory(self) -> Counter:
        c: Counter = Counter()
        for prons in self._exact.values():
            for p in prons:
                c.update(p)
        return c
