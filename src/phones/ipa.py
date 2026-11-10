"""Map espeak-IPA phone-recognizer tokens to the Switchboard 42-phone set."""
from __future__ import annotations

import unicodedata
from pathlib import Path

import yaml

SPECIAL = {"<pad>", "<s>", "</s>", "<unk>"}
# characters that only modify a base symbol: length, tone digits, palatalisation, aspiration, ...
_MODIFIERS = set("ːˑʲʰʷˤ^.:[]\"'1234567890ᵝ")


class IpaMap:
    def __init__(self, path: str | Path) -> None:
        cfg = yaml.safe_load(Path(path).read_text())
        self.table: dict[str, list[str]] = {k: list(v) for k, v in cfg["map"].items()}
        self.dropped: dict[str, int] = {}

    @staticmethod
    def _strip(tok: str) -> str:
        """Remove modifiers and combining marks; keep base letters."""
        out = []
        for ch in tok:
            if ch in _MODIFIERS or unicodedata.category(ch) in ("Mn", "Lm", "Sk"):
                continue
            out.append(ch)
        return "".join(out)

    def lookup(self, tok: str) -> list[str]:
        """Switchboard phones for one recognizer token ([] = no counterpart)."""
        if tok in SPECIAL:
            return []
        if tok in self.table:
            return self.table[tok]
        base = self._strip(tok)
        if base in self.table:
            return self.table[base]
        # r-coloured / compound tokens: map piecewise when every piece is known
        if len(base) > 1 and all(c in self.table for c in base):
            return [p for c in base for p in self.table[c]]
        self.dropped[tok] = self.dropped.get(tok, 0) + 1
        return []

    def map_sequence(self, toks: list[str]) -> list[str]:
        out: list[str] = []
        for t in toks:
            out.extend(self.lookup(t))
        return out

    def covers(self, vocab: list[str]) -> dict[str, int]:
        """Counts of vocabulary tokens by outcome: 'mapped', 'special', 'dropped'."""
        res = {"mapped": 0, "special": 0, "dropped": 0}
        for t in vocab:
            if t in SPECIAL:
                res["special"] += 1
            elif self.lookup(t):
                res["mapped"] += 1
            else:
                res["dropped"] += 1
        return res
