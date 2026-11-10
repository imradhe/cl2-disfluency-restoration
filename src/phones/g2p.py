"""Grapheme-to-phoneme in the Switchboard 42-phone set.

Order of preference: (1) the sw-ms98 lexicon (so tokens look exactly like training),
(2) espeak-ng IPA mapped through ``configs/phonemap.yaml``. Results are cached.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

from .ipa import IpaMap
from .lexicon import Lexicon

_STRESS = "ˈˌ"


class G2P:
    def __init__(self, lexicon: Lexicon, ipa: IpaMap, espeak: str = "espeak-ng", voice: str = "en-us") -> None:
        self.lex, self.ipa, self.espeak, self.voice = lexicon, ipa, espeak, voice
        self._cache: dict[str, list[str]] = {}
        self._keys = sorted(self.ipa.table, key=len, reverse=True)
        self.n_lex = self.n_espeak = 0

    def espeak_ipa(self, word: str) -> str:
        out = subprocess.run(
            [self.espeak, "-q", "--ipa", "-v", self.voice, word],
            capture_output=True, text=True, check=False,
        ).stdout
        return out.replace("\n", " ").strip()

    def ipa_to_phones(self, ipa: str) -> list[str]:
        s = "".join(c for c in ipa if c not in _STRESS and not c.isspace() and c not in "_‿")
        phones: list[str] = []
        i = 0
        while i < len(s):
            for k in self._keys:
                if s.startswith(k, i):
                    phones.extend(self.ipa.table[k])
                    i += len(k)
                    break
            else:
                phones.extend(self.ipa.lookup(s[i]))
                i += 1
        return phones

    def phones(self, word: str) -> list[str]:
        w = word.lower()
        if w in self._cache:
            return self._cache[w]
        prons = self.lex.get(w)
        if prons:
            res, self.n_lex = prons[0], self.n_lex + 1
        else:
            res, self.n_espeak = self.ipa_to_phones(self.espeak_ipa(w)), self.n_espeak + 1
        self._cache[w] = res
        return res

    @classmethod
    def from_files(cls, lexicon_path: str | Path, phonemap_path: str | Path) -> "G2P":
        return cls(Lexicon.from_file(lexicon_path), IpaMap(phonemap_path))
