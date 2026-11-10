"""Simulate phone-recognizer errors on clean lexicon phones.

Training inputs are clean phones from the lexicon, but at test time phones come
from a CTC phone recognizer (deletions, substitutions between acoustically
similar phones, spurious insertions). ``PhoneNoise`` corrupts a phone sequence
accordingly. Rates are defaults; ``calibrate`` (Phase 6) re-estimates them from
recognizer-vs-G2P alignments on real audio that is *not* the test set.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

# Acoustically confusable groups (SWB 42-phone set).
CONFUSION_GROUPS = [
    ["iy", "ih"], ["eh", "ae", "ah"], ["aa", "ao", "ah"], ["uw", "uh"], ["ax", "ah", "ih", "er"],
    ["ey", "eh"], ["ay", "aa", "ey"], ["ow", "ao"], ["aw", "aa", "ow"], ["oy", "ao"],
    ["p", "b"], ["t", "d"], ["k", "g"], ["f", "v"], ["th", "dh", "f"], ["s", "z"], ["sh", "zh", "s"],
    ["ch", "jh", "sh"], ["m", "n"], ["n", "ng"], ["l", "el", "r"], ["er", "r"], ["en", "n"],
    ["w", "v"], ["y", "iy"], ["hh"],
]
FILLER_INSERT_POOL = ["ah", "m", "hh", "ax", "n", "ih"]     # what spurious insertions tend to be


@dataclass
class PhoneNoise:
    p_del: float = 0.08
    p_sub: float = 0.14
    p_ins: float = 0.04
    seed: int | None = None
    _rng: random.Random = field(init=False, repr=False)
    _neigh: dict[str, list[str]] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)
        neigh: dict[str, list[str]] = {}
        for grp in CONFUSION_GROUPS:
            for p in grp:
                neigh.setdefault(p, [])
                for q in grp:
                    if q != p and q not in neigh[p]:
                        neigh[p].append(q)
        self._neigh = neigh

    def corrupt(self, phones: list[str]) -> list[str]:
        rng, out = self._rng, []
        for p in phones:
            r = rng.random()
            if r < self.p_del and len(phones) > 1:
                continue
            if r < self.p_del + self.p_sub and self._neigh.get(p):
                out.append(rng.choice(self._neigh[p]))
            else:
                out.append(p)
            if rng.random() < self.p_ins:
                out.append(rng.choice(FILLER_INSERT_POOL))
        return out or list(phones[:1])

    def spurious(self, max_len: int = 3) -> list[str]:
        """A short phone string the recognizer 'hears' where there was only noise."""
        n = self._rng.randint(1, max_len)
        return [self._rng.choice(FILLER_INSERT_POOL + ["s", "t", "k", "p"]) for _ in range(n)]
