"""Empirical model of the phone recognizer's errors, fitted on recognizer-vs-G2P alignments of real audio.

``PhoneNoise`` corrupts every phone independently with made-up rates. Measured on a verified NPTEL lecture the
real recognizer (wav2vec2-xlsr-53-espeak-cv-ft) behaves differently in ways that matter for this task: it writes
*nothing* for about half of the filled-pause "uh"s (its training labels are phonemized written text, which has no
fillers), turns the ones it does write into ax / aa / ah, drops some whole short words, and emits stray phones
between words. ``stretch_outcomes`` aligns the G2P phones of a stretch of consecutive gold tokens (an utterance)
with the recognized phones of the same stretch, ``fit_profile`` measures the channel from those alignments and
``ProfileNoise`` replays it on training windows, token by token.

The alignment is done on the whole stretch, not token by token: the recognizer's phone times are off by a few tens
of ms against the forced word boundaries, and a per-token alignment books such a shifted phone as a deletion in one
word *and* an insertion in the next (on the lecture it overstated deletions about 2x and strays about 8x).
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

from phones.noise import PhoneNoise

HES = "FILLER_HESITATION"
HES_REF = ("ah",)                        # the one-phone "uh"; the hesitation channel below is specific to it
MIN_HES = 20                             # fewer hesitation samples than this: no hesitation-specific channel
MAX_ORPHAN = 3                           # stray runs are replayed with 1..3 phones
VOWELS = frozenset("aa ae ah ao aw ax ay eh er ey ih iy ow oy uh uw".split())
# An "uh" counts as *written* only if the phone aligned to it is a vowel. A consonant aligned to it (d, s, z, hh ...)
# is a neighbouring word's phone that the aligner paired with the missing "uh" -- not a rendering of it.


def bucket(n: int) -> str:
    return "1" if n <= 1 else "2" if n == 2 else "3+"


def align(ref: list[str], hyp: list[str]) -> list[tuple[str | None, str | None]]:
    """Levenshtein alignment: (r, h) match when equal, substitution when different, (r, None) deletion, (None, h)
    insertion. Ties prefer substitution over deletion over insertion."""
    n, m = len(ref), len(hyp)
    D = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        D[i][0] = i
    for j in range(1, m + 1):
        D[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            D[i][j] = min(D[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]), D[i - 1][j] + 1, D[i][j - 1] + 1)
    ops: list[tuple[str | None, str | None]] = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and D[i][j] == D[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]):
            ops.append((ref[i - 1], hyp[j - 1])); i -= 1; j -= 1
        elif i > 0 and D[i][j] == D[i - 1][j] + 1:
            ops.append((ref[i - 1], None)); i -= 1
        else:
            ops.append((None, hyp[j - 1])); j -= 1
    return ops[::-1]


def stretch_outcomes(tokens: list[tuple[str, list[str]]], hyp: list[str]) -> list[dict]:
    """Align the G2P phones of consecutive tokens ``[(label, phones)]`` with the recognized phones ``hyp`` of the
    same stretch. Per token: ``out[k]`` = what the recognizer made of its k-th phone (itself, another phone, or None
    when deleted) and ``ins[k]`` = phones the recognizer added right after it. Insertions before the first phone of
    the stretch are ignored (stretch edge)."""
    rows = [{"label": lab, "ref": list(ref), "out": [None] * len(ref), "ins": [[] for _ in ref]} for lab, ref in tokens]
    flat = [(t, k) for t, (_, ref) in enumerate(tokens) for k in range(len(ref))]
    ref_flat = [p for _, ref in tokens for p in ref]
    ri, last = -1, None
    for r, h in align(ref_flat, list(hyp)):
        if r is not None:
            ri += 1
            last = flat[ri]
            rows[last[0]]["out"][last[1]] = h
        elif last is not None:
            rows[last[0]]["ins"][last[1]].append(h)
    return rows


def fit_profile(rows: list, prior: float = 8.0, sub_prior: float = 2.0, meta: dict | None = None) -> dict:
    """``rows``: ``stretch_outcomes`` of the fitting stretches, flattened (a (label, G2P phones, recognized phones)
    tuple is accepted as a one-token stretch). A token is *dropped* when all its phones were deleted. Per-phone
    rates, measured on the tokens that survive, are shrunk towards the global rate (``prior`` pseudo-phones);
    substitution targets get ``sub_prior`` pseudo-counts spread over the hand-made confusion groups of
    ``PhoneNoise``. Insertions after the last phone of a token are "strays" (own rate, run length and pool);
    insertions inside a token are per-phone."""
    neigh = PhoneNoise()._neigh
    rows = [stretch_outcomes([(r[0], r[1])], r[2])[0] if isinstance(r, tuple) else r for r in rows]
    tok_n, tok_empty = Counter(), Counter()
    hes_n, hes_empty, hes_str = 0, 0, Counter()
    n_ref, n_mid, n_del, n_sub, n_ins = Counter(), Counter(), Counter(), Counter(), Counter()
    sub_to: dict[str, Counter] = defaultdict(Counter)
    ins_pool: Counter = Counter()
    strays: list[list[str]] = []
    n_tok = 0
    for row in rows:
        ref, out, ins = row["ref"], row["out"], row["ins"]
        if not ref:
            continue
        n_tok += 1
        if ins[-1]:
            strays.append(list(ins[-1]))
        dropped = all(o is None for o in out)
        if row["label"] == HES and tuple(ref) == HES_REF:
            hes_n += 1
            if out[0] in VOWELS:
                hes_str[(out[0],)] += 1
            else:
                hes_empty += 1
            continue
        b = bucket(len(ref))
        tok_n[b] += 1
        if dropped:
            tok_empty[b] += 1
            continue
        for k, r in enumerate(ref):
            n_ref[r] += 1
            if out[k] is None:
                n_del[r] += 1
            elif out[k] != r:
                n_sub[r] += 1
                sub_to[r][out[k]] += 1
            if k < len(ref) - 1:
                n_mid[r] += 1
                if ins[k]:
                    n_ins[r] += len(ins[k])
                    ins_pool.update(ins[k])
    tot, tot_mid = max(1, sum(n_ref.values())), max(1, sum(n_mid.values()))
    g_del, g_sub, g_ins = sum(n_del.values()) / tot, sum(n_sub.values()) / tot, sum(n_ins.values()) / tot_mid

    phones = {}
    for p, n in n_ref.items():
        to = Counter(sub_to[p])
        for q in neigh.get(p, []):
            to[q] += sub_prior / len(neigh[p])
        z = sum(to.values())
        phones[p] = {"n": n, "del": round((n_del[p] + prior * g_del) / (n + prior), 4),
                     "sub": round((n_sub[p] + prior * g_sub) / (n + prior), 4),
                     "ins": round((n_ins[p] + prior * g_ins) / (n_mid[p] + prior), 4),
                     "to": {q: round(c / z, 4) for q, c in to.most_common()} if z else {}}
    prof = {
        "tok_del": {b: round((tok_empty[b] + 0.1) / (tok_n[b] + 5), 4) for b in ("1", "2", "3+")},     # unseen bucket: ~2%
        "global": {"del": round(g_del, 4), "sub": round(g_sub, 4), "ins": round(g_ins, 4)},
        "phones": phones,
        "ins_pool": dict(ins_pool.most_common()) or {"ax": 1},
        "meta": dict(meta or {}, n_tokens=n_tok, n_ref_phones=tot),
    }
    if hes_n >= MIN_HES:
        prof["hes"] = {"n": hes_n, "p_empty": round((hes_empty + 1) / (hes_n + 2), 4),
                       "strings": [[list(s), c] for s, c in hes_str.most_common()]}
    if strays and n_tok:
        lens = Counter(min(len(r), MAX_ORPHAN) for r in strays)
        pool = Counter(p for r in strays for p in r)
        prof["orphan"] = {"p": round(len(strays) / n_tok, 4), "len": {str(k): v for k, v in sorted(lens.items())},
                          "pool": dict(pool.most_common())}
    return prof


def save_profile(prof: dict, path: str | Path) -> None:
    Path(path).write_text(json.dumps(prof, indent=1, ensure_ascii=False))


def load_profile(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


class ProfileNoise(PhoneNoise):
    """Replays a fitted recognizer channel token by token; with probability ``1 - p_profile`` an example gets the
    plain iid ``PhoneNoise`` instead (hedges against a profile fitted on one speaker). Needs the token structure of
    the phone stream (``groups``: token index per phone, ``classes``: label per token); without it, iid noise."""

    def __init__(self, profile: dict, p_profile: float = 0.7, seed: int | None = None, **kw) -> None:
        super().__init__(seed=seed, **kw)
        self.profile, self.p_profile = profile, p_profile
        g = profile["global"]
        self._default = {"del": g["del"], "sub": g["sub"], "ins": g["ins"], "to": {}}
        self._ph = {}
        for p, d in profile["phones"].items():
            to = d["to"] or {q: 1.0 for q in self._neigh.get(p, [])}
            self._ph[p] = (d, list(to), list(to.values()))
        self._ins_keys, self._ins_w = list(profile["ins_pool"]), list(profile["ins_pool"].values())
        hes = profile.get("hes")
        # (a profile whose hesitations were never written at all has no strings to sample: always dropped)
        self._hes = (hes["p_empty"] if hes["strings"] else 1.0, [tuple(s) for s, _ in hes["strings"]],
                     [c for _, c in hes["strings"]]) if hes else None
        orp = profile.get("orphan")
        self._orp = (orp["p"], [int(k) for k in orp["len"]], list(orp["len"].values()),
                     list(orp["pool"]), list(orp["pool"].values())) if orp else None

    def _token(self, label: str, ref: list[str]) -> list[tuple[int, str]]:
        """(index of the ref phone it stems from, recognized phone) for one token."""
        rng = self._rng
        if label == "NOISE":
            return list(enumerate(ref))
        if self._hes and label == HES and tuple(ref) == HES_REF:
            if rng.random() < self._hes[0]:
                return []
            return [(k, p) for k, p in enumerate(rng.choices(self._hes[1], self._hes[2])[0])]
        if rng.random() < self.profile["tok_del"][bucket(len(ref))]:
            return []
        out: list[tuple[int, str]] = []
        for k, r in enumerate(ref):
            d, keys, w = self._ph.get(r, (self._default, [], []))
            u = rng.random()
            if u < d["del"]:
                pass
            elif u < d["del"] + d["sub"] and keys:
                out.append((k, rng.choices(keys, w)[0]))
            else:
                out.append((k, r))
            if k < len(ref) - 1 and rng.random() < d["ins"]:        # inside the token; strays come after it
                out.append((k, rng.choices(self._ins_keys, self._ins_w)[0]))
        return out or [(0, ref[0])]            # a token that survives keeps at least one phone

    def corrupt_idx(self, phones: list[str], groups: list[int] | None = None,
                    classes: list[str] | None = None) -> tuple[list[str], list[int]]:
        if groups is None or classes is None or self._rng.random() >= self.p_profile:
            return super().corrupt_idx(phones)
        assert len(groups) == len(phones), "token groups out of sync with the phone stream"
        rng, out, src = self._rng, [], []
        i, n = 0, len(phones)
        while i < n:
            j = i
            while j + 1 < n and groups[j + 1] == groups[i]:
                j += 1
            for k, p in self._token(classes[groups[i]], phones[i:j + 1]):
                out.append(p)
                src.append(i + min(k, j - i))
            if self._orp and rng.random() < self._orp[0]:           # stray phones between this token and the next
                m = rng.choices(self._orp[1], self._orp[2])[0]
                for p in rng.choices(self._orp[3], self._orp[4], k=m):
                    out.append(p)
                    src.append(j)
            i = j + 1
        return (out, src) if out else (list(phones[:1]), [0] * len(phones[:1]))
