"""Parse single ISIP / Mississippi State Switchboard transcript tokens.

Conventions (ISIP transcription guidelines, verified against the released data):

* partial word, tail cut        ``an[y]-``        spoken "an",  intended "any"
* partial word, head cut        ``-[th]at``       spoken "at",  intended "that"
* partial word, both cut        ``-[a]b[out]-``   spoken "b",   intended "about"
* partial with apostrophe       ``i['m]-``        spoken "i",   intended "i'm"
* bare partial (no bracket)     ``i-``            spoken "i",   intended unknown
* mispronunciation              ``[splace/space]``
* mispronounced partial         ``[cam[pany]-/company]``  (brackets may nest)
* noises                        ``[laughter] [noise] [vocalized-noise] [silence]``
* noise over a word             ``[laughter-don't]``
* coinage                       ``{weatherwise}``
* pronunciation variant         ``about_1``       (lexicon key; normalized word "about")
* asides                        ``<b_aside> <e_aside>``
* unclear speech brackets       ``(( ))``         (rare; ignored)

``lex_key`` is the raw token (case preserved): the pronunciation lexicon is
keyed by the raw form, including the bracket syntax, and lists the phones of
the part that was actually *spoken*.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# token kinds
WORD = "WORD"
PARTIAL = "PARTIAL"
MISPRON = "MISPRON"
NOISE = "NOISE"            # [laughter] [noise] [vocalized-noise]
SILENCE = "SILENCE"        # [silence]
NOISE_WORD = "NOISE_WORD"  # [laughter-don't]: a real word spoken through noise
COINAGE = "COINAGE"
ASIDE = "ASIDE"
UNCLEAR = "UNCLEAR"

NON_SPEECH_KINDS = {NOISE, SILENCE, ASIDE, UNCLEAR}

_NOISE_RE = re.compile(r"^\[(laughter|noise|vocalized-noise)\]$")
_SILENCE_RE = re.compile(r"^\[silence\]$")
_NOISE_WORD_RE = re.compile(r"^\[([a-z_]+)-([^\[\]/]+)\]$")
_BARE_END_RE = re.compile(r"^[a-z']+-$")
_BARE_START_RE = re.compile(r"^-[a-z']+$")
_COINAGE_RE = re.compile(r"^\{(.+)\}$")
_ASIDE_RE = re.compile(r"^<[be]_aside>$")
_VARIANT_RE = re.compile(r"^(.+?)_(\d+)$")


@dataclass(frozen=True)
class Tok:
    raw: str
    kind: str
    spoken: str = ""     # orthography of what was actually produced
    intended: str = ""   # full intended word ("" when unknown)
    cut: str = ""        # PARTIAL: "end", "start" or "both"; MISPRON: "end" if cut off
    noise: str = ""      # NOISE / NOISE_WORD: noise type
    lex_key: str = ""    # raw token, key into the pronunciation lexicon

    @property
    def is_speech(self) -> bool:
        return self.kind not in NON_SPEECH_KINDS


def _outside_brackets(s: str) -> str:
    """Characters outside any [...] group (i.e. what was pronounced), dashes removed."""
    out, depth = [], 0
    for ch in s:
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth = max(0, depth - 1)
        elif depth == 0:
            out.append(ch)
    return "".join(out).strip("-")


def _all_letters(s: str) -> str:
    """Whole orthography: bracket content included, brackets and edge dashes removed."""
    return s.replace("[", "").replace("]", "").strip("-")


def _matching_close(s: str, start: int) -> int:
    """Index of the ']' matching the '[' at ``start`` or -1."""
    depth = 0
    for i in range(start, len(s)):
        if s[i] == "[":
            depth += 1
        elif s[i] == "]":
            depth -= 1
            if depth == 0:
                return i
    return -1


def _top_level_slash(inner: str) -> int:
    depth = 0
    for i, ch in enumerate(inner):
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
        elif ch == "/" and depth == 0:
            return i
    return -1


def parse_token(raw: str) -> Tok:
    """Classify one whitespace-delimited transcript token."""
    raw = raw.strip()
    t = raw.lower()
    if not t:
        raise ValueError("empty token")

    if _SILENCE_RE.match(t):
        return Tok(raw, SILENCE, noise="silence", lex_key=raw)
    m = _NOISE_RE.match(t)
    if m:
        return Tok(raw, NOISE, noise=m.group(1), lex_key=raw)
    if _ASIDE_RE.match(t):
        return Tok(raw, ASIDE, lex_key=raw)
    if t in ("((", "))"):
        return Tok(raw, UNCLEAR, lex_key=raw)

    # [spoken/intended], brackets may nest and a '-' may follow the closing bracket
    if t.startswith("["):
        close = _matching_close(t, 0)
        if close > 0:
            inner = t[1:close]
            slash = _top_level_slash(inner)
            if slash >= 0 and close >= len(t) - 2:
                tail = t[close + 1:]
                if tail in ("", "-"):
                    spoken_raw, intended_raw = inner[:slash], inner[slash + 1:]
                    return Tok(
                        raw, MISPRON,
                        spoken=_outside_brackets(spoken_raw),
                        intended=_all_letters(intended_raw),
                        cut="end" if (tail == "-" or spoken_raw.endswith("-")) else "",
                        lex_key=raw,
                    )

    m = _NOISE_WORD_RE.match(t)
    if m:
        noise, word = m.group(1), m.group(2)
        return Tok(raw, NOISE_WORD, spoken=word, intended=word, noise=noise, lex_key=raw)

    # partial words with brackets: an[y]-  -[th]at  -[a]b[out]-  i['m]-
    if "[" in t and (t.startswith("-") or t.endswith("-")):
        lead, trail = t.startswith("-"), t.endswith("-")
        cut = "both" if (lead and trail) else ("start" if lead else "end")
        return Tok(
            raw, PARTIAL,
            spoken=_outside_brackets(t), intended=_all_letters(t), cut=cut, lex_key=raw,
        )

    if _BARE_END_RE.match(t):
        return Tok(raw, PARTIAL, spoken=t[:-1], intended="", cut="end", lex_key=raw)
    if _BARE_START_RE.match(t):
        return Tok(raw, PARTIAL, spoken=t[1:], intended="", cut="start", lex_key=raw)

    m = _COINAGE_RE.match(t)
    if m:
        w = m.group(1)
        return Tok(raw, COINAGE, spoken=w, intended=w, lex_key=raw)

    base = t
    m = _VARIANT_RE.match(t)
    if m:
        base = m.group(1)
    return Tok(raw, WORD, spoken=base, intended=base, lex_key=raw)


def parse_utterance(text: str) -> list[Tok]:
    """Parse a whole utterance string into tokens."""
    return [parse_token(w) for w in text.split()]
