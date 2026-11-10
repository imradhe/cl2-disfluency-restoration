"""Parser for BhashaCheck ``.rsml`` annotation files (NPTEL lecture transcripts).

File layout::

    # BhashaCheck config            <- header blocks: [settings] [speakers] [tags] ...
    ...
    1
    00:00:15,228 --> 00:00:18,276
    primary=1|verified=1|flagged=0|note=...
    We are discussing about !!ml[artificial neural network](artificial neural network)

Inline markup (from the header's [tags] section):

* hesitations     ``@uhh @umm @hmm @ugh @huh @tsk @uh-huh @ehh``
* disfluency spans ``@X-start ... @X-end`` for filler, repetition, broken-word, repair,
  false-start, prolongation. ``repair`` / ``false-start`` contain ``reparandum - repair``
  (a standalone ``-``); only the part before it is the discarded material.
* isolated tags   ``@pause @short-pause @long-pause @breathe @stutter-block ...`` (dropped)
* prosody / paralinguistic spans (pitch, emphasis, speech rate, laughing ...) are transparent
* entities        ``!!ml[display](spoken)``, ``!!mathex[spoken]($latex$)`` -> the bracket text
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

HESITATION_WORD = {
    "@uhh": "uh", "@umm": "um", "@hmm": "hm", "@ugh": "ugh", "@huh": "huh",
    "@tsk": "tsk", "@ehh": "eh", "@ohh": "oh", "@uh-huh": "uh-huh",
}
SPAN_LABEL = {
    "filler": "FILLER_DM", "repetition": "REPETITION", "broken-word": "PARTIAL_WORD",
    "repair": "REPAIR", "false-start": "FALSE_START", "prolongation": "PROLONGATION",
}
HAS_SEPARATOR = {"repair", "false-start"}

_TIME = re.compile(r"(\d+):(\d+):(\d+)[,.](\d+)")
_META = re.compile(r"^primary=(\d)\|verified=(\d)\|flagged=(\d)\|note=(.*)$")
_ENTITY = re.compile(r"(?:!!|!|#)[A-Za-z]+\[([^\]]*)\]\(((?:[^()]|\([^()]*\))*)\)")
_PIECE_TAG = re.compile(r"^(@[a-z][a-z0-9\-]*)([,.?!;:]*)$")
_STRIP = ".,?!;:\"“”()"


def _sec(t: str) -> float:
    m = _TIME.match(t.strip())
    if not m:
        raise ValueError(t)
    h, mi, s, ms = m.groups()
    return int(h) * 3600 + int(mi) * 60 + int(s) + int(ms.ljust(3, "0")[:3]) / 1000


@dataclass
class Seg:
    idx: int
    start: float
    end: float
    primary: int
    verified: int
    flagged: int
    note: str
    text: str


@dataclass
class Rsml:
    path: str
    header: dict[str, dict[str, str]] = field(default_factory=dict)
    segments: list[Seg] = field(default_factory=list)

    @property
    def n_verified(self) -> int:
        return sum(s.verified == 1 for s in self.segments)

    @property
    def fully_verified(self) -> bool:
        return bool(self.segments) and self.n_verified == len(self.segments)

    @property
    def script_ratio(self) -> float:
        """Fraction of letters that are non-Latin (Telugu etc.): ~0 for an English lecture."""
        letters = [c for s in self.segments for c in s.text if c.isalpha()]
        return sum(not c.isascii() for c in letters) / max(1, len(letters))

    @property
    def is_english(self) -> bool:
        return self.script_ratio < 0.01

    @property
    def duration(self) -> float:
        return sum(s.end - s.start for s in self.segments)


def _parse_header(block: str, out: dict[str, dict[str, str]]) -> None:
    section = ""
    for line in block.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("[") and line.endswith("]"):
            section = line[1:-1]
            out.setdefault(section, {})
        elif "=" in line and section:
            k, v = line.split("=", 1)
            out[section][k.strip()] = v.strip()


def parse_rsml(path: str | Path) -> Rsml:
    text = Path(path).read_text(encoding="utf-8")
    r = Rsml(path=str(path))
    for block in re.split(r"\n\s*\n", text.strip()):
        if not block.strip():
            continue
        lines = block.split("\n")
        if len(lines) >= 3 and lines[0].strip().isdigit() and "-->" in lines[1]:
            m = _META.match(lines[2].strip())
            if m:
                a, b = lines[1].split("-->")
                r.segments.append(Seg(
                    int(lines[0]), _sec(a), _sec(b), int(m[1]), int(m[2]), int(m[3]), m[4],
                    " ".join(x.strip() for x in lines[3:]).strip(),
                ))
                continue
        if block.lstrip().startswith(("#", "[")):
            _parse_header(block, r.header)
    return r


def _clean_entities(text: str) -> str:
    return _ENTITY.sub(lambda m: m.group(1), text)


def tag_text(text: str) -> tuple[list[dict], dict[str, int]]:
    """Tokenise one segment into labelled words.

    Returns (tokens, dropped) where each token is
    ``{"spoken", "label", "role", "kind"}`` and ``dropped`` counts discarded isolated tags.
    """
    return _tag_pieces([(p, None) for p in _clean_entities(text).split()])


def _tag_pieces(pieces: list[tuple[str, dict | None]]) -> tuple[list[dict], dict[str, int]]:
    """State machine over already-split pieces; ``info`` (e.g. aligned times) rides along."""
    toks: list[dict] = []
    dropped: dict[str, int] = {}
    stack: list[dict] = []        # open disfluency spans: {"name", "reparandum": bool}

    def top_label() -> tuple[str, str]:
        for sp in reversed(stack):
            name = sp["name"]
            if name in HAS_SEPARATOR:
                if sp["reparandum"]:
                    return SPAN_LABEL[name], "reparandum"
                continue                     # repair part: keep looking outwards
            return SPAN_LABEL[name], {"filler": "filler", "repetition": "reparandum",
                                      "broken-word": "fragment", "prolongation": "prolonged"}[name]
        return "O", ""

    for piece, info in pieces:
        m = _PIECE_TAG.match(piece)
        if m:
            tag = m.group(1)
            if tag in HESITATION_WORD:
                w = HESITATION_WORD[tag]
                toks.append({"spoken": w, "kind": "WORD", "role": "", "info": info,
                             "label": "FILLER_BACKCHANNEL" if w == "uh-huh" else "FILLER_HESITATION"})
                continue
            mm = re.match(r"^@(.+)-(start|end)$", tag)
            if mm and mm.group(1) in SPAN_LABEL:
                name, edge = mm.groups()
                if edge == "start":
                    stack.append({"name": name, "reparandum": name in HAS_SEPARATOR})
                else:
                    for k in range(len(stack) - 1, -1, -1):
                        if stack[k]["name"] == name:
                            del stack[k:]
                            break
                continue
            if mm:                            # prosody / paralinguistic span: transparent
                continue
            dropped[tag] = dropped.get(tag, 0) + 1      # @pause, @breathe, @stutter-block ...
            continue
        if piece == "-" and stack:
            for sp in reversed(stack):
                if sp["name"] in HAS_SEPARATOR and sp["reparandum"]:
                    sp["reparandum"] = False
                    break
            continue
        word = piece.strip(_STRIP).lower()
        if not word or word == "-":
            continue
        label, role = top_label()
        kind = "PARTIAL" if label == "PARTIAL_WORD" else "WORD"
        toks.append({"spoken": word, "kind": kind, "role": role, "label": label, "info": info})
    return toks, dropped


# ------------------------------------------------------------------ aligned files

@dataclass
class AlignedSeg:
    seg: Seg
    entries: list[dict]       # {start, end, kind, text, conf, pause_after, voiced}


def _num(s: str) -> float | None:
    s = s.strip()
    return float(s) if s else None


def parse_aligned(path: str | Path) -> list[AlignedSeg]:
    """Parse ``*.aligned.rsml``: each segment is followed by tab-indented word entries
    (time line, ``confidence=..|kind=..`` line, token text) from a forced alignment."""
    out: list[AlignedSeg] = []
    text = Path(path).read_text(encoding="utf-8")
    for block in re.split(r"\n\s*\n", text.strip()):
        lines = block.split("\n")
        if not (len(lines) >= 3 and lines[0].strip().isdigit() and "-->" in lines[1]):
            continue
        m = _META.match(lines[2].strip())
        if not m:
            continue
        body = [x for x in lines[3:] if not x.startswith("\t")]
        ents = [x.strip("\t") for x in lines[3:] if x.startswith("\t")]
        a, b = lines[1].split("-->")
        seg = Seg(int(lines[0]), _sec(a), _sec(b), int(m[1]), int(m[2]), int(m[3]), m[4], " ".join(body).strip())
        entries = []
        for k in range(0, len(ents) - 2, 3):
            t0, t1 = ents[k].split("-->")
            meta = dict(kv.split("=", 1) for kv in ents[k + 1].split("|") if "=" in kv)
            entries.append({
                "start": _sec(t0), "end": _sec(t1), "kind": meta.get("kind", ""), "text": ents[k + 2].strip(),
                "conf": _num(meta.get("confidence", "")), "pause_after": _num(meta.get("pause_after_milliseconds", "")),
                "voiced": _num(meta.get("voiced_fraction", "")),
            })
        out.append(AlignedSeg(seg, entries))
    return out


def aligned_to_side(path: str | Path, sid: str | None = None) -> dict:
    """Per-side record with *real* forced-alignment times for every gold token."""
    sid = sid or Path(path).stem.replace(".aligned", "")
    toks_out: list[dict] = []
    dropped: dict[str, int] = {}
    prev_end = None
    for aseg in parse_aligned(path):
        pieces = [(_clean_entities(e["text"]).strip(), e) for e in aseg.entries if e["text"].strip()]
        toks, d = _tag_pieces(pieces)
        for k, v in d.items():
            dropped[k] = dropped.get(k, 0) + v
        for j, t in enumerate(toks):
            e = t["info"]
            first = j == 0
            toks_out.append({
                "i": len(toks_out), "raw": t["spoken"], "kind": t["kind"], "spoken": t["spoken"],
                "intended": "" if t["kind"] == "PARTIAL" else t["spoken"],
                "cut": "end" if t["kind"] == "PARTIAL" else "", "label": t["label"], "role": t["role"],
                "prolonged": t["label"] == "PROLONGATION", "start": round(e["start"], 3), "end": round(e["end"], 3),
                "pause_before": 0.0 if first or prev_end is None else round(max(0.0, e["start"] - prev_end), 3),
                "utt": f"{sid}-{aseg.seg.idx:04d}", "utt_first": first, "utt_last": j == len(toks) - 1,
                "phones": None, "conf": e["conf"], "pause_after": e["pause_after"], "voiced": e["voiced"],
            })
            prev_end = e["end"]
    return {"sid": sid, "conv": "0", "side": "A", "tokens": toks_out, "dropped_tags": dropped}


def to_side(rsml: Rsml, sid: str | None = None) -> dict:
    """Convert a parsed lecture into the same per-side record the Switchboard pipeline uses.

    Token times are *approximate* (segment time split evenly over its tokens) because no
    word-level forced alignment is available yet; phones are filled in by the caller.
    """
    sid = sid or Path(rsml.path).stem
    out: list[dict] = []
    dropped: dict[str, int] = {}
    for seg in rsml.segments:
        toks, d = tag_text(seg.text)
        for k, v in d.items():
            dropped[k] = dropped.get(k, 0) + v
        n = max(1, len(toks))
        step = (seg.end - seg.start) / n
        for j, t in enumerate(toks):
            out.append({
                "i": len(out), "raw": t["spoken"], "kind": t["kind"], "spoken": t["spoken"],
                "intended": "" if t["kind"] == "PARTIAL" else t["spoken"], "cut": "end" if t["kind"] == "PARTIAL" else "",
                "label": t["label"], "role": t["role"], "prolonged": t["label"] == "PROLONGATION",
                "start": round(seg.start + j * step, 3), "end": round(seg.start + (j + 1) * step, 3),
                "pause_before": 0.0, "utt": f"{sid}-{seg.idx:04d}", "utt_first": j == 0,
                "utt_last": j == n - 1, "phones": None,
            })
    return {"sid": sid, "conv": "0", "side": "A", "tokens": out, "dropped_tags": dropped}
