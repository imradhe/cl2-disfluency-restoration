"""Turn the sequence model's per-slot predictions into restored tokens and a restored transcript."""
from __future__ import annotations

from dataclasses import dataclass

from model.gaps import COPY_L, COPY_R

from .windows import Window

UNKNOWN = "<unk>"        # a unit was detected (e.g. a partial word) but its identity cannot be restored


@dataclass
class Token:
    text: str
    source: str               # "asr" | "restored"
    start: float
    end: float
    label: str = ""           # disfluency type (restored tokens)
    rest: str = ""            # restoration class the model predicted (restored tokens)
    window: int = -1

    def to_json(self) -> dict:
        return {"text": self.text, "source": self.source, "start": round(self.start, 3), "end": round(self.end, 3),
                "label": self.label, "rest": self.rest, "window": self.window}


def render_restoration(rest: str, left: list[str], right: list[str]) -> str | None:
    """Surface form of a restoration class, or ``None`` when nothing is to be inserted."""
    if rest.startswith(("FIL:", "W:")):
        return rest.split(":", 1)[1]
    if rest.startswith("COPY_R"):
        k = int(rest[6:])
        return right[k - 1] if 1 <= k <= min(COPY_R, len(right)) else UNKNOWN
    if rest.startswith("COPY_L"):
        k = int(rest[6:])
        return left[-k] if 1 <= k <= min(COPY_L, len(left)) else UNKNOWN
    if rest == "UNK":
        return UNKNOWN
    return None             # DROP (noise), SELF, <pad>


def restore_windows(windows: list[Window], preds: list[list[list[str]]],
                    skip_types: frozenset[str] = frozenset()) -> list[Token]:
    """Merge every window's ASR words with the units predicted for its slots.

    ``preds[w][s]`` are the ``"LABEL|TARGET"`` units of slot ``s`` of window ``w`` (slot ``s`` = in front of word
    ``s``; the last slot follows the last word). Units whose type is in ``skip_types`` are detected but not
    inserted. The model gives no times, so the units of a slot share the pause between its two neighbouring words.
    """
    out: list[Token] = []
    for wi, (win, slots) in enumerate(zip(windows, preds)):
        texts = [w.text for w in win.words]
        n = len(texts)
        for s in range(n + 1):
            units = [u.split("|", 1) for u in slots[s] if u.split("|", 1)[0] not in skip_types] if s < len(slots) else []
            t0 = win.words[s - 1].end if s > 0 else win.t0
            t1 = win.words[s].start if s < n else win.t1
            t1 = max(t0, t1)
            rendered = [(lab, tgt, render_restoration(tgt, texts[:s], texts[s:])) for lab, tgt in units]
            rendered = [r for r in rendered if r[2] is not None]
            for k, (lab, tgt, surface) in enumerate(rendered):
                a = t0 + (t1 - t0) * k / len(rendered)
                b = t0 + (t1 - t0) * (k + 1) / len(rendered)
                out.append(Token(surface, "restored", a, b, label=lab, rest=tgt, window=wi))
            if s < n:
                w = win.words[s]
                out.append(Token(w.text, "asr", w.start, w.end, window=wi))
    return out


def to_text(tokens: list[Token], annotate: bool = False) -> str:
    """Space-joined transcript; ``annotate`` wraps restored tokens as ``[LABEL:text]``."""
    if not annotate:
        return " ".join(t.text for t in tokens)
    return " ".join(f"[{t.label}:{t.text}]" if t.source == "restored" else t.text for t in tokens)
