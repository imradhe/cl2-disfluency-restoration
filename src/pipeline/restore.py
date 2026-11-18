"""Turn BERT's per-gap predictions into restored tokens and a restored transcript."""
from __future__ import annotations

from dataclasses import dataclass

from model.gaps import COPY_L, COPY_R

from .asr import Word
from .extract import GapCand

UNKNOWN = "<unk>"        # a unit was detected (e.g. a partial word) but its identity cannot be restored


@dataclass
class Token:
    text: str
    source: str               # "asr" | "restored"
    start: float
    end: float
    label: str = ""           # disfluency type (restored tokens)
    rest: str = ""            # restoration class the model predicted (restored tokens)
    gap: int = -1

    def to_json(self) -> dict:
        return {"text": self.text, "source": self.source, "start": round(self.start, 3), "end": round(self.end, 3),
                "label": self.label, "rest": self.rest, "gap": self.gap}


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


def restore(words: list[Word], gaps: list[GapCand], preds: list[list[dict]], ctx: int = 16,
            skip_types: frozenset[str] = frozenset()) -> list[Token]:
    """Merge the ASR words with the units predicted for every gap.

    ``preds[g]`` are the predicted units of ``gaps[g]`` (``{s, e, type, restoration}``, phone-relative).
    Units whose type is in ``skip_types`` are detected but not inserted into the transcript.
    """
    by_slot: dict[int, list[Token]] = {}
    texts = [w.text for w in words]
    for g, (cand, units) in enumerate(zip(gaps, preds)):
        left, right = texts[max(0, cand.slot - ctx):cand.slot], texts[cand.slot:cand.slot + ctx]
        for u in sorted(units, key=lambda u: u["s"]):
            if u["type"] in skip_types:
                continue
            surface = render_restoration(u["restoration"], left, right)
            if surface is None:
                continue
            t0, t1 = cand.times[u["s"]][0], cand.times[min(u["e"], len(cand.times)) - 1][1]
            by_slot.setdefault(cand.slot, []).append(
                Token(surface, "restored", t0, t1, label=u["type"], rest=u["restoration"], gap=g))
    out: list[Token] = []
    for k, w in enumerate(words):
        out += by_slot.get(k, [])
        out.append(Token(w.text, "asr", w.start, w.end))
    out += by_slot.get(len(words), [])
    return out


def to_text(tokens: list[Token], annotate: bool = False) -> str:
    """Space-joined transcript; ``annotate`` wraps restored tokens as ``[LABEL:text]``."""
    if not annotate:
        return " ".join(t.text for t in tokens)
    return " ".join(f"[{t.label}:{t.text}]" if t.source == "restored" else t.text for t in tokens)
