"""From ASR words + recognized phones to gap examples for the BERT model."""
from __future__ import annotations

from dataclasses import dataclass, field

from .align import AlignParams, align_phones, fit_cost, insertion_slots
from .asr import Word
from .phonerec import RecPhone


@dataclass
class GapParams:
    min_phones: int = 1          # shorter insertion runs are ignored
    max_phones: int = 30         # longer ones are more likely dropped speech than a disfluency
    min_prob: float = 0.0        # mean recognizer posterior needed to keep a gap
    min_dur: float = 0.15        # estimated speech duration (s) needed to keep a gap; see phone_durations.
    #                              Gaps under ~0.1 s are mostly pronunciation / boundary artefacts (13% hit a gold
    #                              disfluency on english_001, vs 35-60% above 0.1 s).
    phone_dur_cap: float = 0.15  # see phone_durations
    choose_variants: bool = True # pick each word's pronunciation variant from what was heard
    align: AlignParams = field(default_factory=AlignParams)


@dataclass
class GapCand:
    slot: int                    # global word index the gap sits before (n_words = after the last word)
    phones: list[str]
    start: float
    end: float
    dur: float                   # estimated speech duration of the gap (see phone_durations)
    probs: list[float]
    times: list[tuple[float, float]]    # (start, end) of every phone

    def to_json(self) -> dict:
        return {"slot": self.slot, "phones": self.phones, "start": round(self.start, 3), "end": round(self.end, 3),
                "dur": round(self.dur, 3), "mean_p": round(sum(self.probs) / len(self.probs), 3)}


def phone_durations(rec: list[RecPhone], cap: float = 0.15, floor: float = 0.04) -> list[float]:
    """Per-phone speech duration estimate.

    CTC emits phones as 1-3 frame spikes, so ``end - start`` badly underestimates how long a phone
    was spoken. The time until the next recognized phone, clipped to ``[floor, cap]``, is closer to
    the segment durations the model was trained with (Switchboard word-level alignments).
    """
    out = []
    for k, r in enumerate(rec):
        d = rec[k + 1].start - r.start if k + 1 < len(rec) else r.end - r.start
        out.append(min(cap, max(floor, d)))
    return out


def choose_pronunciation(word: Word, variants: list[list[str]], rec: list[RecPhone], pad: float,
                         params: AlignParams) -> list[str]:
    """The lexicon pronunciation best supported by the phones heard in the word's span (+/- ``pad`` s)."""
    if len(variants) == 1:
        return variants[0]
    heard = [r.ph for r in rec if word.start - pad <= (r.start + r.end) / 2 <= word.end + pad]
    costs = [fit_cost(v, heard, params) for v in variants]
    return variants[costs.index(min(costs))]          # ties keep the lexicon's first (canonical) entry


def expected_phones(words: list[Word], g2p, rec: list[RecPhone] | None = None, pad: float = 0.25,
                    params: AlignParams = AlignParams()) -> tuple[list[str], list[int], list[float]]:
    """Concatenated G2P phones of the words, the index (within ``words``) of each phone's word, and its
    time (linear interpolation inside the word's Whisper span). With ``rec`` the pronunciation variant
    of each word is chosen by how well the recognized phones support it (else the first variant)."""
    ph: list[str] = []
    wi: list[int] = []
    ts: list[float] = []
    for k, w in enumerate(words):
        variants = g2p.variants(w.text)
        p = choose_pronunciation(w, variants, rec, pad, params) if rec is not None else variants[0]
        for q, x in enumerate(p):
            ph.append(x)
            wi.append(k)
            ts.append(w.start + (q + 0.5) / len(p) * (w.end - w.start))
    return ph, wi, ts


def find_gaps(words: list[Word], rec: list[RecPhone], g2p, base_word: int = 0,
              params: GapParams = GapParams()) -> tuple[list[GapCand], dict]:
    """Gaps in one chunk. ``words`` are the chunk's ASR words, ``rec`` its recognized phones, and
    ``base_word`` the global index of ``words[0]`` (so ``GapCand.slot`` is global)."""
    exp_ph, exp_wi, exp_t = expected_phones(words, g2p, rec if params.choose_variants else None, params=params.align)
    rec_t = [(r.start + r.end) / 2 for r in rec]
    ops = align_phones(exp_ph, exp_wi, exp_t, [r.ph for r in rec], rec_t, params.align)
    slots, n_inside = insertion_slots(ops, exp_wi, len(words))
    durs = phone_durations(rec, params.phone_dur_cap)
    stats = {"exp_phones": len(exp_ph), "rec_phones": len(rec), "inserted": sum(len(v) for v in slots.values()) + n_inside,
             "inside_word": n_inside, "too_short": 0, "too_long": 0, "low_prob": 0, "too_brief": 0, "kept": 0,
             "matched": sum(op[0] == "M" for op in ops), "substituted": sum(op[0] == "S" for op in ops),
             "deleted": sum(op[0] == "D" for op in ops)}
    gaps: list[GapCand] = []
    for slot in sorted(slots):
        idx = sorted(slots[slot])
        if len(idx) < params.min_phones:
            stats["too_short"] += 1
            continue
        if len(idx) > params.max_phones:
            stats["too_long"] += 1
            continue
        probs = [rec[j].p for j in idx]
        if sum(probs) / len(probs) < params.min_prob:
            stats["low_prob"] += 1
            continue
        dur = sum(durs[j] for j in idx)
        if dur < params.min_dur:
            stats["too_brief"] += 1
            continue
        gaps.append(GapCand(
            slot=base_word + slot, phones=[rec[j].ph for j in idx], start=rec[idx[0]].start, end=rec[idx[-1]].end,
            dur=dur, probs=probs, times=[(rec[j].start, rec[j].end) for j in idx],
        ))
        stats["kept"] += 1
    return gaps, stats


def build_examples(gaps: list[GapCand], words: list[str], sid: str, ctx: int = 16) -> list[dict]:
    """Gap examples in the training schema (no gold ``units``): ``words`` is the whole ASR word list."""
    out = []
    for g, cand in enumerate(gaps):
        out.append({
            "id": f"{sid}:g{g}", "sid": sid, "kind": "gap",
            "left": words[max(0, cand.slot - ctx):cand.slot], "right": words[cand.slot:cand.slot + ctx],
            "phones": cand.phones, "units": [], "dur": round(cand.dur, 3),
        })
    return out
