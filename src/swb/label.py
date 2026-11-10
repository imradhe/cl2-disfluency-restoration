"""Rule-based disfluency labelling of one Switchboard speaker side.

Gold (explicit in the ISIP markup): NOISE, PARTIAL_WORD, MISPRONUNCIATION,
hyphenated backchannels, uh/um/ah/hm/huh.
Silver (derived here): FILLER_BACKCHANNEL (yeah/right/... as short turns),
FILLER_DM, REPETITION, REPAIR, FALSE_START, PROLONGATION.

Labelling is *token level*; each token gets exactly one primary label from
``configs/labels.yaml::tags`` plus a ``role`` for tokens belonging to a
reparandum / interregnum.
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable

from .corpus import Side, WTok, pause_before
from .markers import (
    ASIDE, COINAGE, MISPRON, NOISE, NOISE_WORD, PARTIAL, SILENCE, UNCLEAR, WORD, Tok,
)

O = "O"
FILLER_HESITATION = "FILLER_HESITATION"
FILLER_BACKCHANNEL = "FILLER_BACKCHANNEL"
FILLER_DM = "FILLER_DM"
REPETITION = "REPETITION"
REPAIR = "REPAIR"
FALSE_START = "FALSE_START"
PARTIAL_WORD = "PARTIAL_WORD"
PROLONGATION = "PROLONGATION"
MISPRONUNCIATION = "MISPRONUNCIATION"
NOISE_TAG = "NOISE"

# labels whose tokens stay in the fluent skeleton (events *inside* a word)
IN_WORD_LABELS = {O, PROLONGATION, MISPRONUNCIATION}

FUNCTION_WORDS = set(
    "the a an and but so to of in on at for it that is was i you we they he she be have do "
    "just then or as if because with".split()
)
# words that make "you know"/"i mean" a literal clause rather than a discourse marker
_LITERAL_PREV = {"do", "did", "does", "don't", "didn't", "if", "whether", "what", "that", "how",
                 "as", "can", "could", "would", "should", "who", "i", "we", "they", "to"}
_LITERAL_NEXT = {"what", "how", "where", "who", "when", "why", "if", "whether", "about", "the",
                 "a", "an", "that", "of", "any"}
_INITIAL_ACK = {"yeah", "yep", "yes", "right", "sure", "okay", "well", "so", "actually"}
# "a kind of metal" is literal; "it was kind of cold" is a hedge/discourse marker
_LITERAL_PREV_KIND = {"a", "an", "the", "this", "that", "what", "any", "some", "every", "one",
                      "different", "same", "another", "each", "no"}
# words that join clauses: the abandoned fragment starts after them, never includes them
_CLAUSE_LINK = {"and", "but", "so", "because", "then", "or", "that", "which", "if", "when", "while"}
# a clause-initial word right after an abandoned fragment signals a restart
_RESTART_START = {"i", "we", "you", "they", "he", "she", "it", "the", "a", "that", "there", "so",
                  "but", "and", "well", "what", "how", "why", "do", "did", "is", "are", "was",
                  "were", "my", "our", "his", "her", "this", "if", "when"}


@dataclass
class LTok:
    i: int
    wt: WTok
    pause_before: float = 0.0
    pause_after: float = 0.0
    utt_first: bool = False       # first speech token of its utterance
    utt_last: bool = False
    utt_len: int = 0              # speech tokens in the utterance (noise excluded)
    label: str = O
    role: str = ""                # reparandum | interregnum | repair
    prolonged: bool = False       # unusually long (any token)

    @property
    def tok(self) -> Tok:
        return self.wt.tok

    @property
    def text(self) -> str:
        return self.tok.spoken or self.tok.raw.lower()


# ---------------------------------------------------------------- build

def build_ltoks(side: Side) -> list[LTok]:
    """Speech + noise tokens of a side with pause and utterance-position features."""
    words = side.words
    keep = [k for k, w in enumerate(words) if w.tok.kind not in (SILENCE, ASIDE, UNCLEAR)]
    toks: list[LTok] = []
    for i, k in enumerate(keep):
        toks.append(LTok(i=i, wt=words[k], pause_before=pause_before(words, k)))
    for a, b in zip(toks, toks[1:]):
        a.pause_after = b.pause_before
    # utterance position features (noise tokens do not count as speech)
    by_utt: dict[str, list[LTok]] = defaultdict(list)
    for t in toks:
        by_utt[t.wt.utt_id].append(t)
    for utt in by_utt.values():
        speech = [t for t in utt if t.tok.kind != NOISE]
        for t in utt:
            t.utt_len = len(speech)
        if speech:
            speech[0].utt_first = True
            speech[-1].utt_last = True
    return toks


# ---------------------------------------------------------------- lexical labels

def _label_lexical(toks: list[LTok], cfg: dict) -> None:
    hes = set(cfg["hesitation_gold"]) | set(cfg["hesitation_silver"])
    bc_gold = set(cfg["backchannel_gold"])
    bc_words = set(cfg["backchannel_words"])
    bc_max = cfg["backchannel_max_tokens"]
    thr = cfg["dm_pause_threshold_s"]
    dm_single = set(cfg["discourse_markers_single"])

    for t in toks:
        k, w = t.tok.kind, t.text
        if k == NOISE:
            t.label = NOISE_TAG
        elif k == PARTIAL:
            t.label = PARTIAL_WORD
        elif k == MISPRON:
            t.label = PARTIAL_WORD if t.tok.cut else MISPRONUNCIATION
        elif k == WORD:
            if w in bc_gold:
                t.label = FILLER_BACKCHANNEL
            elif w in hes:
                t.label = FILLER_HESITATION
            elif w in bc_words and t.utt_len <= bc_max:
                t.label = FILLER_BACKCHANNEL

    # multi-word discourse markers ("you know", "i mean", ...)
    for pat in cfg["discourse_markers_multi"]:
        n = len(pat)
        for s in range(len(toks) - n + 1):
            win = toks[s:s + n]
            if all(x.label == O and x.tok.kind == WORD and x.text == p for x, p in zip(win, pat)):
                prev = toks[s - 1].text if s > 0 else ""
                nxt = toks[s + n].text if s + n < len(toks) else ""
                if prev in _LITERAL_PREV or nxt in _LITERAL_NEXT:
                    continue
                if pat[1] == "of" and prev in _LITERAL_PREV_KIND:
                    continue
                for x in win:
                    x.label = FILLER_DM

    # single-word markers / acknowledgements
    fillers = {FILLER_HESITATION, FILLER_BACKCHANNEL, FILLER_DM}
    for idx, t in enumerate(toks):
        if t.label != O or t.tok.kind != WORD:
            continue
        w = t.text
        prev_f = idx > 0 and toks[idx - 1].label in fillers
        next_f = idx + 1 < len(toks) and toks[idx + 1].label in fillers
        initial = t.utt_first or (idx > 0 and toks[idx - 1].label in fillers and toks[idx - 1].utt_first)
        if w in _INITIAL_ACK and initial and w not in {"so", "well", "actually"}:
            t.label = FILLER_DM
        elif w in dm_single:
            isolated = t.pause_before >= thr or t.pause_after >= thr
            if (initial and isolated) or prev_f or next_f or (w == "well" and initial):
                t.label = FILLER_DM


# ---------------------------------------------------------------- structural labels

def _eq(a: LTok, b: LTok) -> bool:
    """Token equality tolerant to partial words (a fragment matches the word it starts)."""
    ta, tb = a.tok, b.tok
    sa, sb = a.text, b.text
    if ta.kind == PARTIAL and tb.kind == PARTIAL:
        return sa == sb
    for frag, full in ((ta, tb), (tb, ta)):
        if frag.kind == PARTIAL and frag.spoken:
            word = full.spoken
            return word.endswith(frag.spoken) if frag.cut == "start" else word.startswith(frag.spoken)
    return sa == sb


def _content(toks: list[LTok]) -> list[LTok]:
    """Tokens that take part in repetition/repair matching (fillers, noise excluded)."""
    return [t for t in toks if t.label in (O, PARTIAL_WORD, MISPRONUNCIATION)]


def _label_repetitions(toks: list[LTok], cfg: dict) -> None:
    max_n = cfg["repetition"]["max_ngram"]
    stop = {tuple(p) for p in cfg["repetition"].get("intentional", [])}
    c = _content(toks)
    i = 0
    while i < len(c):
        done = False
        for n in range(min(max_n, (len(c) - i) // 2), 0, -1):
            a, b = c[i:i + n], c[i + n:i + 2 * n]
            if len(b) < n or not all(_eq(x, y) for x, y in zip(a, b)):
                continue
            if tuple(x.text for x in a) * 2 in stop:     # "bye bye": deliberate reduplication
                continue
            if c[i + n].pause_before > 1.0:      # long silence: probably separate statements
                continue
            for x in a:
                if x.label == O:
                    x.label, x.role = REPETITION, "reparandum"
            i += n
            done = True
            break
        if not done:
            i += 1


def _label_repairs(toks: list[LTok], cfg: dict) -> None:
    rc = cfg["repair"]
    min_ov = rc["min_copy_overlap"]
    c = _content(toks)
    i = 0
    while i < len(c):
        matched = False
        for n in range(min(6, (len(c) - i) // 2), 1, -1):
            a, b = c[i:i + n], c[i + n:i + 2 * n]
            if len(b) < n or c[i + n].pause_before > 1.0:
                continue
            if any(x.label in (REPETITION, REPAIR) for x in a):
                continue
            hits = sum(_eq(x, y) for x, y in zip(a, b))
            if not _eq(a[0], b[0]) or hits >= n:
                continue
            ratio = hits / n
            # precision guard: parallel phrasing ("from X to Y") is common, so a repair needs a
            # disfluency cue at the interruption point, unless the near-copy is long.
            cue = a[-1].label == PARTIAL_WORD or _has_interregnum(toks, a[-1], b[0])
            if (cue and ratio >= min_ov) or (n >= 5 and ratio >= 0.8):
                for x in a:
                    if x.label == O:
                        x.label, x.role = REPAIR, "reparandum"
                for x in b:
                    if x.label == O:
                        x.role = "repair"
                _mark_interregnum(toks, a[-1], b[0])
                i += n
                matched = True
                break
        if not matched:
            i += 1


def _has_interregnum(toks: list[LTok], last_reparandum: LTok, first_repair: LTok) -> bool:
    return any(
        t.label in (FILLER_HESITATION, FILLER_DM)
        for t in toks[last_reparandum.i + 1:first_repair.i]
    )


def _mark_interregnum(toks: list[LTok], last_reparandum: LTok, first_repair: LTok) -> None:
    for t in toks[last_reparandum.i + 1:first_repair.i]:
        if t.label in (FILLER_HESITATION, FILLER_DM, FILLER_BACKCHANNEL):
            t.role = "interregnum"


def _label_false_starts(toks: list[LTok], cfg: dict) -> None:
    """Abandoned fragments: a tail-cut partial word that is not completed or copied."""
    for idx, t in enumerate(toks):
        if t.label != PARTIAL_WORD or t.role or t.tok.cut not in ("end", "both"):
            continue
        nxt = next((u for u in toks[idx + 1:] if u.label in (O, PARTIAL_WORD, REPETITION, REPAIR)), None)
        if nxt is None or _eq(t, nxt):               # "w[ent]- went": word-level restart
            continue
        if nxt.text not in _RESTART_START:           # no sign of a new clause after the cut
            continue
        run: list[LTok] = []
        j = idx - 1
        while j >= 0 and len(run) < 4:
            u = toks[j]
            if u.label in (FILLER_HESITATION, FILLER_DM, FILLER_BACKCHANNEL, NOISE_TAG):
                j -= 1
                continue
            if u.label != O or u.text in _CLAUSE_LINK:   # previous clause / conjunction: stop
                break
            run.append(u)
            if u.utt_first or u.pause_before >= 0.3:     # chunk onset reached
                break
            j -= 1
        if len(run) >= 2:
            for u in run:
                u.label, u.role = FALSE_START, "reparandum"


# ---------------------------------------------------------------- prolongation

class DurStats:
    """Per-word log-duration statistics collected over the corpus."""

    def __init__(self) -> None:
        self._xs: dict[str, list[float]] = defaultdict(list)
        self.mean: dict[str, float] = {}
        self.std: dict[str, float] = {}

    def add(self, toks: Iterable[LTok]) -> None:
        for t in toks:
            if t.tok.kind == WORD and t.wt.dur > 0.02:
                self._xs[t.text].append(math.log(t.wt.dur))

    def finalize(self, min_n: int = 30) -> None:
        for w, xs in self._xs.items():
            if len(xs) >= min_n:
                m = sum(xs) / len(xs)
                v = sum((x - m) ** 2 for x in xs) / max(1, len(xs) - 1)
                self.mean[w], self.std[w] = m, max(math.sqrt(v), 0.15)
        self._xs = defaultdict(list)

    def z(self, word: str, dur: float) -> float | None:
        if word not in self.mean or dur <= 0.02:
            return None
        return (math.log(dur) - self.mean[word]) / self.std[word]


def _label_prolongations(toks: list[LTok], cfg: dict, stats: DurStats) -> None:
    pc = cfg["prolongation"]
    for t in toks:
        if t.tok.kind != WORD:
            continue
        z = stats.z(t.text, t.wt.dur)
        if z is None or z < pc["zscore_threshold"] or t.wt.dur < pc["min_duration_s"]:
            continue
        t.prolonged = True
        if t.label != O:
            continue
        if pc["function_words_only"] and t.text not in FUNCTION_WORDS:
            continue
        if pc["exclude_phrase_final"] and (t.pause_after >= pc["pause_s"] or t.utt_last):
            continue
        t.label = PROLONGATION


# ---------------------------------------------------------------- public API

def label_side(side: Side, cfg: dict, stats: DurStats | None = None) -> list[LTok]:
    toks = build_ltoks(side)
    _label_lexical(toks, cfg)
    _label_repetitions(toks, cfg)
    _label_repairs(toks, cfg)
    _label_false_starts(toks, cfg)
    if stats is not None:
        _label_prolongations(toks, cfg, stats)
    return toks


def collect_stats(sides: Iterable[Side]) -> DurStats:
    stats = DurStats()
    for s in sides:
        stats.add(build_ltoks(s))
    stats.finalize()
    return stats
