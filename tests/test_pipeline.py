"""Pipeline pieces that need no model: VAD chunking, ASR word normalisation, phone alignment,
gap extraction and restoration rendering."""
import pytest

from pipeline.align import align_phones, insertion_slots
from pipeline.asr import Word, norm_word, words_from_segments
from pipeline.audio import chunk_speech
from pipeline.extract import GapParams, build_examples, find_gaps, phone_durations
from pipeline.phonerec import RecPhone, collapse, to_swb
from pipeline.restore import UNKNOWN, render_restoration, restore, to_text


# ------------------------------------------------------------------ audio / asr

def test_chunk_speech_merges_short_pauses_and_respects_max_len():
    spans = [(1.0, 3.0), (3.3, 5.0), (9.0, 10.0), (10.2, 38.0)]
    ch = chunk_speech(spans, total=40.0, max_len=28.0, max_gap=0.6, pad=0.0)
    assert [(c.start, c.end) for c in ch[:2]] == [(1.0, 5.0), (9.0, 10.0)]
    assert all(c.end - c.start <= 28.0 + 1e-6 for c in ch)       # 10.2-38.0 is 27.8 s, cannot join 9-10
    assert [c.idx for c in ch] == list(range(len(ch)))


def test_chunk_speech_pads_without_overlap_and_stays_in_bounds():
    ch = chunk_speech([(0.05, 1.0), (1.4, 2.0)], total=2.1, max_gap=0.1, pad=0.3)
    assert ch[0].start == 0.0 and ch[-1].end == 2.1
    assert ch[0].end <= ch[1].start                             # padded 0.2 each way into a 0.4 s pause


def test_chunk_speech_splits_overlong_span():
    ch = chunk_speech([(0.0, 70.0)], total=70.0, max_len=28.0, pad=0.0)
    assert len(ch) == 3 and all(c.end - c.start <= 28.0 for c in ch)


def test_norm_word_strips_edges_keeps_inner_apostrophe():
    assert norm_word(" Don't,") == "don't"
    assert norm_word('"Hello."') == "hello"
    assert norm_word(" ...") == ""


def test_words_from_segments_offsets_and_drops_empty():
    segs = [{"words": [{"word": " So", "start": 0.0, "end": 0.2, "probability": 0.9},
                       {"word": ",", "start": 0.2, "end": 0.2},
                       {"word": " yes.", "start": 0.3, "end": 0.6}]}]
    ws = words_from_segments(segs, offset=10.0)
    assert [(w.text, w.start, w.end) for w in ws] == [("so", 10.0, 10.2), ("yes", 10.3, 10.6)]


def test_collapse_merges_repeats_and_drops_blank():
    import numpy as np
    vocab = ["<pad>", "a", "b"]
    ids = np.array([0, 1, 1, 0, 1, 2, 2, 0])
    toks = collapse(ids, np.ones(8), vocab, t_off=1.0)
    assert [t["tok"] for t in toks] == ["a", "a", "b"]
    assert toks[0]["start"] == pytest.approx(1.02) and toks[0]["end"] == pytest.approx(1.06)


# ------------------------------------------------------------------ alignment

def run_align(words, rec, times=None):
    """words: list of (phones) per word, each 0.3 s long back to back; rec: list of (phone, time)."""
    exp_ph, exp_wi, exp_t = [], [], []
    for k, ph in enumerate(words):
        for q, p in enumerate(ph):
            exp_ph.append(p)
            exp_wi.append(k)
            exp_t.append(k * 0.3 + (q + 0.5) / len(ph) * 0.3)
    rec_ph = [p for p, _ in rec]
    rec_t = [t for _, t in rec]
    ops = align_phones(exp_ph, exp_wi, exp_t, rec_ph, rec_t)
    return ops, insertion_slots(ops, exp_wi, len(words))


def test_perfect_recognition_has_no_insertions():
    words = [["dh", "ax"], ["k", "ae", "t"]]
    rec = [("dh", 0.07), ("ax", 0.22), ("k", 0.35), ("ae", 0.45), ("t", 0.55)]
    ops, (slots, inside) = run_align(words, rec)
    assert all(o[0] == "M" for o in ops) and slots == {} and inside == 0


def test_filler_between_words_becomes_a_gap_at_the_boundary():
    words = [["dh", "ax"], ["k", "ae", "t"]]
    # dh ax [ah m] k ae t -- the filler sits in the 0.3-0.6 s window, words shifted to make room
    exp = words
    rec = [("dh", 0.07), ("ax", 0.22), ("ah", 0.3), ("m", 0.4), ("k", 0.65), ("ae", 0.75), ("t", 0.85)]
    _, (slots, inside) = run_align(exp, rec)
    assert slots == {1: [2, 3]} and inside == 0


def test_repeated_word_inserts_at_boundary_not_inside_word():
    words = [["dh", "ax"], ["k", "ae", "t"]]
    rec = [("dh", 0.07), ("ax", 0.22), ("dh", 0.35), ("ax", 0.45), ("k", 0.6), ("ae", 0.7), ("t", 0.8)]
    _, (slots, inside) = run_align(words, rec)
    assert inside == 0 and list(slots) == [1]                    # gap between "the" and "cat"
    assert [rec[j][0] for j in slots[1]] == ["dh", "ax"]


def test_leading_and_trailing_insertions_use_edge_slots():
    words = [["k", "ae", "t"]]
    rec = [("ah", 0.0), ("k", 0.05), ("ae", 0.15), ("t", 0.25), ("m", 0.5)]
    _, (slots, _) = run_align(words, rec)
    assert slots == {0: [0], 1: [4]}


def test_missed_phone_is_a_deletion_not_a_gap():
    words = [["k", "ae", "t"], ["s", "ae", "t"]]
    rec = [("k", 0.05), ("t", 0.25), ("s", 0.35), ("ae", 0.45), ("t", 0.55)]     # "ae" of cat missed
    ops, (slots, _) = run_align(words, rec)
    assert slots == {} and sum(o[0] == "D" for o in ops) == 1


def test_substitution_of_confusable_phone_is_not_an_insertion():
    words = [["b", "ae", "t"]]
    rec = [("p", 0.05), ("ae", 0.15), ("t", 0.25)]
    ops, (slots, _) = run_align(words, rec)
    assert [o[0] for o in ops] == ["S", "M", "M"] and slots == {}


def test_empty_sides():
    assert align_phones([], [], [], ["ah"], [0.1]) == [("I", None, 0)]
    assert align_phones(["ah"], [0], [0.1], [], []) == [("D", 0, None)]
    slots, inside = insertion_slots([("I", None, 0), ("I", None, 1)], [], 0)
    assert slots == {0: [0, 1]} and inside == 0


# ------------------------------------------------------------------ gap extraction

class FakeG2P:
    table = {"the": ["dh", "ax"], "cat": ["k", "ae", "t"], "sat": ["s", "ae", "t"]}

    def phones(self, w):
        return self.table[w]

    def variants(self, w):
        return [self.table[w]] + self.extra.get(w, [])

    extra: dict = {}


def rec(ph, t0, step=0.1):
    return [RecPhone(p, t0 + k * step, t0 + k * step + 0.04, 0.9) for k, p in enumerate(ph)]


def test_find_gaps_global_slot_and_stats():
    words = [Word("the", 0.0, 0.3), Word("cat", 0.6, 0.9)]
    r = rec(["dh", "ax"], 0.05) + rec(["ah", "m"], 0.3) + rec(["k", "ae", "t"], 0.62)
    gaps, st = find_gaps(words, r, FakeG2P(), base_word=10)
    assert [(g.slot, g.phones) for g in gaps] == [(11, ["ah", "m"])]
    assert st["kept"] == 1 and st["matched"] == 5 and st["deleted"] == 0
    g = gaps[0]
    # "ah" lasts until "m" (0.1 s); "m" until "k" (0.22 s away) but is capped at 0.15 s
    assert g.start == pytest.approx(0.3) and g.dur == pytest.approx(0.25)


def test_find_gaps_filters():
    words = [Word("the", 0.0, 0.3), Word("cat", 0.6, 0.9)]
    r = rec(["dh", "ax"], 0.05) + rec(["ah"], 0.3) + rec(["k", "ae", "t"], 0.62)
    assert find_gaps(words, r, FakeG2P(), params=GapParams(min_phones=2))[1]["too_short"] == 1
    assert find_gaps(words, r, FakeG2P(), params=GapParams(max_phones=0))[1]["too_long"] == 1
    assert find_gaps(words, r, FakeG2P(), params=GapParams(min_prob=0.95))[1]["low_prob"] == 1
    assert find_gaps(words, r, FakeG2P(), params=GapParams(min_dur=0.2))[1]["too_brief"] == 1   # 0.15 s (capped)
    assert find_gaps(words, r, FakeG2P(), params=GapParams(min_dur=0.15))[1]["kept"] == 1


def test_chunk_with_no_words_turns_all_phones_into_one_gap():
    gaps, st = find_gaps([], rec(["ah", "m"], 1.0), FakeG2P(), base_word=7, params=GapParams(min_dur=0.0))
    assert [(g.slot, g.phones) for g in gaps] == [(7, ["ah", "m"])]


def test_phone_durations_clip_to_cap_and_floor():
    r = [RecPhone("a", 0.0, 0.02, 1), RecPhone("b", 0.01, 0.03, 1), RecPhone("c", 1.0, 1.02, 1)]
    assert phone_durations(r, cap=0.15, floor=0.04) == [0.04, 0.15, 0.04]


def test_build_examples_context_windows():
    cand = find_gaps([Word("the", 0.0, 0.3), Word("cat", 0.6, 0.9)],
                     rec(["dh", "ax"], 0.05) + rec(["ah"], 0.3) + rec(["k", "ae", "t"], 0.62), FakeG2P())[0]
    ex = build_examples(cand, ["the", "cat"], sid="x", ctx=16)
    assert ex[0]["left"] == ["the"] and ex[0]["right"] == ["cat"] and ex[0]["units"] == [] and ex[0]["kind"] == "gap"


def test_to_swb_splits_multi_phone_tokens():
    class M:
        def lookup(self, t):
            return {"aɪ": ["ay"], "ɑːɹ": ["aa", "r"], "?": []}[t]
    out = to_swb([{"tok": "ɑːɹ", "start": 1.0, "end": 1.2, "p": 0.5}, {"tok": "?", "start": 2, "end": 2.1, "p": 1}], M())
    assert [(p.ph, round(p.start, 2), round(p.end, 2)) for p in out] == [("aa", 1.0, 1.1), ("r", 1.1, 1.2)]


# ------------------------------------------------------------------ restoration

def test_render_restoration_classes():
    left, right = ["so", "i", "went"], ["the", "store"]
    assert render_restoration("FIL:um", left, right) == "um"
    assert render_restoration("W:well", left, right) == "well"
    assert render_restoration("COPY_R1", left, right) == "the"
    assert render_restoration("COPY_L2", left, right) == "i"
    assert render_restoration("COPY_R4", left, right) == UNKNOWN          # pointer beyond the context
    assert render_restoration("UNK", left, right) == UNKNOWN
    assert render_restoration("DROP", left, right) is None
    assert render_restoration("SELF", left, right) is None


def test_restore_inserts_in_order_and_to_text():
    from pipeline.extract import GapCand
    words = [Word("so", 0, .2), Word("the", .5, .7), Word("store", .8, 1.0)]
    gap = GapCand(slot=1, phones=["ah", "m", "dh", "ax"], start=.2, end=.45, dur=.3, probs=[1] * 4,
                  times=[(.2, .24), (.3, .34), (.35, .39), (.4, .44)])
    preds = [[{"s": 2, "e": 4, "type": "REPETITION", "restoration": "COPY_R1"},
              {"s": 0, "e": 2, "type": "FILLER_HESITATION", "restoration": "FIL:um"}]]
    toks = restore(words, [gap], preds)
    assert to_text(toks) == "so um the the store"
    assert to_text(toks, annotate=True) == "so [FILLER_HESITATION:um] [REPETITION:the] the store"
    assert [t.start for t in toks if t.source == "restored"] == [.2, .35]


def test_restore_drops_noise_and_handles_trailing_slot():
    from pipeline.extract import GapCand
    words = [Word("so", 0, .2)]
    gaps = [GapCand(1, ["s"], .3, .4, .1, [1], [(.3, .4)]), GapCand(1, ["ah"], .5, .6, .1, [1], [(.5, .6)])]
    preds = [[{"s": 0, "e": 1, "type": "NOISE", "restoration": "DROP"}], [{"s": 0, "e": 1, "type": "FILLER_HESITATION", "restoration": "FIL:uh"}]]
    assert to_text(restore(words, gaps, preds)) == "so uh"


# ------------------------------------------------------------------ pronunciation variants

def test_fit_cost_ignores_surrounding_phones_and_charges_missing_ones():
    from pipeline.align import fit_cost
    assert fit_cost(["k", "ae", "t"], ["s", "k", "ae", "t", "z"]) == 0.0           # neighbours are free
    assert fit_cost(["k", "ae", "t"], ["k", "t"]) == pytest.approx(1.0)            # one deletion
    assert fit_cost(["k", "ae", "t"], []) == pytest.approx(3.0)
    assert fit_cost(["b"], ["p"]) == pytest.approx(0.4)                            # confusable substitution


def test_choose_pronunciation_prefers_variant_supported_by_the_audio():
    from pipeline.align import AlignParams
    from pipeline.extract import choose_pronunciation
    w = Word("and", 1.0, 1.3)
    heard = [RecPhone("s", 0.7, 0.74, 1), RecPhone("ae", 1.05, 1.1, 1), RecPhone("n", 1.15, 1.2, 1), RecPhone("d", 1.25, 1.3, 1)]
    full, reduced = ["ae", "n", "d"], ["ax", "n"]
    assert choose_pronunciation(w, [reduced, full], heard, 0.25, AlignParams()) == full
    assert choose_pronunciation(w, [full, reduced], heard, 0.25, AlignParams()) == full
    assert choose_pronunciation(w, [full], [], 0.25, AlignParams()) == full        # single variant: no search
    assert choose_pronunciation(w, [reduced, full], [], 0.25, AlignParams()) == reduced   # nothing heard: first wins


def test_variant_choice_removes_a_spurious_gap():
    # "and" has a full and a reduced form; the speaker said the full one, so no "d" gap may appear
    class G(FakeG2P):
        table = {"and": ["ax", "n"], "cat": ["k", "ae", "t"]}
        extra = {"and": [["ae", "n", "d"]]}
    words = [Word("and", 0.0, 0.3), Word("cat", 0.4, 0.7)]
    r = rec(["ae", "n", "d"], 0.02) + rec(["k", "ae", "t"], 0.42)
    assert find_gaps(words, r, G())[0] == []
    assert [g.phones for g in find_gaps(words, r, G(), params=GapParams(choose_variants=False))[0]] == [["d"]]


def test_restore_skips_requested_types_but_keeps_the_rest():
    from pipeline.extract import GapCand
    words = [Word("so", 0, .2), Word("the", .5, .7)]
    gap = GapCand(1, ["d", "ah"], .2, .4, .3, [1, 1], [(.2, .3), (.3, .4)])
    preds = [[{"s": 0, "e": 1, "type": "PARTIAL_WORD", "restoration": "COPY_R1"},
              {"s": 1, "e": 2, "type": "FILLER_HESITATION", "restoration": "FIL:uh"}]]
    assert to_text(restore(words, [gap], preds)) == "so the uh the"      # COPY_R1 repeats "the", then the filler
    assert to_text(restore(words, [gap], preds, skip_types=frozenset({"PARTIAL_WORD"}))) == "so uh the"
