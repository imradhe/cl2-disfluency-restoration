"""Pipeline pieces that need no model: VAD chunking, ASR word normalisation, CTC collapse, windowing of
words + recognized phones, and restoration rendering."""
import pytest

from pipeline.asr import FILLER_SPELLING, Word, norm_word, normalize_fillers, words_from_segments
from pipeline.audio import chunk_speech
from pipeline.phonerec import RecPhone, collapse, to_swb
from pipeline.restore import UNKNOWN, render_restoration, restore_windows, to_text
from pipeline.windows import Window, make_windows


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


def test_normalize_fillers_respells_whisper_ah_and_touches_nothing_else():
    ws = [Word("so", 0.0, 0.2), Word("ah", 0.3, 0.6, 0.4), Word("yes", 0.7, 0.9), Word("ah", 1.0, 1.2), Word("um", 1.3, 1.5)]
    out, n = normalize_fillers(ws)
    assert [w.text for w in out] == ["so", "uh", "yes", "uh", "um"] and n == 2           # "um" is already Switchboard spelling
    assert (out[1].start, out[1].end, out[1].prob) == (0.3, 0.6, 0.4)                       # times and confidence are kept
    assert ws[1].text == "ah" and out[0] is ws[0]                                           # input is not modified
    assert normalize_fillers(ws, {})[1] == 0 and normalize_fillers([])[0] == []
    assert normalize_fillers(ws, {"yes": "yeah"})[0][2].text == "yeah" and FILLER_SPELLING == {"ah": "uh"}


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



def test_to_swb_splits_multi_phone_tokens():
    class M:
        def lookup(self, t):
            return {"aɪ": ["ay"], "ɑːɹ": ["aa", "r"], "?": []}[t]
    out = to_swb([{"tok": "ɑːɹ", "start": 1.0, "end": 1.2, "p": 0.5}, {"tok": "?", "start": 2, "end": 2.1, "p": 1}], M())
    assert [(p.ph, round(p.start, 2), round(p.end, 2)) for p in out] == [("aa", 1.0, 1.1), ("r", 1.1, 1.2)]


# ------------------------------------------------------------------ windows

class FakeG2P:
    table = {"the": ["dh", "ax"], "cat": ["k", "ae", "t"], "sat": ["s", "ae", "t"], "on": ["aa", "n"]}

    def phones(self, w):
        return self.table.get(w, ["ax"])


def rec(ph, t0, step=0.1):
    return [RecPhone(p, t0 + k * step, t0 + k * step + 0.04, 0.9) for k, p in enumerate(ph)]


def test_make_windows_single_window_keeps_all_phones_in_order():
    words = [Word("the", 0.0, 0.3), Word("cat", 0.6, 0.9)]
    r = rec(["dh", "ax"], 0.05) + rec(["ah", "m"], 0.3) + rec(["k", "ae", "t"], 0.62)
    (w,) = make_windows(words, r, FakeG2P(), 0.0, 1.0)
    assert [x.text for x in w.words] == ["the", "cat"] and w.final
    assert w.exp == [["dh", "ax"], ["k", "ae", "t"]]
    assert w.rec == ["dh", "ax", "ah", "m", "k", "ae", "t"]            # the filler is kept, nothing is aligned away
    ex = w.example("x", 3)
    assert ex["id"] == "x:w3" and ex["slots"] == [[], [], []] and ex["wtags"] == [0, 0]


def test_make_windows_splits_by_word_count_at_the_longest_pause_and_by_time():
    words = [Word("cat", k * 0.5, k * 0.5 + 0.3) for k in range(12)]
    words[8] = Word("cat", 4.4, 4.7)                                       # 0.1 s later than the rest: a pause before word 8
    r = []
    for k, w in enumerate(words):
        r += rec(["k", "ae", "t"], w.start + 0.02, step=0.08)
    wins = make_windows(words, r, FakeG2P(), 0.0, 6.5, max_words=10)
    assert [len(w.words) for w in wins] == [8, 4]                           # cut moved back to the pause before word 8
    assert [w.final for w in wins] == [False, True]
    assert sum(len(w.rec) for w in wins) == len(r) and len(wins[0].rec) == 8 * 3 and len(wins[1].rec) == 4 * 3
    assert wins[0].t1 == wins[1].t0


def test_make_windows_lone_filler_chunk_and_empty_chunk():
    (w,) = make_windows([], rec(["ah", "m"], 1.0), FakeG2P(), 0.5, 2.0)
    assert w.words == [] and w.rec == ["ah", "m"] and w.final and w.example("x", 0)["slots"] == [[]]
    assert make_windows([], [], FakeG2P(), 0.0, 1.0) == []


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


def _win(words, t0=0.0, t1=5.0):
    ws = [Word(w, 0.5 + k * 0.5, 0.8 + k * 0.5) for k, w in enumerate(words)]
    return Window(ws, [["ax"]] * len(ws), [], t0, t1, True)


def test_restore_windows_inserts_in_front_of_the_slot_word_in_order():
    win = _win(["so", "the", "store"])
    slots = [[], ["FILLER_HESITATION|FIL:um", "REPETITION|COPY_R1"], [], []]
    toks = restore_windows([win], [slots])
    assert to_text(toks) == "so um the the store"
    assert to_text(toks, annotate=True) == "so [FILLER_HESITATION:um] [REPETITION:the] the store"
    um, rep = [t for t in toks if t.source == "restored"]
    assert 0.8 <= um.start < um.end <= rep.end <= 1.0 and um.end == pytest.approx(rep.start)   # the pause after "so"


def test_restore_windows_end_slot_noise_and_skip_types():
    win = _win(["so"])
    slots = [["PARTIAL_WORD|COPY_R1"], ["NOISE|DROP", "FILLER_HESITATION|FIL:uh"]]
    assert to_text(restore_windows([win], [slots])) == "so so uh"          # the partial-word unit copies "so"
    toks = restore_windows([win], [slots], skip_types=frozenset({"PARTIAL_WORD"}))
    assert to_text(toks) == "so uh"                                       # NOISE|DROP inserts nothing; end slot works
    assert toks[-1].end == 5.0                                            # the end slot runs to the window end


def test_restore_windows_across_windows():
    a, b = _win(["so"]), _win(["yes"])
    toks = restore_windows([a, b], [[[], []], [["FILLER_DM|FIL:well"], []]])
    assert to_text(toks) == "so well yes"
    assert [t.window for t in toks] == [0, 1, 1]
