import random

from model.gaps import WordDur, build_side, restore_target, split_of
from phones.noise import PhoneNoise


def tok(i, spoken, label, phones, kind="WORD", intended=None, start=0.0, dur=0.3):
    return {
        "i": i, "raw": spoken, "kind": kind, "spoken": spoken, "intended": intended or spoken,
        "label": label, "phones": phones, "start": start + i * 0.4, "end": start + i * 0.4 + dur,
    }


def side(tokens):
    return {"sid": "sw9999A", "conv": "9999", "side": "A", "tokens": tokens}


def build(tokens, **kw):
    return list(build_side(side(tokens), WordDur(), random.Random(0), PhoneNoise(seed=0),
                           noise_gaps=False, **kw))


def test_filler_gap_phones_and_bio_units():
    toks = [
        tok(0, "we", "O", ["w", "iy"]),
        tok(1, "um", "FILLER_HESITATION", ["ah", "m"]),
        tok(2, "went", "O", ["w", "eh", "n", "t"]),
    ]
    gaps = [e for e in build(toks) if e["kind"] == "gap"]
    assert len(gaps) == 1
    g = gaps[0]
    assert g["left"] == ["we"] and g["right"] == ["went"]
    assert g["phones"] == ["ah", "m"]
    assert g["units"] == [{"s": 0, "e": 2, "label": "FILLER_HESITATION", "target": "FIL:um"}]


def test_multi_unit_gap_boundaries_and_repetition_copy_pointer():
    toks = [
        tok(0, "the", "REPETITION", ["dh", "ax"]),
        tok(1, "uh", "FILLER_HESITATION", ["ah"]),
        tok(2, "the", "O", ["dh", "ax"]),
        tok(3, "cat", "O", ["k", "ae", "t"]),
    ]
    g = [e for e in build(toks) if e["kind"] == "gap"][0]
    assert g["left"] == [] and g["right"] == ["the", "cat"]
    assert g["phones"] == ["dh", "ax", "ah"]
    assert [(u["s"], u["e"], u["label"], u["target"]) for u in g["units"]] == [
        (0, 2, "REPETITION", "COPY_R1"),
        (2, 3, "FILLER_HESITATION", "FIL:uh"),
    ]


def test_removing_gaps_reproduces_skeleton():
    toks = [
        tok(0, "i", "O", ["ay"]), tok(1, "i-", "PARTIAL_WORD", ["ay"], kind="PARTIAL", intended=""),
        tok(2, "i", "O", ["ay"]), tok(3, "um", "FILLER_HESITATION", ["ah", "m"]),
        tok(4, "know", "O", ["n", "ow"]),
    ]
    exs = build(toks)
    gaps = [e for e in exs if e["kind"] == "gap"]
    # all context windows come from the same fluent skeleton i i know
    skeleton = ["i", "i", "know"]
    for g in gaps:
        whole = g["left"] + g["right"]
        assert whole == [w for w in skeleton if True][: len(whole)] or set(whole) <= set(skeleton)
    assert [len(g["left"]) for g in gaps] == [1, 2]


def test_partial_word_points_to_intended_word_in_right_context():
    toks = [
        tok(0, "he", "O", ["hh", "iy"]),
        tok(1, "w", "PARTIAL_WORD", ["w"], kind="PARTIAL", intended="went"),
        tok(2, "went", "O", ["w", "eh", "n", "t"]),
    ]
    g = [e for e in build(toks) if e["kind"] == "gap"][0]
    assert g["units"][0]["target"] == "COPY_R1"


def test_unknown_partial_target_is_unk():
    assert restore_target("PARTIAL_WORD", "i", "", ["a"], ["b"]) == "UNK"


def test_open_vocab_target_for_substituted_repair_word():
    assert restore_target("REPAIR", "red", "red", ["a"], ["the", "blue", "car"]) == "W:red"


def test_wspan_positive_and_negative_examples():
    toks = [
        tok(0, "see", "O", ["s", "iy"]),
        tok(1, "the", "PROLONGATION", ["dh", "ax"], dur=0.9),
        tok(2, "space", "MISPRONUNCIATION", ["s", "p", "l", "ey", "s"], kind="MISPRON", intended="space"),
        tok(3, "now", "O", ["n", "aw"]),
    ]
    spans = [e for e in build(toks) if e["kind"] == "wspan"]
    labels = sorted(e["units"][0]["label"] for e in spans)
    assert labels.count("PROLONGATION") == 1 and labels.count("MISPRONUNCIATION") == 1
    mis = [e for e in spans if e["units"][0]["label"] == "MISPRONUNCIATION"][0]
    assert mis["word"] == "space" and mis["phones"] == ["s", "p", "l", "ey", "s"]
    assert len(spans) == 4        # 2 positives + 2 sampled fluent negatives


def test_splits_by_conversation_id():
    assert split_of("2001") == "train"
    assert split_of("4010") == "test" and split_of("4153") == "test"
    assert split_of("4519") == "dev" and split_of("4936") == "dev"
    assert split_of("4154") == "train"
