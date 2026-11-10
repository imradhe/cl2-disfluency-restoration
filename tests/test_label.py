from pathlib import Path

import yaml

from swb.corpus import Side, Utt, WTok
from swb.label import (
    FALSE_START, FILLER_BACKCHANNEL, FILLER_DM, FILLER_HESITATION, MISPRONUNCIATION, NOISE_TAG,
    O, PARTIAL_WORD, PROLONGATION, REPAIR, REPETITION, DurStats, label_side,
)
from swb.markers import parse_token

CFG = yaml.safe_load((Path(__file__).parent.parent / "configs" / "labels.yaml").read_text())


def mk_side(utts, tok_dur=0.3, gap=0.6, durs=None):
    """Build a Side from utterance strings; ``durs`` maps token index -> duration."""
    words, t, k = [], 0.0, 0
    for u_i, text in enumerate(utts):
        uid = f"swTESTA-ms98-a-{u_i:04d}"
        for raw in text.split():
            d = (durs or {}).get(k, tok_dur)
            words.append(WTok(uid, t, t + d, parse_token(raw)))
            t += d
            k += 1
        t += gap
    return Side("TEST", "A", [Utt("u", 0, t, " ".join(utts))], words)


def labels(utts, **kw):
    toks = label_side(mk_side(utts, **kw), CFG)
    return [(x.text, x.label) for x in toks]


def only(utts, label, **kw):
    return [w for w, l in labels(utts, **kw) if l == label]


def test_gold_fillers_and_noise():
    got = dict(labels(["um uh-huh [laughter] hello"]))
    assert got["um"] == FILLER_HESITATION
    assert got["uh-huh"] == FILLER_BACKCHANNEL
    assert got["[laughter]"] == NOISE_TAG
    assert got["hello"] == O


def test_short_turn_yeah_is_backchannel_but_long_turn_is_dm():
    assert labels(["yeah"]) == [("yeah", FILLER_BACKCHANNEL)]
    long = dict(labels(["yeah i think the plan works fine for us"]))
    assert long["yeah"] == FILLER_DM and long["think"] == O


def test_you_know_dm_vs_literal():
    assert only(["it was you know pretty bad"], FILLER_DM) == ["you", "know"]
    assert only(["do you know what happened"], FILLER_DM) == []


def test_partial_word_gold():
    got = dict(labels(["an[y]- is there a code"]))
    assert got["an"] == PARTIAL_WORD


def test_mispronunciation_stays_in_word_labels():
    got = dict(labels(["it is in [splace/space] now"]))
    assert got["splace"] == MISPRONUNCIATION


def test_simple_repetition_first_copy_labelled():
    toks = labels(["we went to the the store today"])
    assert [l for w, l in toks if w == "the"] == [REPETITION, O]


def test_triple_repetition_labels_all_but_last():
    toks = labels(["it was very big and and and wide"])
    assert [l for w, l in toks if w == "and"] == [REPETITION, REPETITION, O]


def test_repetition_across_filler():
    toks = labels(["we have um we have time"])
    assert [l for w, l in toks if w == "we"] == [REPETITION, O]
    assert dict(toks)["um"] == FILLER_HESITATION


def test_intentional_reduplication_is_not_a_repetition():
    assert only(["bye bye now"], REPETITION) == []


def test_ngram_repetition():
    toks = labels(["we have a we have a plan"])
    assert [l for w, l in toks if w in ("we", "have", "a")] == [REPETITION] * 3 + [O] * 3


def test_partial_then_completed_word_is_only_partial():
    toks = labels(["he w[ent]- went home"])
    assert dict(toks)["w"] == PARTIAL_WORD
    assert [l for w, l in toks if w == "went"] == [O]


def test_i_partial_then_i_is_not_repetition_label():
    toks = labels(["i- i really felt good"])
    assert dict(toks)["i"] in (PARTIAL_WORD, O)       # fragment keeps its gold label
    assert only(["i- i really felt good"], REPETITION) == []


def test_repair_substitution():
    toks = labels(["we have a red car we have a blue car today"])
    assert [l for w, l in toks if w == "red"] == [REPAIR]
    assert [l for w, l in toks if w == "blue"] == [O]


def test_parallel_phrasing_is_not_a_repair():
    assert only(["we drove from national park to national park today"], REPAIR) == []
    assert only(["i know i think so"], REPAIR) == []


def test_short_repair_needs_a_cue_such_as_a_filler():
    toks = labels(["the green uh the red one"])
    assert [l for w, l in toks if w == "green"] == [REPAIR]
    assert [l for w, l in toks if w == "red"] == [O]


def test_kind_of_literal_vs_hedge():
    assert only(["it is a kind of metal"], FILLER_DM) == []
    assert only(["it was kind of cold"], FILLER_DM) == ["kind", "of"]


def test_false_start_needs_restart_word_and_is_capped():
    # no clause-initial word after the cut -> no false start
    assert only(["the old big red th[ing]- happened"], FALSE_START) == []
    toks = labels(["well the old big red house th[ing]- i saw it"])
    assert len([w for w, l in toks if l == FALSE_START]) <= 4


def test_repair_around_partial_keeps_partial_label():
    toks = labels(["is there an[y]- is there a code"])
    d = [(w, l) for w, l in toks]
    assert ("an", PARTIAL_WORD) in d
    assert [l for w, l in toks if w == "is"] == [REPAIR, O]


def test_false_start_partial_not_completed():
    toks = labels(["so what do you th[ink]- i believe it is fine"])
    fs = [w for w, l in toks if l == FALSE_START]
    assert fs[-3:] == ["what", "do", "you"] or "what" in fs
    assert dict(toks)["th"] == PARTIAL_WORD


def test_no_false_start_for_single_preceding_token():
    toks = labels(["and seaso[n]- really same"])
    assert FALSE_START not in [l for w, l in toks]


def test_prolongation_function_word_uses_duration_stats():
    stats = DurStats()
    stats.mean["the"], stats.std["the"] = -2.0, 0.3     # ~0.135 s typical
    side = mk_side(["we saw the big dog today", "ok"], durs={2: 0.9})
    toks = label_side(side, CFG, stats)
    assert [x.label for x in toks if x.text == "the"] == [PROLONGATION]


def test_phrase_final_lengthening_is_not_prolongation():
    stats = DurStats()
    stats.mean["the"], stats.std["the"] = -2.0, 0.3
    side = mk_side(["we saw the", "big dog"], durs={2: 0.9})
    toks = label_side(side, CFG, stats)
    assert [x.label for x in toks if x.text == "the"] == [O]
