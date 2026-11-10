from pathlib import Path

from nptel.rsml import parse_rsml, tag_text, to_side

SAMPLE = """# BhashaCheck config
[settings]
default_code_mixing_language = en

[tags]
hesitations = @umm, @uhh

1
00:00:15,228 --> 00:00:18,276
primary=1|verified=1|flagged=0|note=
We are discussing about !!ml[artificial neural network](artificial neural network)

2
00:00:34,876 --> 00:00:41,956
primary=1|verified=0|flagged=1|note=ASR text had 'Pitch powered'
@uhh @repair-start feed for - !!ml[feed forward neural network](feed-forward neural network) @repair-end, @false-start-start we will be concentrating on a - let us model @false-start-end particular neuron
"""


def labels(text):
    toks, _ = tag_text(text)
    return [(t["spoken"], t["label"]) for t in toks]


def test_header_segments_and_verified_flags(tmp_path: Path):
    p = tmp_path / "x.rsml"
    p.write_text(SAMPLE, encoding="utf-8")
    r = parse_rsml(p)
    assert len(r.segments) == 2
    assert r.header["settings"]["default_code_mixing_language"] == "en"
    assert (r.segments[0].start, r.segments[0].end) == (15.228, 18.276)
    assert [s.verified for s in r.segments] == [1, 0]
    assert not r.fully_verified and r.is_english


def test_entities_use_bracket_text():
    assert labels("about !!ml[output layer]() and !!mathex[ith]($i^{th}$) layer") == [
        ("about", "O"), ("output", "O"), ("layer", "O"), ("and", "O"), ("ith", "O"), ("layer", "O")]
    # latex containing parentheses must not leak into the text
    assert labels("at !!mathex[i minus one th]($(i - 1)^{th}$) layer")[1:3] == [("i", "O"), ("minus", "O")]


def test_hesitation_tags_become_filler_words():
    assert labels("and @uhh so @umm yes @uh-huh") == [
        ("and", "O"), ("uh", "FILLER_HESITATION"), ("so", "O"), ("um", "FILLER_HESITATION"),
        ("yes", "O"), ("uh-huh", "FILLER_BACKCHANNEL")]


def test_repair_only_reparandum_is_labelled():
    got = labels("@repair-start feed for - feed forward @repair-end, network")
    assert got == [("feed", "REPAIR"), ("for", "REPAIR"), ("feed", "O"), ("forward", "O"), ("network", "O")]


def test_false_start_and_nested_hesitation_in_reparandum():
    got = labels("@repair-start of a @uhh layer - of a particular layer @repair-end which")
    assert got[:3] == [("of", "REPAIR"), ("a", "REPAIR"), ("uh", "FILLER_HESITATION")]
    assert [l for w, l in got[3:]] == ["REPAIR", "O", "O", "O", "O", "O"]
    fs = labels("@false-start-start we will be on a - let us model @false-start-end the")
    assert [l for _, l in fs] == ["FALSE_START"] * 5 + ["O"] * 4


def test_repetition_filler_broken_word_prolongation():
    assert labels("the @repetition-start in @repetition-end in our") == [
        ("the", "O"), ("in", "REPETITION"), ("in", "O"), ("our", "O")]
    assert labels("a @filler-start you know @filler-end b") == [
        ("a", "O"), ("you", "FILLER_DM"), ("know", "FILLER_DM"), ("b", "O")]
    assert labels("also @broken-word-start un @broken-word-end interpret") == [
        ("also", "O"), ("un", "PARTIAL_WORD"), ("interpret", "O")]
    assert labels("camera @prolongation-start is @prolongation-end denoted")[1] == ("is", "PROLONGATION")


def test_isolated_and_prosody_tags_are_handled():
    toks, dropped = tag_text("so @pause it @falling-pitch-start goes @falling-pitch-end @stutter-block down")
    assert [t["spoken"] for t in toks] == ["so", "it", "goes", "down"]
    assert dropped == {"@pause": 1, "@stutter-block": 1}


def test_to_side_record_shape(tmp_path: Path):
    p = tmp_path / "lec.rsml"
    p.write_text(SAMPLE, encoding="utf-8")
    side = to_side(parse_rsml(p))
    t = side["tokens"][0]
    assert t["i"] == 0 and t["utt_first"] and "phones" in t and t["end"] > t["start"]
    assert side["sid"] == "lec"


ALIGNED = (
    "1\n00:00:10,000 --> 00:00:12,000\nprimary=1|verified=1|flagged=0|note=\nwe @uhh will\n"
    "\t00:00:10,100 --> 00:00:10,300\n\tconfidence=0.9|weakest_letter_confidence=0.8|voiced_fraction=1.00|pause_after_milliseconds=0|pause_voiced_fraction=0.00|kind=word|script=latin|flagged=0\n\twe\n"
    "\t00:00:10,500 --> 00:00:10,900\n\tconfidence=|weakest_letter_confidence=|voiced_fraction=1.00|pause_after_milliseconds=40|pause_voiced_fraction=0.00|kind=hesitation|script=latin|flagged=0\n\t@uhh\n"
    "\t00:00:11,000 --> 00:00:11,200\n\tconfidence=0.7|weakest_letter_confidence=0.5|voiced_fraction=1.00|pause_after_milliseconds=0|pause_voiced_fraction=0.00|kind=word|script=latin|flagged=0\n\twill\n"
)


def test_aligned_entries_give_real_times_and_labels(tmp_path):
    from nptel.rsml import aligned_to_side, parse_aligned
    p = tmp_path / "x.aligned.rsml"
    p.write_text(ALIGNED, encoding="utf-8")
    segs = parse_aligned(p)
    assert len(segs) == 1 and [e["kind"] for e in segs[0].entries] == ["word", "hesitation", "word"]
    assert segs[0].entries[1]["conf"] is None and segs[0].entries[0]["conf"] == 0.9
    side = aligned_to_side(p)
    toks = side["tokens"]
    assert [(t["spoken"], t["label"]) for t in toks] == [("we", "O"), ("uh", "FILLER_HESITATION"), ("will", "O")]
    assert (toks[1]["start"], toks[1]["end"]) == (10.5, 10.9)
    assert toks[1]["pause_before"] == 0.2 and toks[0]["pause_before"] == 0.0
    assert side["sid"] == "x"
