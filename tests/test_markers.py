from swb.markers import (
    ASIDE, COINAGE, MISPRON, NOISE, NOISE_WORD, PARTIAL, SILENCE, WORD,
    parse_token, parse_utterance,
)


def test_partial_end_cut():
    t = parse_token("an[y]-")
    assert (t.kind, t.spoken, t.intended, t.cut) == (PARTIAL, "an", "any", "end")
    assert t.lex_key == "an[y]-"


def test_partial_longer_stem():
    t = parse_token("ass[uming]-")
    assert (t.kind, t.spoken, t.intended, t.cut) == (PARTIAL, "ass", "assuming", "end")


def test_partial_apostrophe_in_bracket():
    t = parse_token("i['m]-")
    assert (t.kind, t.spoken, t.intended, t.cut) == (PARTIAL, "i", "i'm", "end")


def test_partial_start_cut():
    t = parse_token("-[th]at")
    assert (t.kind, t.spoken, t.intended, t.cut) == (PARTIAL, "at", "that", "start")


def test_partial_both_cut():
    t = parse_token("-[a]b[out]-")
    assert (t.kind, t.spoken, t.intended, t.cut) == (PARTIAL, "b", "about", "both")


def test_bare_partial_i():
    t = parse_token("i-")
    assert (t.kind, t.spoken, t.intended, t.cut) == (PARTIAL, "i", "", "end")


def test_mispronunciation():
    t = parse_token("[splace/space]")
    assert (t.kind, t.spoken, t.intended, t.lex_key) == (MISPRON, "splace", "space", "[splace/space]")
    assert t.cut == ""


def test_mispronounced_partial_nested_brackets():
    t = parse_token("[Cam[pany]-/Company]")
    assert (t.kind, t.spoken, t.intended, t.cut) == (MISPRON, "cam", "company", "end")
    t2 = parse_token("[Schmor[koff]-/Schwar[zkoff]-]")
    assert (t2.kind, t2.spoken, t2.intended) == (MISPRON, "schmor", "schwarzkoff")


def test_mispronunciation_with_trailing_dash():
    t = parse_token("[flage/fade]-")
    assert (t.kind, t.spoken, t.intended, t.cut) == (MISPRON, "flage", "fade", "end")


def test_noises():
    assert parse_token("[laughter]").kind == NOISE
    assert parse_token("[noise]").noise == "noise"
    assert parse_token("[vocalized-noise]").noise == "vocalized-noise"
    assert parse_token("[silence]").kind == SILENCE


def test_noise_over_word():
    t = parse_token("[laughter-don't]")
    assert (t.kind, t.noise, t.spoken) == (NOISE_WORD, "laughter", "don't")
    assert t.is_speech


def test_coinage_and_aside():
    c = parse_token("{weatherwise}")
    assert (c.kind, c.spoken) == (COINAGE, "weatherwise")
    assert parse_token("<b_aside>").kind == ASIDE
    assert not parse_token("<e_aside>").is_speech


def test_variant_keeps_lex_key_and_normalizes_word():
    t = parse_token("about_1")
    assert (t.kind, t.spoken, t.lex_key) == (WORD, "about", "about_1")


def test_hyphenated_backchannel_is_a_word_not_a_partial():
    for w in ("uh-huh", "um-hum", "huh-uh", "hum-um"):
        t = parse_token(w)
        assert t.kind == WORD and t.spoken == w


def test_plain_word_and_case_folding():
    t = parse_token("Hello")
    assert (t.kind, t.spoken, t.lex_key) == (WORD, "hello", "Hello")


def test_parse_utterance_real_line():
    # real line from sw2001A: partials, noises, backchannels
    toks = parse_utterance("well i was i was in a uh private so um and uh i['m]- i'm not [laughter] but")
    kinds = [t.kind for t in toks]
    assert kinds.count(PARTIAL) == 1 and kinds.count(NOISE) == 1
    assert all(t.is_speech for t in toks if t.kind in (WORD, PARTIAL))
