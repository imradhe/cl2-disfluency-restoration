"""Fitted recognizer-error channel: alignment, profile fitting, token-wise replay, and its use in the windows."""
import random

import pytest

from model.seq import SeqDataset, build_seq_side
from phones.channel import (
    HES, MIN_HES, ProfileNoise, align, bucket, fit_profile, load_profile, save_profile, stretch_outcomes,
)
from phones.noise import PhoneNoise

from test_seq import VOCAB, FakeTok, Lex, example, side, tok


def test_align_marks_matches_substitutions_deletions_and_insertions():
    assert align(["a", "b", "c"], ["a", "b", "c"]) == [("a", "a"), ("b", "b"), ("c", "c")]
    assert align(["a", "b", "c"], ["a", "x", "c"]) == [("a", "a"), ("b", "x"), ("c", "c")]       # substitution, not del+ins
    assert align(["a", "b", "c"], ["a", "c"]) == [("a", "a"), ("b", None), ("c", "c")]
    assert align(["a", "c"], ["a", "x", "c"]) == [("a", "a"), (None, "x"), ("c", "c")]
    assert align([], ["a"]) == [(None, "a")] and align(["a"], []) == [("a", None)]


def test_bucket_by_number_of_phones():
    assert [bucket(n) for n in (1, 2, 3, 7)] == ["1", "2", "3+", "3+"]


def rows(n_hes_empty, n_hes_ax, n_words=200):
    out = [(HES, ["ah"], [])] * n_hes_empty + [(HES, ["ah"], ["ax"])] * n_hes_ax
    out += [("O", ["k", "ae", "t"], ["k", "ae", "t"])] * n_words
    out += [("O", ["dh", "ax"], ["dh", "ih"])] * 30 + [("O", ["ax"], [])] * 10 + [("O", ["ax"], ["ax"])] * 30
    return out


def test_stretch_alignment_does_not_book_a_shifted_phone_as_an_error():
    toks = [("O", ["k", "ae", "t"]), ("O", ["s", "ow"])]
    a, b = stretch_outcomes(toks, ["k", "ae", "t", "s", "ow"])                                # recognizer is perfect ...
    assert a["out"] == ["k", "ae", "t"] and b["out"] == ["s", "ow"] and not any(a["ins"] + b["ins"])
    # ... even when the phone times put the "t" in the next word's span: per word that looks like 1 deletion + 1 insertion
    per_word = [stretch_outcomes([t], h) for t, h in zip(toks, (["k", "ae"], ["t", "s", "ow"]))]
    assert per_word[0][0]["out"][2] is None and per_word[1][0]["ins"][0] == []                # (the old, wrong bookkeeping)
    a, b = stretch_outcomes(toks, ["k", "ae", "dh", "s", "ow"])
    assert a["out"] == ["k", "ae", "dh"] and a["ref"] == ["k", "ae", "t"]                       # substitution
    a, b = stretch_outcomes(toks, ["k", "ae", "s", "ow"])
    assert a["out"] == ["k", "ae", None] and b["out"] == ["s", "ow"]                            # deletion
    a, b = stretch_outcomes(toks, ["k", "ae", "t", "n", "s", "ow"])
    assert a["ins"][2] == ["n"]                                                                 # stray after the last phone of A
    a, b = stretch_outcomes(toks, ["n", "k", "ae", "t", "s", "ow"])
    assert not any(a["ins"] + b["ins"])                                                         # leading insertion = stretch edge, ignored
    assert stretch_outcomes([], ["a"]) == [] and stretch_outcomes([("O", ["a"])], [])[0]["out"] == [None]


def test_fit_profile_measures_hesitation_dropout_token_drops_and_phone_rates():
    p = fit_profile(rows(30, 30))
    assert p["hes"]["n"] == 60 and 0.45 < p["hes"]["p_empty"] < 0.55 and p["hes"]["strings"] == [[["ax"], 30]]
    assert p["tok_del"]["1"] == pytest.approx((10 + 0.1) / (40 + 5), abs=1e-3)               # 10 of 40 one-phone words vanish
    assert p["tok_del"]["3+"] < 0.01
    assert p["phones"]["ax"]["sub"] > p["phones"]["k"]["sub"]                                  # dh ax -> dh ih shows up as ax -> ih
    assert p["phones"]["ax"]["to"]["ih"] > 0.5
    assert abs(sum(p["phones"]["ax"]["to"].values()) - 1) < 0.01
    assert "orphan" not in p                                                                   # no stray phones in these rows
    assert fit_profile([("O", ["k", "ae", "t"], ["k", "ae", "t"])])["tok_del"]["1"] < 0.05     # a bucket with no data: ~2%, not 50%


def test_strays_are_the_insertions_after_the_last_phone_of_a_token_the_others_are_per_phone():
    r = ([("O", ["s", "ae", "t"], ["s", "ae", "t", "n"])] * 10 + [("O", ["s", "ae", "t"], ["s", "ae", "t"])] * 30
         + [("O", ["s", "ae", "t"], ["s", "ah", "ae", "t"])] * 10)
    p = fit_profile(r)
    assert p["orphan"]["p"] == pytest.approx(10 / 50) and p["orphan"]["len"] == {"1": 10} and p["orphan"]["pool"] == {"n": 10}
    assert p["ins_pool"] == {"ah": 10}                                                         # inside the token, after "s"
    assert p["phones"]["s"]["ins"] > 0.1 and p["phones"]["t"]["ins"] < p["phones"]["s"]["ins"]


def test_hesitations_are_kept_out_of_the_general_phone_statistics_and_need_enough_samples():
    p = fit_profile(rows(MIN_HES - 1, 0))
    assert "hes" not in p                                                                      # too few: generic channel only
    p = fit_profile(rows(25, 0, n_words=0))
    assert p["hes"]["p_empty"] > 0.9 and "ah" not in p["phones"]                               # 'ah' of "uh" does not pollute 'ah'


def test_an_uh_counts_as_written_only_when_aligned_to_a_vowel():
    # a consonant aligned to the "uh" is a neighbour's phone, not a rendering of it
    r = [(HES, ["ah"], ["aa"])] * 10 + [(HES, ["ah"], ["d"])] * 10 + [(HES, ["ah"], [])] * 10 + [(HES, ["ah"], ["ax"])] * 10
    p = fit_profile(r)
    assert p["hes"]["n"] == 40 and p["hes"]["p_empty"] == pytest.approx(21 / 42)                # 20 of 40 not vowels (+1/+2 smoothing)
    assert sorted(map(tuple, (s for s, _ in p["hes"]["strings"]))) == [("aa",), ("ax",)]


def test_profile_roundtrip(tmp_path):
    p = fit_profile(rows(30, 30))
    save_profile(p, tmp_path / "p.json")
    assert load_profile(tmp_path / "p.json") == p


def stream(*toks):
    """phones, groups, classes of a list of (label, phones)."""
    ph, g = [], []
    for i, (_, p) in enumerate(toks):
        ph += p
        g += [i] * len(p)
    return ph, g, [lab for lab, _ in toks]


def test_profile_noise_drops_hesitations_but_not_other_tokens():
    prof = fit_profile(rows(30, 0, n_words=0) + rows(0, 0, n_words=100))
    assert prof["hes"]["p_empty"] > 0.9
    ph, g, cl = stream(("O", ["k", "ae", "t"]), (HES, ["ah"]), ("O", ["s", "ae", "t"]))
    noise = ProfileNoise({**prof, "tok_del": {"1": 0.0, "2": 0.0, "3+": 0.0}}, p_profile=1.0, seed=0)
    gone = 0
    for _ in range(100):
        out, src = noise.corrupt_idx(ph, g, cl)
        assert len(out) == len(src) and all(0 <= s < len(ph) for s in src)
        gone += "ah" not in out and 3 not in src
    assert gone >= 85                                                                          # the "uh" almost always vanishes
    out, src = ProfileNoise({**prof, "hes": {**prof["hes"], "p_empty": 0.0, "strings": [[["aa"], 1]]}},
                            p_profile=1.0, seed=0).corrupt_idx(ph, g, cl)
    assert "aa" in out and src[out.index("aa")] == 3                                           # rewritten, still tied to its token


def test_profile_noise_without_token_structure_or_with_p_profile_zero_is_the_iid_noise():
    prof = fit_profile(rows(30, 30))
    ph = ["k", "ae", "t", "s", "ae", "t"] * 4
    a = ProfileNoise(prof, p_profile=1.0, seed=3).corrupt_idx(ph)
    assert a == PhoneNoise(seed=3).corrupt_idx(ph)
    # p_profile=0: the fitted channel is never used (the branch itself costs one random draw, so the stream differs
    # from a bare PhoneNoise, but a hesitation then survives at the iid rate instead of being dropped half of the time)
    hes_prof = fit_profile(rows(60, 0, n_words=0))
    ph2, g2, cl2 = stream(("O", ["k", "ae", "t"]), (HES, ["ah"]), ("O", ["s", "ae", "t"]))
    kept = sum("ah" in ProfileNoise(hes_prof, 0.0, seed=s).corrupt_idx(ph2, g2, cl2)[0] for s in range(200))
    gone = sum("ah" in ProfileNoise(hes_prof, 1.0, seed=s).corrupt_idx(ph2, g2, cl2)[0] for s in range(200))
    assert kept > 150 and gone < 20


def test_profile_noise_is_deterministic_never_empty_and_checks_the_structure():
    prof = fit_profile(rows(30, 30))
    ph, g, cl = stream(("O", ["k", "ae", "t"]), ("REPETITION", ["dh", "ax"]), ("O", ["ax"]))
    assert ProfileNoise(prof, 1.0, seed=5).corrupt_idx(ph, g, cl) == ProfileNoise(prof, 1.0, seed=5).corrupt_idx(ph, g, cl)
    drop_all = {**prof, "tok_del": {"1": 1.0, "2": 1.0, "3+": 1.0}}
    out, src = ProfileNoise(drop_all, 1.0, seed=0).corrupt_idx(ph, g, cl)
    assert out == ph[:1] and src == [0]                                                        # same guard as PhoneNoise
    with pytest.raises(AssertionError):
        ProfileNoise(prof, 1.0, seed=0).corrupt_idx(ph, g[:-1], cl)


def test_noise_tokens_pass_through_unchanged_and_strays_inherit_the_previous_token():
    prof = fit_profile([("O", ["k", "ae", "t"], ["k", "ae", "t", "n"])] * 40)                  # a stray after every token
    assert prof["orphan"]["p"] == 1.0
    ph, g, cl = stream(("NOISE", ["s", "t"]), ("O", ["k", "ae", "t"]))
    out, src = ProfileNoise({**prof, "tok_del": {"1": 0.0, "2": 0.0, "3+": 0.0}}, 1.0, seed=0).corrupt_idx(ph, g, cl)
    assert out[:2] == ["s", "t"] and src[:2] == [0, 1]
    assert out[2] == "n" and src[2] == 1                                                       # stray after the NOISE token


def test_windows_carry_token_structure_matching_the_recognized_phones():
    s = side(tok(0, "so"), tok(1, "um", "FILLER_HESITATION"), tok(2, "the", "REPETITION"), tok(3, "the"), tok(4, "cat"))
    (ex,) = list(build_seq_side(s, Lex(), random.Random(0), PhoneNoise(seed=0), sizes=(20, 20)))
    assert len(ex["rec_tok"]) == len(ex["rec"]) and ex["rec_tok"] == sorted(ex["rec_tok"])
    assert ex["tok_lab"] == ["O", "FILLER_HESITATION", "REPETITION", "O", "O"]
    assert [ex["tok_lab"][t] for t in ex["rec_tok"]][2:4] == ["FILLER_HESITATION"] * 2        # the "um" phones
    # a hesitation the ASR wrote anyway (keep_p=1) sits in the transcript but keeps its label for the noise
    (kept,) = list(build_seq_side(s, Lex(), random.Random(0), PhoneNoise(seed=0), sizes=(20, 20), keep_p=1.0))
    assert kept["words"][1] == "um" and kept["tok_lab"][1] == "FILLER_HESITATION"
    # spurious noise tokens are labelled NOISE
    (nz,) = list(build_seq_side(side(tok(0, "so"), tok(1, "[n]", "NOISE", kind="NOISE"), tok(2, "cat")), Lex(),
                                random.Random(0), PhoneNoise(seed=0), sizes=(20, 20), noise_p=1.0))
    assert "NOISE" in nz["tok_lab"]


def test_dataset_keeps_aux_labels_in_step_with_the_profile_noise():
    prof = fit_profile(rows(30, 30) + [("O", ["k", "ae", "t"], ["k", "ae", "t", "n"])] * 30)
    ex = example(rec=["s", "ow", "ah", "m"], rec_role=[0, 0, 1, 1], rec_ptr=[0, 0, 1, 1], rec_tok=[0, 0, 1, 1],
                 tok_lab=["O", "FILLER_HESITATION"])
    noise = ProfileNoise(prof, 1.0, seed=2)
    for _ in range(50):
        it = SeqDataset([ex], FakeTok(), VOCAB, noise=noise)[0]
        assert len(it["rec_pos"]) == len(it["rec_role"]) == len(it["rec_ptr"])
        assert all(r in (0, 1) for r in it["rec_role"])
    old = {k: v for k, v in ex.items() if k not in ("rec_tok", "tok_lab")}                       # data built before the profile existed
    assert len(SeqDataset([old], FakeTok(), VOCAB, noise=noise)[0]["rec_pos"]) > 0
