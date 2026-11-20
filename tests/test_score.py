"""End-to-end disfluency F1: word alignment, the rule tags for Whisper's own words, and the two scores."""
import pytest

from pipeline.score import align_idx, disfluency_f1, restoration_f1, rule_tags, system_tokens


def test_align_idx_pairs_equal_substituted_and_unmatched_words():
    assert align_idx(["a", "b", "c"], ["a", "b", "c"]) == [(0, 0), (1, 1), (2, 2)]
    assert align_idx(["a", "x", "c"], ["a", "b", "c"]) == [(0, 0), (1, 1), (2, 2)]            # substitution aligns
    assert align_idx(["a", "c"], ["a", "b", "c"]) == [(0, 0), (None, 1), (1, 2)]               # b missing from the output
    assert align_idx(["a", "b", "c"], ["a", "c"]) == [(0, 0), (1, None), (2, 1)]               # extra b in the output
    assert align_idx([], ["a"]) == [(None, 0)] and align_idx(["a"], []) == [(0, None)] and align_idx([], []) == []


def test_align_idx_is_consistent_on_longer_repetitive_input():
    a = "so the the cat sat on the the mat".split()
    b = "so the cat sat on the mat".split()
    ops = align_idx(a, b)
    assert sorted(i for i, _ in ops if i is not None) == list(range(len(a)))                  # every index exactly once, in order
    assert sorted(j for _, j in ops if j is not None) == list(range(len(b)))
    assert sum(i is None or j is None for i, j in ops) == 2                                    # the two extra "the"


def test_rule_tags_mark_lexical_fillers_and_the_first_of_an_immediate_repeat():
    w = "so uh the the the cat um sat".split()
    assert rule_tags(w) == [None, "FILLER_HESITATION", "REPETITION", "REPETITION", None, None, "FILLER_HESITATION", None]
    assert rule_tags([]) == [] and rule_tags(["ah"]) == [None]                                  # Whisper's "ah" is not tagged unless respelled


def tok(text, source="asr", start=0.0, label=""):
    return {"text": text, "source": source, "start": start, "label": label}


def test_system_tokens_tag_restored_tokens_and_apply_the_rules_to_whisper_words():
    toks = [tok("So,", start=0.0), tok("uh", "restored", 1.0, "FILLER_HESITATION"), tok("the", start=1.0), tok("the", start=2.0),
            tok("cat", start=3.0), tok("...", start=3.5)]
    assert system_tokens(toks) == [("so", None, "rule"), ("uh", "FILLER_HESITATION", "restored"), ("the", "REPETITION", "rule"),
                                   ("the", None, "rule"), ("cat", None, "rule")]
    assert [s[0] for s in system_tokens(toks, drop_types=frozenset({"FILLER_HESITATION"}))] == ["so", "the", "the", "cat"]
    assert all(s[1] is None or s[2] == "restored" for s in system_tokens(toks, rules=False))
    assert [s[0] for s in system_tokens(toks, t_from=2.0)] == ["the", "cat"]


GOLD = [("so", "O"), ("uh", "FILLER_HESITATION"), ("the", "REPETITION"), ("the", "O"), ("cat", "O"), ("sat", "O")]


def asr(*words):
    return [tok(w, start=float(k)) for k, w in enumerate(words)]


def out(*items):
    """Output tokens in order: a plain string is a Whisper word, a (word, type) pair a restored token."""
    return [tok(i, start=float(k)) if isinstance(i, str) else tok(i[0], "restored", float(k), i[1]) for k, i in enumerate(items)]


def test_a_transcript_with_all_its_disfluencies_scores_one_and_a_clean_one_scores_zero():
    full = disfluency_f1(GOLD, system_tokens(asr("so", "uh", "the", "the", "cat", "sat")))
    assert full["untyped"]["f1"] == 1.0 and full["typed"]["f1"] == 1.0 and full["correct_by_source"] == {"rule": 2}
    clean = disfluency_f1(GOLD, system_tokens(asr("so", "the", "cat", "sat")))
    assert clean["untyped"]["tp"] == 0 and clean["untyped"]["f1"] == 0.0 and clean["untyped"]["gold"] == 2


def test_restored_tokens_count_as_correct_only_at_a_matching_gold_token_and_types_are_scored_separately():
    good = system_tokens(out("so", ("uh", "FILLER_HESITATION"), ("the", "REPETITION"), "the", "cat", "sat"))
    r = disfluency_f1(GOLD, good)
    assert r["untyped"]["tp"] == 2 and r["untyped"]["p"] == 1.0 and r["untyped"]["r"] == 1.0
    assert r["correct_by_source"] == {"restored": 2}
    r = disfluency_f1(GOLD, system_tokens(out("so", ("uh", "FILLER_DM"), "the", "cat", "sat")))
    assert r["untyped"]["tp"] == 1 and r["typed"]["tp"] == 0                                   # right word and place, wrong type
    r = disfluency_f1(GOLD, system_tokens(out("so", ("um", "FILLER_HESITATION"), "the", ("cat", "REPETITION"), "cat", "sat")))
    assert r["untyped"]["tp"] == 0 and r["untyped"]["system"] == 2 and r["untyped"]["p"] == 0.0


def test_gold_disfluent_words_that_whisper_wrote_but_nothing_tagged_are_reported_not_credited():
    gold = [("so", "FILLER_DM"), ("the", "O"), ("cat", "O")]
    r = disfluency_f1(gold, system_tokens(asr("so", "the", "cat")))
    assert r["untyped"]["tp"] == 0 and r["gold_present_but_untagged"] == 1


def test_restoration_f1_scores_only_restored_tokens_against_what_whisper_dropped():
    # Whisper already wrote the repetition ("the the"); only the filler is missing from its words
    sysd = system_tokens(out("so", ("uh", "FILLER_HESITATION"), "the", "the", "cat", "sat"))
    r = restoration_f1(GOLD, sysd)
    assert r["gold_missing_from_whisper"] == 1 and r["untyped"] == {"tp": 1, "system": 1, "gold": 1, "p": 1.0, "r": 1.0, "f1": 1.0}
    # restoring a second copy of what Whisper already has, or a word that is not in the gold, is a false positive
    sysd = system_tokens(out("so", "uh", ("uh", "FILLER_HESITATION"), "the", "the", "cat", "sat", ("hmm", "FILLER_HESITATION")))
    r = restoration_f1(GOLD, sysd)
    assert r["gold_missing_from_whisper"] == 0 and r["untyped"]["tp"] == 0 and r["untyped"]["system"] == 2
    assert r["per_type"]["FILLER_HESITATION"]["system"] == 2
    assert restoration_f1(GOLD, system_tokens(asr("so", "the", "cat", "sat")))["untyped"]["f1"] == 0.0   # nothing restored


def test_f1_is_zero_not_a_division_error_without_any_system_or_gold_disfluency():
    assert disfluency_f1([("a", "O")], system_tokens(asr("a")))["untyped"]["f1"] == 0.0
    assert restoration_f1([("a", "O")], system_tokens(asr("a")))["untyped"]["f1"] == 0.0
