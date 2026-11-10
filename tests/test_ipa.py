from pathlib import Path

import pytest

from phones.ipa import IpaMap
from phones.lexicon import SWB_PHONES

ROOT = Path(__file__).parent.parent


@pytest.fixture(scope="module")
def ipa():
    return IpaMap(ROOT / "configs" / "phonemap.yaml")


def test_all_targets_are_valid_swb_phones(ipa):
    for k, v in ipa.table.items():
        assert all(p in SWB_PHONES for p in v), (k, v)


@pytest.mark.parametrize("tok,expected", [
    ("aɪ", ["ay"]), ("dʒ", ["jh"]), ("ɑːɹ", ["aa", "r"]), ("ʔ", []), ("ɾ", ["d"]),
    ("iː", ["iy"]), ("tʃ", ["ch"]), ("ə", ["ax"]), ("ŋ", ["ng"]),
])
def test_known_tokens(ipa, tok, expected):
    assert ipa.lookup(tok) == expected


def test_modifier_variants_are_normalised(ipa):
    assert ipa.lookup("tʰ") == ["t"]
    assert ipa.lookup("kʰ") == ["k"]
    assert ipa.lookup("pː") == ["p"]


def test_special_tokens_and_unknowns_never_raise(ipa):
    assert ipa.lookup("<pad>") == []
    assert ipa.lookup("☃") == []
    assert ipa.map_sequence(["s", "t", "ɑːɹ", "<unk>"]) == ["s", "t", "aa", "r"]


def test_total_on_recognizer_vocab_if_available(ipa):
    """The map must classify every token of the real recognizer vocabulary (when cached locally)."""
    import json
    cands = list((ROOT / ".cache").rglob("w2v_vocab.json"))
    if not cands:
        pytest.skip("recognizer vocab not cached yet")
    vocab = list(json.loads(cands[0].read_text()))
    res = ipa.covers(vocab)
    assert sum(res.values()) == len(vocab)
    assert res["mapped"] >= 50       # the English-relevant core must be mapped
