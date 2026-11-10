import os
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "hf"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

transformers = pytest.importorskip("transformers")

from model.dataset import TAG2ID, TAGS, GapDataset, collate  # noqa: E402
from model.train import build_tokenizer  # noqa: E402

VOCAB = ["<pad>", "SELF", "UNK", "DROP", "COPY_R1", "FIL:um", "W:red"]


@pytest.fixture(scope="module")
def tok():
    try:
        return build_tokenizer("bert-base-uncased")
    except Exception:                      # model not cached / offline
        pytest.skip("bert-base-uncased not cached")


GAP = {
    "id": "t:g0", "sid": "t", "kind": "gap", "left": ["we", "went"], "right": ["the", "store"],
    "phones": ["dh", "ax", "ah", "m"], "dur": 0.5,
    "units": [{"s": 0, "e": 2, "label": "REPETITION", "target": "COPY_R1"},
              {"s": 2, "e": 4, "label": "FILLER_HESITATION", "target": "FIL:um"}],
}
WSPAN = {
    "id": "t:w0", "sid": "t", "kind": "wspan", "left": ["see"], "right": ["now"], "word": "space",
    "phones": ["s", "p", "l", "ey", "s"], "dur": 0.4, "zb": 2,
    "units": [{"s": 0, "e": 5, "label": "MISPRONUNCIATION", "target": "SELF"}],
}


def decode(tok, ids):
    return tok.convert_ids_to_tokens(ids)


def test_gap_layout_tags_and_spans(tok):
    ds = GapDataset([GAP], tok, VOCAB, noise=None)
    item = ds[0]
    toks = decode(tok, item["ids"])
    assert toks[0] == "[CLS]" and toks[-1] == "[SEP]"
    gi = item["ctx_pos"]
    assert toks[gi] == "[GAP]"
    (s0, e0), (s1, e1) = item["spans"]
    assert toks[s0:e0] == ["<ph:dh>", "<ph:ax>"] and toks[s1:e1] == ["<ph:ah>", "<ph:m>"]
    assert [TAGS[item["tags"][k]] for k in range(s0, e1)] == [
        "B-REPETITION", "I-REPETITION", "B-FILLER_HESITATION", "I-FILLER_HESITATION"]
    assert item["targets"] == [VOCAB.index("COPY_R1"), VOCAB.index("FIL:um")]
    # everything that is not a phone carries no tag
    assert all(t == -100 for k, t in enumerate(item["tags"]) if not (s0 <= k < e1))
    assert toks[e1] == "[/GAP]"


def test_wspan_marker_index_and_word(tok):
    item = GapDataset([WSPAN], tok, VOCAB, noise=None)[0]
    toks = decode(tok, item["ids"])
    assert toks[item["ctx_pos"]] == "[WSPAN]"
    assert "space" in toks and "[Z2]" in toks
    s, e = item["spans"][0]
    assert toks[s:e] == ["<ph:s>", "<ph:p>", "<ph:l>", "<ph:ey>", "<ph:s>"]
    assert TAGS[item["tags"][s]] == "B-MISPRONUNCIATION"


def test_collate_shapes_and_unit_indices(tok):
    ds = GapDataset([GAP, WSPAN], tok, VOCAB, noise=None)
    b = collate([ds[0], ds[1]])
    B, T = b["input_ids"].shape
    assert B == 2 and b["attention_mask"].sum() == len(ds[0]["ids"]) + len(ds[1]["ids"])
    assert b["unit_batch"].tolist() == [0, 0, 1]
    assert b["unit_start"].shape == b["unit_end"].shape == b["unit_target"].shape == (3,)
    assert (b["unit_end"] <= T).all()


def test_noise_keeps_unit_spans_valid(tok):
    from phones.noise import PhoneNoise
    ds = GapDataset([GAP] * 20, tok, VOCAB, noise=PhoneNoise(p_del=0.4, p_sub=0.3, p_ins=0.3, seed=3))
    for i in range(20):
        item = ds[i]
        assert len(item["spans"]) == 2 and all(e > s for s, e in item["spans"])
        assert item["spans"][0][1] == item["spans"][1][0]      # still contiguous


def test_tag_inventory_is_consistent():
    assert TAGS[0] == "O" and len(TAGS) == len(TAG2ID)
    assert "B-PARTIAL_WORD" in TAG2ID and "I-FALSE_START" in TAG2ID and "B-NOISE" in TAG2ID
