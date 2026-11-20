"""Sequence-model data (windows of transcript + G2P + recognized phones), encoding, decoding and metrics."""
import random

import pytest
import torch

from model.dataset import IGNORE
from model.seq import (
    K_UNITS, NONE, UNK, SeqDataset, build_seq_side, collate_seq, count_units, decode_slot, make_unit_vocab,
    slot_prf, split_windows, unit_label, word_cost,
)
from phones.noise import PhoneNoise
from collections import Counter

PH = {"so": ["s", "ow"], "um": ["ah", "m"], "the": ["dh", "ax"], "cat": ["k", "ae", "t"], "sat": ["s", "ae", "t"],
      "on": ["aa", "n"], "mat": ["m", "ae", "t"], "went": ["w", "eh", "n", "t"], "hello": ["hh", "ax", "l", "ow"]}


class Lex:
    def get(self, w):
        return [PH[w]] if w in PH else []


def tok(i, spoken, label="O", kind="WORD", intended=None):
    return {"i": i, "spoken": spoken, "intended": spoken if intended is None else intended, "label": label,
            "kind": kind, "phones": PH.get(spoken, ["ax"])}


def side(*toks):
    return {"sid": "swX", "tokens": list(toks)}


def build(s, **kw):
    kw.setdefault("sizes", (20, 20))
    return list(build_seq_side(s, Lex(), random.Random(0), PhoneNoise(seed=0), **kw))


# ------------------------------------------------------------------ builder

def test_filler_and_repetition_become_slot_units_and_stay_in_the_recognized_phones():
    s = side(tok(0, "so"), tok(1, "um", "FILLER_HESITATION"), tok(2, "the", "REPETITION"), tok(3, "the"), tok(4, "cat"))
    (ex,) = build(s)
    assert ex["words"] == ["so", "the", "cat"]
    assert ex["exp"] == [PH["so"], PH["the"], PH["cat"]]
    assert ex["rec"] == PH["so"] + PH["um"] + PH["the"] + PH["the"] + PH["cat"]      # everything the speaker said
    assert ex["slots"] == [[], ["FILLER_HESITATION|FIL:um", "REPETITION|COPY_R1"], [], []]
    assert ex["wtags"] == [0, 0, 0]


def test_trailing_units_go_to_the_end_slot_of_the_last_window():
    (ex,) = build(side(tok(0, "so"), tok(1, "um", "FILLER_HESITATION")))
    assert ex["slots"] == [[], ["FILLER_HESITATION|FIL:um"]] and ex["rec"] == PH["so"] + PH["um"]


def test_prolonged_and_mispronounced_words_stay_in_the_transcript_and_get_word_tags():
    mis = tok(1, "cet", "MISPRONUNCIATION", intended="cat")
    mis["phones"] = ["s", "eh", "t"]
    (ex,) = build(side(tok(0, "so", "PROLONGATION"), mis))
    assert ex["words"] == ["so", "cat"] and ex["wtags"] == [1, 2]
    assert ex["exp"][1] == PH["cat"] and ex["rec"][2:] == ["s", "eh", "t"]            # G2P of the intended word, audio of the spoken one


def test_keep_p_keeps_disfluent_tokens_in_the_transcript():
    s = side(tok(0, "so"), tok(1, "um", "FILLER_HESITATION"), tok(2, "cat"))
    (ex,) = build(s, keep_p=1.0)
    assert ex["words"] == ["so", "um", "cat"] and ex["slots"] == [[], [], [], []]
    (ex,) = build(side(tok(0, "so"), tok(1, "uh", "PARTIAL_WORD", kind="PARTIAL", intended="u"), tok(2, "cat")), keep_p=1.0)
    assert ex["words"] == ["so", "cat"]                                                 # partial words are never transcribed


def test_noise_tokens_become_noise_drop_units_with_spurious_phones():
    (ex,) = build(side(tok(0, "so"), tok(1, "[noise]", "NOISE", kind="NOISE"), tok(2, "cat")), noise_p=1.0)
    assert ex["slots"][1] == ["NOISE|DROP"] and len(ex["rec"]) > len(PH["so"] + PH["cat"])


def test_windows_cut_slots_and_hand_the_boundary_gap_to_the_next_window():
    toks = [tok(0, "so"), tok(1, "cat"), tok(2, "sat"), tok(3, "um", "FILLER_HESITATION"), tok(4, "on"), tok(5, "the"), tok(6, "mat")]
    a, b = build(side(*toks), sizes=(3, 3))
    assert (a["words"], b["words"]) == (["so", "cat", "sat"], ["on", "the", "mat"])
    assert a["slots"][-1] is None and a["rec"] == PH["so"] + PH["cat"] + PH["sat"]    # not this window's: it follows the last word
    assert b["slots"][0] == ["FILLER_HESITATION|FIL:um"] and b["rec"][:2] == PH["um"]  # ... but the next window's slot 0
    assert b["slots"][-1] == []                                                         # the last window predicts its end slot


def test_copy_pointers_only_point_at_words_inside_the_window():
    # "sat" is repeated across the window boundary: the model never sees the first "sat", so it must be W:sat
    toks = [tok(0, "so"), tok(1, "cat"), tok(2, "sat"), tok(3, "sat", "REPETITION"), tok(4, "on"), tok(5, "mat")]
    a, b = build(side(*toks), sizes=(3, 3))
    assert b["slots"][0] == ["REPETITION|W:sat"]
    # inside one window the same repetition is a pointer to the next word
    (c,) = build(side(tok(0, "so"), tok(1, "cat", "REPETITION"), tok(2, "cat"), tok(3, "sat")))
    assert c["slots"][1] == ["REPETITION|COPY_R1"]
    # a repetition trailing the last word copies to the left
    (d,) = build(side(tok(0, "so"), tok(1, "cat"), tok(2, "cat", "REPETITION")))
    assert d["slots"][-1] == ["REPETITION|COPY_L1"]


def test_window_budget_limits_long_words():
    costs = [word_cost(8)] * 30
    sizes = [b - a for a, b in split_windows(costs, lambda: 24, budget=100.0)]
    assert max(sizes) * word_cost(8) <= 100.0 and sum(sizes) == 30
    assert [b - a for a, b in split_windows([word_cost(30)], lambda: 5, budget=10.0)] == [1]   # never an empty window


def test_unit_vocab_has_none_unk_and_label_unk_first():
    c = Counter({"REPETITION|COPY_R1": 5, "FILLER_DM|FIL:so": 3, "FILLER_DM|FIL:well": 1})
    v = make_unit_vocab(c, top_k=2)
    assert v[:2] == [NONE, UNK] and "REPETITION|UNK" in v and "FILLER_DM|UNK" in v
    assert v[-2:] == ["REPETITION|COPY_R1", "FILLER_DM|FIL:so"]
    ex = {"slots": [["REPETITION|COPY_R1"], None, []]}
    c2 = Counter(); count_units(ex, c2)
    assert c2 == {"REPETITION|COPY_R1": 1}


# ------------------------------------------------------------------ dataset

class FakeTok:
    cls_token_id, sep_token_id, unk_token_id = 1, 2, 3

    def __init__(self):
        self.ids = {}

    def convert_tokens_to_ids(self, t):
        return self.ids.setdefault(t, 100 + len(self.ids))

    def encode(self, w, add_special_tokens=False):
        return [10 + (hash(w) % 50)] * (2 if len(w) > 4 else 1)


VOCAB = [NONE, UNK, "FILLER_HESITATION|UNK", "REPETITION|UNK", "FILLER_HESITATION|FIL:um", "REPETITION|COPY_R1"]


def example(**kw):
    ex = {"words": ["so", "hello", "cat"], "exp": [PH["so"], PH["hello"], PH["cat"]], "rec": ["s", "ow", "ah", "m"],
          "slots": [[], ["FILLER_HESITATION|FIL:um"], [], None], "wtags": [0, 1, 0]}
    ex.update(kw)
    return ex


def test_encoding_layout_positions_and_labels():
    ds = SeqDataset([example()], FakeTok(), VOCAB)
    it = ds[0]
    ids, t = it["ids"], FakeTok()
    # [CLS] so <s><ow> hello(2 pieces) <ph>x4 cat <k><ae><t> [END] [SEP] rec... [SEP]
    assert it["word_pos"] == [1, 4, 10] and it["slot_pos"] == [1, 4, 10, 14]
    assert len(ids) == 1 + (1 + 2) + (2 + 4) + (1 + 3) + 2 + 4 + 1
    assert ids[0] == 1 and ids[-1] == 2 and ids[15] == 2                               # [END] at 14, [SEP] at 15
    assert it["types"] == [0] * 16 + [1] * 5
    assert it["slot_lab"] == [[0] * K_UNITS, [4, 0, 0, 0], [0] * K_UNITS, [IGNORE] * K_UNITS]
    assert it["word_tag"] == [0, 1, 0]


def test_unknown_units_fall_back_to_label_unk_and_extra_units_are_truncated():
    ds = SeqDataset([example(slots=[["REPETITION|W:zebra"], ["FILLER_HESITATION|FIL:oh"] + ["REPETITION|COPY_R1"] * 5, [], []])],
                    FakeTok(), VOCAB)
    lab = ds[0]["slot_lab"]
    assert lab[0][0] == VOCAB.index("REPETITION|UNK") and lab[1][0] == VOCAB.index("FILLER_HESITATION|UNK")
    assert lab[1][1:] == [VOCAB.index("REPETITION|COPY_R1")] * (K_UNITS - 1)


def test_noise_is_applied_to_recognized_phones_only_and_truncation_is_counted():
    ex = example(rec=["s", "ow", "ah", "m"] * 3)
    ds = SeqDataset([ex], FakeTok(), VOCAB, noise=PhoneNoise(p_del=1.0, p_sub=0.0, p_ins=0.0, seed=0))
    assert len(ds[0]["ids"]) < len(SeqDataset([ex], FakeTok(), VOCAB)[0]["ids"])      # the phones were corrupted
    short = SeqDataset([ex], FakeTok(), VOCAB, max_len=22)
    assert len(short[0]["ids"]) == 22 and short.truncated == 1


def test_collate_pads_and_flattens():
    ds = SeqDataset([example(), example(words=["cat"], exp=[PH["cat"]], rec=["k"], slots=[[], []], wtags=[0])], FakeTok(), VOCAB)
    b = collate_seq([ds[0], ds[1]])
    assert b["input_ids"].shape[0] == 2 and b["attention_mask"][1].sum() < b["attention_mask"][0].sum()
    assert b["slot_batch"].tolist() == [0, 0, 0, 0, 1, 1] and b["slot_lab"].shape == (6, K_UNITS)
    assert b["word_batch"].tolist() == [0, 0, 0, 1] and b["word_tag"].tolist() == [0, 1, 0, 0]


# ------------------------------------------------------------------ decoding / metrics

def test_decode_slot_stops_at_none_and_none_bias_restores_more():
    v = [NONE, "A", "B"]
    logits = torch.tensor([[0.0, 3.0, 0.0], [0.0, 0.0, 2.0], [5.0, 0.0, 0.0], [0.0, 9.0, 0.0]])
    assert decode_slot(logits, v) == ["A", "B"]                                         # stops at the third head's <none>
    quiet = torch.tensor([[2.0, 1.5, 0.0]] + [[5.0, 0.0, 0.0]] * 3)
    assert decode_slot(quiet, v) == [] and decode_slot(quiet, v, none_bias=1.0) == ["A"]


def test_slot_prf_types_exact_and_slot_level():
    gold = [[[], ["FILLER_HESITATION|FIL:um", "REPETITION|COPY_R1"], None, ["FILLER_DM|FIL:so"]]]
    pred = [[["REPETITION|COPY_R1"], ["REPETITION|COPY_R2", "FILLER_HESITATION|FIL:um"], ["FILLER_DM|FIL:x"], []]]
    r = slot_prf(gold, pred)
    assert r["per_type"]["FILLER_HESITATION"]["f1"] == 1.0
    assert r["per_type"]["REPETITION"] == {"support": 1, "predicted": 2, "p": 0.5, "r": 1.0, "f1": 0.6667}
    assert r["per_type"]["FILLER_DM"]["r"] == 0.0 and r["per_type"]["FILLER_DM"]["support"] == 1
    assert r["exact"]["tp"] == 1 and r["exact"]["r"] == pytest.approx(1 / 3, abs=1e-3)   # only "FIL:um" matches exactly
    assert r["slot"]["p"] == pytest.approx(0.5) and r["slot"]["r"] == pytest.approx(0.5)   # the None slot is skipped
    assert r["restoration_acc_on_type_matches"] == pytest.approx(0.5)                   # 2 type matches (um, rep): 1 exact
    assert unit_label("REPETITION|COPY_R1") == "REPETITION"


# ------------------------------------------------------------------ model

def test_bert_seq_forward_backward_with_a_tiny_encoder(monkeypatch):
    from transformers import BertConfig, BertModel

    import model.bert_seq as bs
    cfg = BertConfig(vocab_size=200, hidden_size=32, num_hidden_layers=1, num_attention_heads=2, intermediate_size=64,
                     max_position_embeddings=64)
    monkeypatch.setattr(bs.AutoModel, "from_pretrained", lambda name: BertModel(cfg))
    m = bs.BertSeq("tiny", 220, len(VOCAB))
    ds = SeqDataset([example(), example(slots=[["REPETITION|COPY_R1"], [], [], []])], FakeTok(), VOCAB)
    batch = collate_seq([ds[0], ds[1]])
    out = m(batch)
    assert out["slot_logits"].shape == (8, K_UNITS, len(VOCAB)) and out["word_logits"].shape == (6, 3)
    out["loss"].backward()
    assert m.slot_head[-1].weight.grad is not None and torch.isfinite(out["loss"])
    only_ignored = {**batch, "slot_lab": torch.full_like(batch["slot_lab"], IGNORE)}
    assert float(m(only_ignored)["slot_loss"].detach()) == 0.0                                   # no NaN when nothing is labelled


# ------------------------------------------------------------------ auxiliary per-phone labels

def test_builder_labels_every_recognized_phone_with_a_role_and_a_word_pointer():
    from model.seq import ROLES
    s = side(tok(0, "so"), tok(1, "um", "FILLER_HESITATION"), tok(2, "the", "REPETITION"), tok(3, "the"), tok(4, "cat"),
             tok(5, "um", "FILLER_HESITATION"))
    (ex,) = build(s)
    fh, rep = ROLES.index("FILLER_HESITATION"), ROLES.index("REPETITION")
    assert ex["rec"] == PH["so"] + PH["um"] + PH["the"] + PH["the"] + PH["cat"] + PH["um"]
    assert ex["rec_role"] == [0, 0, fh, fh, rep, rep, 0, 0, 0, 0, 0, fh, fh]
    assert ex["rec_ptr"] == [0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 3, 3]          # "so" -> 0; um, the(rep), the -> 1; cat -> 2; trailing um -> 3 (= n)
    assert len(ex["rec_role"]) == len(ex["rec_ptr"]) == len(ex["rec"])


def test_noise_labels_for_spurious_phones_and_non_final_windows_have_no_trailing_phones():
    from model.seq import ROLES
    toks = [tok(0, "so"), tok(1, "[noise]", "NOISE", kind="NOISE"), tok(2, "cat"), tok(3, "sat"), tok(4, "on")]
    a, b = build(side(*toks), sizes=(2, 2), noise_p=1.0)
    n = ROLES.index("NOISE")
    assert n in a["rec_role"] and a["rec_ptr"][a["rec_role"].index(n)] == 1       # the junk precedes word 1 ("cat")
    assert max(a["rec_ptr"]) <= len(a["words"]) - 1                                  # nothing points at the end slot of a non-final window


def test_noise_corrupt_idx_tracks_the_source_phone_and_keeps_the_random_stream():
    ph = ["a", "b", "c"]
    out, src = PhoneNoise(p_del=0.0, p_sub=0.0, p_ins=1.0, seed=1).corrupt_idx(ph)
    assert len(out) == 6 and src == [0, 0, 1, 1, 2, 2]
    out, src = PhoneNoise(p_del=1.0, p_sub=0.0, p_ins=0.0, seed=1).corrupt_idx(ph)
    assert src == [0] and len(out) == 1                                              # never empty
    n1, n2 = PhoneNoise(seed=7), PhoneNoise(seed=7)
    assert n1.corrupt(ph * 5) == n2.corrupt_idx(ph * 5)[0]                           # corrupt() == corrupt_idx()[0]


def test_dataset_aux_labels_follow_the_noise_and_the_truncation():
    ex = example(rec=["s", "ow", "ah", "m"], rec_role=[0, 0, 1, 1], rec_ptr=[0, 0, 1, 1])
    plain = SeqDataset([ex], FakeTok(), VOCAB)[0]
    assert plain["rec_role"] == [0, 0, 1, 1] and plain["rec_ptr"] == [0, 0, 1, 1]
    n_a = plain["types"].index(1)
    assert plain["rec_pos"] == list(range(n_a, n_a + 4))                             # positions of the recognized phones
    noisy = SeqDataset([ex], FakeTok(), VOCAB, noise=PhoneNoise(p_del=0.0, p_sub=0.0, p_ins=1.0, seed=0))[0]
    assert noisy["rec_role"] == [0, 0, 0, 0, 1, 1, 1, 1] and noisy["rec_ptr"] == [0, 0, 0, 0, 1, 1, 1, 1]
    cut = SeqDataset([ex], FakeTok(), VOCAB, max_len=n_a + 3)[0]
    assert len(cut["rec_pos"]) == len(cut["rec_role"]) == len(cut["rec_ptr"]) == 2    # truncated together with the phones
    assert SeqDataset([example()], FakeTok(), VOCAB)[0]["rec_pos"] == []             # inference examples carry no aux labels


def test_aux_heads_train_and_old_checkpoints_load(monkeypatch, tmp_path):
    from transformers import BertConfig, BertModel

    import model.bert_seq as bs
    from model.seq import MAX_PTR, ROLES
    cfg = BertConfig(vocab_size=200, hidden_size=32, num_hidden_layers=1, num_attention_heads=2, intermediate_size=64,
                     max_position_embeddings=64)
    monkeypatch.setattr(bs.AutoModel, "from_pretrained", lambda name: BertModel(cfg))
    m = bs.BertSeq("tiny", 220, len(VOCAB))
    ds = SeqDataset([example(rec_role=[0, 0, 1, 1], rec_ptr=[0, 0, 1, 1])], FakeTok(), VOCAB)
    out = m(collate_seq([ds[0]]))
    assert torch.isfinite(out["role_loss"]) and torch.isfinite(out["ptr_loss"])
    assert m.role_head.out_features == len(ROLES) and m.ptr_head.out_features == MAX_PTR
    out["loss"].backward()
    assert m.role_head.weight.grad is not None and m.ptr_head.weight.grad is not None
    # a checkpoint without the auxiliary heads (trained before they existed) still loads for inference
    old = {k: v for k, v in m.state_dict().items() if not k.startswith(("role_head", "ptr_head"))}
    ck = tmp_path / "ck"; (ck / "tokenizer").mkdir(parents=True)
    torch.save(old, ck / "model.pt")
    (ck / "unit_vocab.json").write_text(__import__("json").dumps(VOCAB))
    monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained", lambda p: FakeSizedTok(220))
    monkeypatch.setattr("model.infer.pick_device", lambda: torch.device("cpu"))
    model2, _, vocab2, _ = bs.load_seq_model(ck, bert="tiny")
    assert vocab2 == VOCAB and not model2.training


class FakeSizedTok:
    def __init__(self, n):
        self.n = n

    def __len__(self):
        return self.n


def test_dataset_rejects_aux_labels_that_do_not_match_the_recognized_phones():
    ex = example(rec=["s", "ow", "ah", "m"], rec_role=[0, 0], rec_ptr=[0, 0])
    with pytest.raises(AssertionError, match="out of sync"):
        SeqDataset([ex], FakeTok(), VOCAB)[0]
