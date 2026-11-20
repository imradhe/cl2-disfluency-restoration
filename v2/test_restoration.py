"""Checks for leakage, surface restoration, and complete-reference scoring."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from restoration import K, RestorationBert, RestorationDataset, collate, restore
from train import evaluate


class Tokenizer:
    cls_token_id, sep_token_id, unk_token_id = 1, 2, 3

    def convert_tokens_to_ids(self, token):
        return 4 if token == "[END]" else 5

    def encode(self, word, add_special_tokens=False):
        return [6]


def example(units=None):
    units = units if units is not None else ["FILLER_HESITATION|FIL:um"]
    return dict(id="test", sid="sw0001A", conv="0001", words=["the", "cat"],
                phones=["ah", "m", "dh", "ax", "k", "ae", "t"],
                slots=[units, [], []], reference=["um"]*len(units) + ["the", "cat"],
                wtags=[0, 0], roles=[1, 1, 0, 0, 0, 0, 0], pointers=[0]*4+[1]*3)


def test_labels_and_gold_alignments_never_enter_encoder_inputs():
    tok = Tokenizer()
    a = example()
    b = {**a, "slots": [[], [], []], "roles": [7]*7, "pointers": [2]*7, "wtags": [1, 2]}
    ds = RestorationDataset([a, b], tok, ["UNK", "FIL:um"])
    first, second = ds[0], ds[1]
    assert first["ids"] == second["ids"] and first["segments"] == second["segments"]
    model = RestorationBert(32, 2, hidden=16, layers=1).eval()
    with torch.no_grad():
        p = model(collate([first]), loss=False)
        q = model(collate([second]), loss=False)
    torch.testing.assert_close(p["type_logits"], q["type_logits"])
    assert torch.isfinite(model(collate([first]))["loss"])
    model(collate([first]))["loss"].backward()
    assert model.bert.embeddings.word_embeddings.weight.grad is not None


def test_partial_words_are_not_completed_and_copy_pointers_render_in_context():
    slots = [["PARTIAL_WORD|W:th-", "REPETITION|COPY_R1"], [], []]
    assert restore(["the", "cat"], slots) == ["th-", "the", "the", "cat"]


def test_pronunciation_guidance_uses_flat_words_and_relocates_slot_positions():
    a = example()
    a["expected"] = [["dh", "ax"], ["k", "ae", "t"]]
    item = RestorationDataset([a], Tokenizer(), ["UNK", "FIL:um"])[0]
    assert item["word_pos"] == [1, 4]
    assert item["slot_pos"] == [1, 4, 8]
    assert item["ids"][:10] == [1, 6, 5, 5, 6, 5, 5, 5, 4, 2]


def test_pretrained_encoder_is_loaded_and_sets_head_dimensions(monkeypatch):
    from transformers import BertConfig, BertModel
    called = []

    def load(name, **kwargs):
        called.append(name)
        return BertModel(BertConfig(vocab_size=24, hidden_size=32, num_hidden_layers=2,
                                    num_attention_heads=4, intermediate_size=64), add_pooling_layer=False)

    monkeypatch.setattr(BertModel, "from_pretrained", load)
    model = RestorationBert(40, 2, pretrained="pretrained-test")
    assert called == ["pretrained-test"]
    assert model.bert.config.vocab_size == 40
    assert model.type_head.in_features == 32
    assert model.bert.config.num_hidden_layers == 2


def test_full_gold_overflow_is_counted_in_metrics(tmp_path):
    class EmptyModel:
        def eval(self):
            return self

        def __call__(self, batch, **kwargs):
            return dict(predicted_types=torch.zeros(len(batch["slot_pos"]), K, dtype=torch.long),
                        predicted_targets=torch.zeros(len(batch["slot_pos"]), K, dtype=torch.long),
                        word_logits=torch.tensor([[1., 0., 0.], [1., 0., 0.]]))

    m = evaluate(EmptyModel(), [example(["FILLER_HESITATION|FIL:um"]*(K+2))],
                 Tokenizer(), ["UNK", "FIL:um"], torch.device("cpu"))
    assert m["typed_restored_token"]["support"] == K+2
    assert m["overflow_gold_units_counted_as_misses"] == 2
    assert m["flat_edit_errors"] == K+2
    assert m["restored_wer"] == m["flat_wer"]


def test_downloaded_prediction_scoring_ignores_boundary_slots(tmp_path):
    import gzip
    import json
    from install_colab import counts_by_conversation, micro_f1
    path = tmp_path / "predictions.jsonl.gz"
    row = dict(id="sw4010A:v2:0", flat_transcript="the cat",
               gold_slots=[["FILLER_HESITATION|FIL:um"], None, []],
               predicted_slots=[["FILLER_HESITATION|FIL:um", "REPETITION|COPY_R1"],
                                ["FILLER_HESITATION|FIL:um"], []])
    with gzip.open(path, "wt") as f:
        f.write(json.dumps(row)+"\n")
    counts, n = counts_by_conversation(path)
    assert n == 1 and counts["4010"].tolist() == [1, 2, 1]
    assert float(micro_f1(counts["4010"])) == 2/3
