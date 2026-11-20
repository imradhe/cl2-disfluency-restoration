"""Compact BERT insertion model for flat text + a full, unsegmented phone stream.

All supervised labels are excluded from the model input. Phone roles and word
pointers supply auxiliary losses during training only. Surface prediction is
factored from type prediction to avoid a large vocabulary softmax on every
empty insertion position.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import torch
from torch import nn
from transformers import BertConfig, BertModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from phones.noise import PhoneNoise  # noqa: E402
from phones.lexicon import SWB_PHONES  # noqa: E402

TYPES = ["NONE", "FILLER_HESITATION", "FILLER_BACKCHANNEL", "FILLER_DM",
         "REPETITION", "REPAIR", "FALSE_START", "PARTIAL_WORD"]
TYPE_ID = {name: i for i, name in enumerate(TYPES)}
WTAGS = ["O", "PROLONGATION", "MISPRONUNCIATION"]
K = 8
MAX_WORDS = 16
MAX_LENGTH = 256
SPECIAL = ["[END]"] + [f"<ph:{p}>" for p in SWB_PHONES]


class RestorationDataset(torch.utils.data.Dataset):
    def __init__(self, examples, tokenizer, targets, noise_p=0.0, seed=13):
        self.examples, self.tokenizer = examples, tokenizer
        self.target_ids = {v: i for i, v in enumerate(targets)}
        self.noise_p, self.rng = noise_p, random.Random(seed)
        self.noise = PhoneNoise(seed=seed)
        self.phone_ids = {p: tokenizer.convert_tokens_to_ids(f"<ph:{p}>") for p in SWB_PHONES}
        self.end_id = tokenizer.convert_tokens_to_ids("[END]")
        self.word_cache = {}

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        ex = self.examples[index]
        ids, word_pos = [self.tokenizer.cls_token_id], []
        for word in ex["words"]:
            word_pos.append(len(ids))
            if word not in self.word_cache:
                self.word_cache[word] = self.tokenizer.encode(word, add_special_tokens=False) or [self.tokenizer.unk_token_id]
            ids.extend(self.word_cache[word])
            # Optional pronunciation guidance derived only from the flat words.
            # No gold phone boundary or disfluency label is used here.
            if "expected" in ex:
                ids.extend(self.phone_ids[p] for p in ex["expected"][len(word_pos)-1])
        slot_pos = word_pos + [len(ids)]
        ids += [self.end_id, self.tokenizer.sep_token_id]
        n_text = len(ids)
        phones, src = ex["phones"], list(range(len(ex["phones"])))
        if self.rng.random() < self.noise_p:
            phones, src = self.noise.corrupt_idx(phones)
        limit = 512 if "expected" in ex else MAX_LENGTH
        if n_text + len(phones) + 1 > limit:
            raise ValueError(f"Window too long; refusing to truncate: {ex['id']}")
        phone_pos = list(range(n_text, n_text + len(phones)))
        ids += [self.phone_ids[p] for p in phones] + [self.tokenizer.sep_token_id]
        labels, targets = [], []
        for units in ex["slots"]:
            if units is None:
                labels.append([-100] * K)
                targets.append([-100] * K)
                continue
            labs, tgts = [], []
            for unit in units[:K]:
                label, target = unit.split("|", 1)
                labs.append(TYPE_ID[label])
                tgts.append(self.target_ids.get(target, 0))
            if len(labs) < K:
                labs.append(0)
                tgts.append(-100)
            labels.append(labs + [-100] * (K - len(labs)))
            targets.append(tgts + [-100] * (K - len(tgts)))
        return dict(ids=ids, segments=[0]*n_text + [1]*(len(ids)-n_text),
                    word_pos=word_pos, slot_pos=slot_pos, labels=labels, targets=targets,
                    phone_pos=phone_pos, roles=[ex["roles"][i] for i in src],
                    pointers=[ex["pointers"][i] for i in src], wtags=ex["wtags"])


def collate(items):
    shape = (len(items), max(len(x["ids"]) for x in items))
    ids, segments, mask = (torch.zeros(shape, dtype=torch.long) for _ in range(3))
    values = {name: [] for name in ("slot_batch", "slot_pos", "labels", "targets",
                                   "word_batch", "word_pos", "wtags", "phone_batch", "phone_pos", "roles", "pointers")}
    for i, item in enumerate(items):
        n = len(item["ids"])
        ids[i, :n], segments[i, :n], mask[i, :n] = torch.tensor(item["ids"]), torch.tensor(item["segments"]), 1
        for group in ("slot", "word", "phone"):
            values[f"{group}_batch"].extend([i]*len(item[f"{group}_pos"]))
            values[f"{group}_pos"].extend(item[f"{group}_pos"])
        for name in ("labels", "targets", "wtags", "roles", "pointers"):
            values[name].extend(item[name])
    batch = {name: torch.tensor(value, dtype=torch.long) for name, value in values.items()}
    batch.update(input_ids=ids, token_type_ids=segments, attention_mask=mask)
    return batch


class RestorationBert(nn.Module):
    def __init__(self, vocab_size, n_targets, hidden=128, layers=2, pretrained=None, encoder_config=None):
        super().__init__()
        if pretrained is not None:
            self.bert = BertModel.from_pretrained(pretrained, add_pooling_layer=False)
            self.bert.resize_token_embeddings(vocab_size, mean_resizing=False)
        elif encoder_config is not None:
            self.bert = BertModel(BertConfig.from_dict(encoder_config), add_pooling_layer=False)
        else:
            self.bert = BertModel(BertConfig(vocab_size=vocab_size, hidden_size=hidden,
                                        num_hidden_layers=layers, num_attention_heads=4,
                                        intermediate_size=hidden*4, max_position_embeddings=MAX_LENGTH,
                                        hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.1),
                              add_pooling_layer=False)
        hidden = self.bert.config.hidden_size
        self.position = nn.Embedding(K, hidden)
        self.unit_projection = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.LayerNorm(hidden))
        self.type_head = nn.Linear(hidden, len(TYPES))
        self.type_embedding = nn.Embedding(len(TYPES), hidden)
        self.target_head = nn.Linear(hidden, n_targets)
        self.word_head = nn.Linear(hidden, len(WTAGS))
        self.role_head = nn.Linear(hidden, len(TYPES))
        self.pointer_head = nn.Linear(hidden, MAX_WORDS+1)
        self.register_buffer("type_weights", torch.tensor([0.35] + [1.0]*(len(TYPES)-1)))

    def forward(self, batch, loss=True, type_bias=0.0):
        # No label, gold alignment or pointer is passed into BERT.
        h = self.bert(input_ids=batch["input_ids"], token_type_ids=batch["token_type_ids"],
                      attention_mask=batch["attention_mask"]).last_hidden_state
        slot_h = h[batch["slot_batch"], batch["slot_pos"]]
        unit_h = self.unit_projection(slot_h[:, None, :] + self.position.weight[None, :, :])
        type_logits = self.type_head(unit_h)
        word_logits = self.word_head(h[batch["word_batch"], batch["word_pos"]])
        result = dict(type_logits=type_logits, word_logits=word_logits)
        if loss:
            gold = batch["labels"]
            active = gold > 0
            type_loss = nn.functional.cross_entropy(type_logits.reshape(-1, len(TYPES)), gold.reshape(-1), weight=self.type_weights)
            if active.any():
                target_h = unit_h[active] + self.type_embedding(gold[active])
                target_loss = nn.functional.cross_entropy(self.target_head(target_h), batch["targets"][active])
            else:
                target_loss = h.sum()*0
            phone_h = h[batch["phone_batch"], batch["phone_pos"]]
            role_loss = nn.functional.cross_entropy(self.role_head(phone_h), batch["roles"]) if len(phone_h) else h.sum()*0
            pointer_loss = nn.functional.cross_entropy(self.pointer_head(phone_h), batch["pointers"]) if len(phone_h) else h.sum()*0
            word_loss = nn.functional.cross_entropy(word_logits, batch["wtags"])
            result.update(loss=type_loss + target_loss + 0.5*(role_loss + pointer_loss) + 0.2*word_loss,
                          type_loss=type_loss, target_loss=target_loss, role_loss=role_loss, pointer_loss=pointer_loss)
        else:
            biased = type_logits.clone()
            biased[:, :, 0] -= type_bias
            predicted_types = biased.argmax(-1)
            # Only the prefix before the first NONE is decoded.
            active = (predicted_types > 0).cumprod(dim=1).bool()
            target_ids = torch.zeros_like(predicted_types)
            if active.any():
                target_ids[active] = self.target_head(unit_h[active] + self.type_embedding(predicted_types[active])).argmax(-1)
            result.update(predicted_types=predicted_types.masked_fill(~active, 0), predicted_targets=target_ids)
        return result


def render_target(target, left, right):
    if target.startswith(("FIL:", "W:")):
        return target.split(":", 1)[1]
    if target.startswith("COPY_R"):
        i = int(target[6:])-1
        return right[i] if i < len(right) else "<unk>"
    if target.startswith("COPY_L"):
        i = int(target[6:])
        return left[-i] if i <= len(left) else "<unk>"
    return "<unk>"


def restore(words, slots):
    output = []
    for i in range(len(words)+1):
        for unit in slots[i] or []:
            _, target = unit.split("|", 1)
            output.append(render_target(target, words[:i], words[i:]))
        if i < len(words):
            output.append(words[i])
    return output


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n")
