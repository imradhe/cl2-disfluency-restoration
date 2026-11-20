"""BERT classifier over (fluent transcript + G2P phones + recognized phones).

Heads read the hidden state of each word's first piece (and ``[END]``): ``K_UNITS`` classifiers pick the
disfluency units inserted in front of that position (class 0 = nothing), and one more tags the word itself
(fluent / prolonged / mispronounced). Two auxiliary heads read the recognized-phone positions (training only):
the role of each phone (explained by the transcript / which disfluency) and the word it belongs to or precedes.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModel

from .dataset import IGNORE
from .seq import K_UNITS, MAX_PTR, ROLES, WTAGS


class BertSeq(nn.Module):
    def __init__(self, bert_name: str, vocab_size: int, n_units: int, k_units: int = K_UNITS, aux_weight: float = 0.5) -> None:
        super().__init__()
        self.bert = AutoModel.from_pretrained(bert_name)
        self.bert.resize_token_embeddings(vocab_size)
        h = self.bert.config.hidden_size
        self.k, self.n_units, self.aux_weight = k_units, n_units, aux_weight
        self.drop = nn.Dropout(0.1)
        self.slot_head = nn.Sequential(nn.Linear(h, h), nn.GELU(), nn.Dropout(0.1), nn.Linear(h, k_units * n_units))
        self.word_head = nn.Linear(h, len(WTAGS))
        self.role_head = nn.Linear(h, len(ROLES))
        self.ptr_head = nn.Linear(h, MAX_PTR)

    def forward(self, batch: dict, compute_loss: bool = True) -> dict:
        h = self.drop(self.bert(
            input_ids=batch["input_ids"], token_type_ids=batch["token_type_ids"],
            attention_mask=batch["attention_mask"]).last_hidden_state)                      # [B, T, H]
        slot_logits = self.slot_head(h[batch["slot_batch"], batch["slot_pos"]]).view(-1, self.k, self.n_units)
        word_logits = self.word_head(h[batch["word_batch"], batch["word_pos"]])
        res = {"slot_logits": slot_logits, "word_logits": word_logits}
        if compute_loss:
            lab = batch["slot_lab"]
            slot_loss = nn.functional.cross_entropy(slot_logits.reshape(-1, self.n_units).float(), lab.reshape(-1),
                                                    ignore_index=IGNORE) if bool((lab != IGNORE).any()) else h.sum() * 0
            word_loss = nn.functional.cross_entropy(word_logits.float(), batch["word_tag"]) if len(word_logits) else h.sum() * 0
            n_rec = len(batch["rec_pos"])
            if n_rec:
                hr = h[batch["rec_batch"], batch["rec_pos"]]
                role_loss = nn.functional.cross_entropy(self.role_head(hr).float(), batch["rec_role"])
                ptr_loss = nn.functional.cross_entropy(self.ptr_head(hr).float(), batch["rec_ptr"])
            else:
                role_loss = ptr_loss = h.sum() * 0
            res.update(loss=slot_loss + 0.5 * word_loss + self.aux_weight * (role_loss + ptr_loss),
                       slot_loss=slot_loss, word_loss=word_loss, role_loss=role_loss, ptr_loss=ptr_loss)
        return res


def load_seq_model(ckpt, bert: str = "bert-base-uncased", dev: torch.device | None = None):
    """Load a checkpoint dir written by ``model.train_seq --save`` -> ``(model, tokenizer, unit_vocab, device)``."""
    import json
    from pathlib import Path

    from transformers import AutoTokenizer

    from .infer import pick_device

    ckpt, dev = Path(ckpt), dev or pick_device()
    tok = AutoTokenizer.from_pretrained(ckpt / "tokenizer")
    vocab = json.loads((ckpt / "unit_vocab.json").read_text())
    model = BertSeq(bert, len(tok), len(vocab)).to(dev)
    missing, unexpected = model.load_state_dict(torch.load(ckpt / "model.pt", map_location=dev), strict=False)
    # checkpoints trained before the auxiliary heads existed lack them; they are not used at inference
    if unexpected or any(not k.startswith(("role_head", "ptr_head")) for k in missing):
        raise RuntimeError(f"checkpoint does not match the model: missing={missing} unexpected={unexpected}")
    model.eval()
    return model, tok, vocab, dev
