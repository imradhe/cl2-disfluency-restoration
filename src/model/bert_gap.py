"""BERT gap model: phone-level BIO type tagger + per-unit restoration classifier."""
from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModel

from .dataset import IGNORE


class BertGap(nn.Module):
    def __init__(self, bert_name: str, vocab_size: int, n_tags: int, n_rest: int,
                 rest_weight: float = 1.0) -> None:
        super().__init__()
        self.bert = AutoModel.from_pretrained(bert_name)
        self.bert.resize_token_embeddings(vocab_size)
        h = self.bert.config.hidden_size
        self.drop = nn.Dropout(0.1)
        self.tag_head = nn.Linear(h, n_tags)
        self.rest_head = nn.Sequential(nn.Linear(2 * h, h), nn.GELU(), nn.Dropout(0.1), nn.Linear(h, n_rest))
        self.rest_weight = rest_weight

    def forward(self, batch: dict, compute_loss: bool = True) -> dict:
        out = self.bert(
            input_ids=batch["input_ids"], token_type_ids=batch["token_type_ids"],
            attention_mask=batch["attention_mask"],
        ).last_hidden_state                                           # [B, T, H]
        h = self.drop(out)
        tag_logits = self.tag_head(h)                                  # [B, T, n_tags]

        # unit representation: mean over the unit's phone tokens + the gap/span marker state
        ub, us, ue = batch["unit_batch"], batch["unit_start"], batch["unit_end"]
        T = h.size(1)
        pos = torch.arange(T, device=h.device)[None, :]
        mask = ((pos >= us[:, None]) & (pos < ue[:, None])).float()    # [U, T]
        mask = mask / mask.sum(1, keepdim=True).clamp(min=1)
        unit_repr = torch.einsum("ut,uth->uh", mask, h[ub])            # [U, H]
        marker = h[ub, batch["ctx_pos"][ub]]                           # [U, H]
        rest_logits = self.rest_head(torch.cat([unit_repr, marker], dim=-1))

        res = {"tag_logits": tag_logits, "rest_logits": rest_logits}
        if compute_loss:
            ce = nn.CrossEntropyLoss(ignore_index=IGNORE)
            tag_loss = ce(tag_logits.reshape(-1, tag_logits.size(-1)), batch["tags"].reshape(-1))
            rest_loss = nn.functional.cross_entropy(rest_logits, batch["unit_target"]) if len(ub) else h.sum() * 0
            res.update(loss=tag_loss + self.rest_weight * rest_loss, tag_loss=tag_loss, rest_loss=rest_loss)
        return res
