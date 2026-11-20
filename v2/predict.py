#!/usr/bin/env python3
"""Restore a short flat transcript using its complete spoken-phone stream.

Example:
  .venv/bin/python v2/predict.py --text 'the cat' --phones 'ah m dh ax k ae t'

Input phones must include the sounds of the missing disfluencies. G2P of the
flat text alone contains no such evidence. This model expects MS98 phone IDs.
"""
from __future__ import annotations

import argparse
import json

import torch
from transformers import AutoTokenizer

from restoration import MAX_WORDS, ROOT, TYPES, WTAGS, RestorationBert, RestorationDataset, collate, restore


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=str(ROOT / "v2/checkpoints/bert_compact"))
    ap.add_argument("--weights", default="model.pt", help="model.pt or model_f1_decoder.pt")
    ap.add_argument("--text", required=True, help="flat text, at most 16 whitespace-separated words")
    ap.add_argument("--phones", required=True, help="complete space-separated MS98 phone sequence")
    a = ap.parse_args()
    from pathlib import Path
    path = Path(a.checkpoint)
    args = json.loads((path / "args.json").read_text())
    targets = json.loads((path / "target_vocab.json").read_text())
    tok = AutoTokenizer.from_pretrained(path / "tokenizer", local_files_only=True)
    if args.get("architecture") == "pretrained":
        model = RestorationBert(len(tok), len(targets), encoder_config=json.loads((path / "bert_config.json").read_text()))
    else:
        model = RestorationBert(len(tok), len(targets), hidden=args["hidden"], layers=args["layers"])
    model.load_state_dict(torch.load(path / a.weights, map_location="cpu", weights_only=True))
    model.eval()
    words, phones = a.text.lower().split(), a.phones.lower().split()
    if not words or len(words) > MAX_WORDS:
        ap.error("Provide between 1 and 16 flat words with the corresponding full phone stream.")
    ex = dict(id="input", words=words, phones=phones, slots=[[] for _ in range(len(words)+1)],
              roles=[0]*len(phones), pointers=[0]*len(phones), wtags=[0]*len(words))
    if args.get("pronunciation_guidance"):
        pronunciations = json.loads((path / "pronunciations.json").read_text())
        ex["expected"] = [pronunciations.get(w, []) for w in words]
    batch = collate([RestorationDataset([ex], tok, targets)[0]])
    with torch.no_grad():
        result = model(batch, loss=False)
    slots = []
    for types, target_ids in zip(result["predicted_types"].tolist(), result["predicted_targets"].tolist()):
        units = []
        for typ, target in zip(types, target_ids):
            if not typ:
                break
            units.append(TYPES[typ] + "|" + targets[target])
        slots.append(units)
    print(json.dumps(dict(flat_transcript=" ".join(words), restored_transcript=" ".join(restore(words, slots)),
                          insertions=slots, word_tags=[WTAGS[t] for t in result["word_logits"].argmax(-1).tolist()]), indent=2))


if __name__ == "__main__":
    main()
