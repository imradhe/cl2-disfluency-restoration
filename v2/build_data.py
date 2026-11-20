#!/usr/bin/env python3
"""Build fully flattened transcript/phone pairs with verbatim surface targets."""
from __future__ import annotations

import gzip
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

from transformers import AutoTokenizer

from restoration import K, MAX_LENGTH, MAX_WORDS, ROOT, SPECIAL, TYPE_ID, WTAGS, save_json
sys.path.insert(0, str(ROOT / "src"))
from model.gaps import restore_target, split_of  # noqa: E402

OUT = ROOT / "v2/data"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    # Reuse the exact existing vocabulary when available; fresh checkouts can
    # obtain the standard tokenizer without needing a trained v1 checkpoint.
    local_tokenizer = next((p for p in (OUT / "tokenizer", ROOT / "ckpt/seq_colab/tokenizer") if p.exists()), None)
    tok = AutoTokenizer.from_pretrained(
        local_tokenizer or "bert-base-uncased",
        local_files_only=local_tokenizer is not None,
        cache_dir=str(ROOT / ".cache/hf/hub"),
    )
    tok.add_special_tokens({"additional_special_tokens": SPECIAL})
    tok.save_pretrained(OUT / "tokenizer")
    cache = {}

    def wp(word):
        if word not in cache:
            cache[word] = tok.encode(word, add_special_tokens=False) or [tok.unk_token_id]
        return cache[word]

    writers = {s: gzip.open(OUT / f"{s}.jsonl.gz", "wt") for s in ("train", "dev", "test")}
    stats = {s: Counter() for s in writers}
    conversations = defaultdict(set)
    targets, label_counts = Counter(), defaultdict(Counter)
    unsupported = Counter()
    with gzip.open(ROOT / "data/processed/swb_tokens.jsonl.gz", "rt") as f:
        for line in f:
            side = json.loads(line)
            split = split_of(side["conv"])
            conversations[split].add(side["conv"])
            stats[split]["sides"] += 1
            skeleton, gaps, pending = [], [], []
            for t in side["tokens"]:
                if t["label"] == "NOISE":
                    continue  # no fabricated phone evidence for nonspeech events
                if t["label"] in WTAGS:
                    skeleton.append(t)
                    gaps.append(pending)
                    pending = []
                else:
                    assert t["label"] in TYPE_ID, t["label"]
                    pending.append(t)
            gaps.append(pending)
            if not skeleton:
                stats[split]["all_disfluent_sides_excluded"] += 1
                continue

            # Window boundaries determined only by input words/phone lengths.
            # Budget leaves room for the configured stochastic phone insertions.
            a, wi = 0, 0
            while a < len(skeleton):
                b, cost = a, 3
                while b < len(skeleton) and b-a < MAX_WORDS:
                    s = skeleton[b]
                    word = s["intended"] if s["label"] == "MISPRONUNCIATION" else s["spoken"]
                    delta = len(wp(word)) + len(s["phones"] or []) + sum(len(t["phones"] or []) for t in gaps[b])
                    if b > a and cost + delta > 135:
                        break
                    cost += delta
                    b += 1
                final = b == len(skeleton)
                words = [t["intended"] if t["label"] == "MISPRONUNCIATION" else t["spoken"] for t in skeleton[a:b]]
                slots, gold_slots, phones, roles, pointers = [], [], [], [], []

                def emit(t, label, pointer):
                    p = t["phones"] or []
                    phones.extend(p)
                    roles.extend([TYPE_ID.get(label, 0)] * len(p))
                    pointers.extend([pointer] * len(p))

                for i in range(b-a+1):
                    if i == b-a and not final:
                        slots.append(None)
                        gold_slots.append(None)
                        continue
                    units, surfaces = [], []
                    for t in gaps[a+i]:
                        if t["label"] == "PARTIAL_WORD":
                            # Preserve the spoken fragment instead of completing it.
                            surface = ("-" if t["cut"] in {"start", "both"} else "") + t["spoken"] + ("-" if t["cut"] in {"end", "both"} else "")
                            target = "W:" + surface
                        else:
                            surface = t["spoken"]
                            target = restore_target(t["label"], surface, t["intended"], words[:i], words[i:])
                        units.append(t["label"] + "|" + target)
                        surfaces.append(surface)
                        emit(t, t["label"], i)
                        if split == "train":
                            targets[target] += 1
                        label_counts[split][t["label"]] += 1
                    slots.append(units)
                    gold_slots.append(surfaces)
                    if i < b-a:
                        emit(skeleton[a+i], "NONE", i)
                reference = []
                for i in range(len(words)+1):
                    reference.extend(gold_slots[i] or [])
                    if i < len(words):
                        # Reference includes the actual spoken spelling of substitutions.
                        reference.append(skeleton[a+i]["spoken"])
                length = 3 + sum(len(wp(w)) for w in words) + len(phones)
                if length > MAX_LENGTH:
                    stats[split]["oversize_windows_excluded"] += 1
                    stats[split]["oversize_reference_tokens_excluded"] += len(reference)
                else:
                    ex = dict(id=f"{side['sid']}:v2:{wi}", sid=side["sid"], conv=side["conv"], words=words,
                              phones=phones, slots=slots, reference=reference, roles=roles, pointers=pointers,
                              wtags=[WTAGS.index(t["label"]) for t in skeleton[a:b]])
                    writers[split].write(json.dumps(ex, separators=(",", ":")) + "\n")
                    st = stats[split]
                    st["windows"] += 1
                    st["flat_words"] += len(words)
                    st["reference_words"] += len(reference)
                    st["phones"] += len(phones)
                    st["overflow_units"] += sum(max(0, len(s or [])-K) for s in slots)
                    st["insertion_tokens"] += sum(len(s or []) for s in slots)
                    st["prolongation_tokens"] += sum(x == 1 for x in ex["wtags"])
                    st["mispronunciation_tokens"] += sum(x == 2 for x in ex["wtags"])
                a, wi = b, wi+1
    for f in writers.values():
        f.close()
    assert not any(conversations[a] & conversations[b] for a, b in [("train", "dev"), ("train", "test"), ("dev", "test")])
    vocab = ["UNK"] + [t for t, _ in targets.most_common()]
    save_json(OUT / "target_vocab.json", vocab)
    summary = dict(splits={s: dict(stats[s], conversations=len(conversations[s]), label_tokens=dict(label_counts[s])) for s in stats},
                   target_classes=len(vocab), max_units_per_slot=K, max_words=MAX_WORDS,
                   input="[CLS] flat transcript [END] [SEP] full unsegmented phone stream [SEP]",
                   phone_source="MS98 pronunciation lexicon, not recognized audio",
                   label_source="Existing v1 gold/silver project labels; not all manually verified",
                   split="test: 4010..4153; dev: 4519..4936; train: remainder",
                   tokenizer_source="cached bert-base-uncased WordPiece; no v1 model weights used",
                   source_sha256=hashlib.sha256((ROOT / "data/processed/swb_tokens.jsonl.gz").read_bytes()).hexdigest())
    save_json(OUT / "stats.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
