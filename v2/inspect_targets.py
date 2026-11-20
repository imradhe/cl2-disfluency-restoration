#!/usr/bin/env python3
"""Count canonical-target vocabulary coverage and training-capacity exclusions.

These are coverage diagnostics, not theoretical F1 ceilings: another output
class can sometimes render the same surface as the canonical gold class.
"""
import gzip
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
K = 8


def main():
    vocab = set(json.loads((ROOT / "v2/data/target_vocab.json").read_text()))
    result = {"target_classes": len(vocab), "max_insertions_per_gap": K, "splits": {}}
    for split in ("train", "dev", "test"):
        counts = defaultdict(Counter)
        with gzip.open(ROOT / "v2/data" / f"{split}.jsonl.gz", "rt") as f:
            for line in f:
                ex = json.loads(line)
                for slot in ex["slots"]:
                    for i, unit in enumerate(slot or []):
                        label, target = unit.split("|", 1)
                        c = counts[label]
                        c["gold_tokens"] += 1
                        c["beyond_capacity"] += i >= K
                        c["target_not_in_train_vocabulary"] += target not in vocab
                        c["representable"] += i < K and target in vocab and target != "UNK"
        summary = {}
        for label, c in sorted(counts.items()):
            r = c["representable"]/c["gold_tokens"]
            summary[label] = dict(c, canonical_target_coverage_within_first_k=r)
        result["splits"][split] = summary
    out = ROOT / "v2/reports/target_capacity.json"
    out.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result["splits"]["test"], indent=2))


if __name__ == "__main__":
    main()
