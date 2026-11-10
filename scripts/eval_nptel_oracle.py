#!/usr/bin/env python
"""NPTEL oracle-gap evaluation (text only; no audio needed).

Takes the fully verified English NPTEL lectures, uses the human gold tags to make the
same gap / word-span examples as in training (disfluent tokens removed from the skeleton,
their phones kept), and measures how a Switchboard-trained checkpoint transfers.

Phones are G2P (sw-ms98 lexicon, else espeak-ng), i.e. *clean*; a noisy variant corrupts
them with the recognizer-noise simulator. Token times are approximate (segment time split
evenly), so duration features are not informative here.

Run: HF_HOME=.cache/hf HF_HUB_OFFLINE=1 .venv/bin/python -I scripts/eval_nptel_oracle.py --ckpt ckpt/local60k
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

import torch
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from model.bert_gap import BertGap  # noqa: E402
from model.dataset import TAGS, GapDataset, load_jsonl  # noqa: E402
from model.gaps import WordDur, build_side  # noqa: E402
from model.infer import predict, unit_prf  # noqa: E402
from nptel.rsml import aligned_to_side, parse_rsml, to_side  # noqa: E402
from phones.g2p import G2P  # noqa: E402
from phones.noise import PhoneNoise  # noqa: E402


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")


def load_recognized(path: Path) -> tuple[list[float], list[dict]]:
    """Flatten the recognizer output of one lecture into time-sorted tokens (+ their centres)."""
    toks = []
    for line in path.read_text().splitlines():
        toks.extend(json.loads(line)["tokens"])
    toks.sort(key=lambda t: t["start"])
    # segments are padded and may overlap: drop duplicates at (almost) the same time
    uniq, last = [], None
    for t in toks:
        if last and t["tok"] == last["tok"] and abs(t["start"] - last["start"]) < 0.03:
            continue
        uniq.append(t); last = t
    return [(t["start"] + t["end"]) / 2 for t in uniq], uniq


def recognized_phones(centres: list[float], toks: list[dict], ipa, t0: float, t1: float, tol: float = 0.03) -> list[str]:
    import bisect
    lo = bisect.bisect_left(centres, t0 - tol)
    hi = bisect.bisect_right(centres, t1 + tol)
    return ipa.map_sequence([t["tok"] for t in toks[lo:hi]])


def build_nptel_examples(rsml_dir: Path, g2p: G2P, ctx: int, phones_dir: Path | None = None) -> tuple[list[dict], dict]:
    """phones_dir=None -> G2P phones (clean); else recognized phones inside each token's time span."""
    examples, info = [], {"files": [], "gold_tokens": Counter(), "dropped_tags": Counter(),
                          "gold_units_total": Counter(), "gold_units_without_phones": Counter()}
    for p in sorted(rsml_dir.rglob("*.rsml")):
        if p.name.endswith(".aligned.rsml"):
            continue
        r = parse_rsml(p)
        if not (r.is_english and r.fully_verified):
            continue
        aligned = p.with_name(p.stem + ".aligned.rsml")
        # forced-aligned file gives real word times; otherwise fall back to even splitting
        side = aligned_to_side(aligned, sid=p.stem) if aligned.exists() else to_side(r)
        info.setdefault("aligned", []).append(aligned.exists())
        rec = None
        if phones_dir is not None:
            pf = phones_dir / f"{p.stem}.jsonl"
            if not pf.exists():
                raise SystemExit(f"no recognized phones for {p.stem}: run scripts/nptel_phones.py first")
            rec = load_recognized(pf)
        for t in side["tokens"]:
            ph = g2p.phones(t["spoken"]) if rec is None else recognized_phones(rec[0], rec[1], g2p.ipa, t["start"], t["end"])
            t["phones"] = ph or None
            info["gold_tokens"][t["label"]] += 1
            if t["label"] != "O":
                info["gold_units_total"][t["label"]] += 1
                info["gold_units_without_phones"][t["label"]] += int(not ph)
        info["dropped_tags"].update(side["dropped_tags"])
        info["files"].append({"file": p.name, "segments": len(r.segments), "minutes": round(r.duration / 60, 1)})
        # duration z-scores need per-word reference durations: normalise against the lecture's own
        # speaking rate (unsupervised; no labels used). Falls back to zb=0 without real word times.
        wd = WordDur()
        if aligned.exists():
            wd.add_side(side["tokens"])
            wd.finalize(min_n=10)
        examples += list(build_side(side, wd, random.Random(0), PhoneNoise(seed=0), ctx=ctx, noise_gaps=False))
    return examples, info


def show_samples(items: list[dict], examples: list[dict], preds: list[list[dict]], rest_vocab: list[str], n: int) -> list[str]:
    rng = random.Random(1)
    idx = [i for i, e in enumerate(examples) if e["kind"] == "gap"]
    rng.shuffle(idx)
    lines = []
    for i in idx[:n]:
        e, it = examples[i], items[i]
        gold = " + ".join(f"{label}:{tgt}" for (_, _, label, tgt) in it["gold_units"])
        pred = " + ".join(f"{p['type']}:{p['restoration']}" for p in preds[i]) or "(nothing)"
        lines.append(f"{' '.join(e['left'][-6:])} [[{' '.join(e['phones'])}]] {' '.join(e['right'][:4])}\n"
                     f"     gold: {gold}\n     pred: {pred}")
    return lines


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--bert", default="bert-base-uncased")
    ap.add_argument("--rsml-dir", default=str(ROOT / "data/nptel/raw"))
    ap.add_argument("--lex", default=str(ROOT / "data/raw/switchboard_ms98/sw-ms98-dict.text"))
    ap.add_argument("--gaps", default=str(ROOT / "data/gaps"))
    ap.add_argument("--dev-n", type=int, default=3000)
    ap.add_argument("--samples", type=int, default=15)
    ap.add_argument("--out", default=str(ROOT / "results"))
    ap.add_argument("--phones", choices=["g2p", "recognized"], default="g2p",
                    help="g2p: clean phones from lexicon/espeak; recognized: wav2vec2 output on the real audio")
    ap.add_argument("--phones-dir", default=str(ROOT / "data/nptel/phones"))
    a = ap.parse_args()

    dev = pick_device()
    ckpt = Path(a.ckpt)
    tok = AutoTokenizer.from_pretrained(ckpt / "tokenizer")
    rest_vocab = json.loads((Path(a.gaps) / "restoration_vocab.json").read_text())
    model = BertGap(a.bert, len(tok), len(TAGS), len(rest_vocab)).to(dev)
    model.load_state_dict(torch.load(ckpt / "model.pt", map_location=dev))
    model.eval()

    g2p = G2P.from_files(a.lex, ROOT / "configs/phonemap.yaml")
    nptel, info = build_nptel_examples(Path(a.rsml_dir), g2p, ctx=16,
                                       phones_dir=Path(a.phones_dir) if a.phones == "recognized" else None)
    print(f"phone source: {a.phones}")
    if a.phones == "recognized":
        tot, miss = info["gold_units_total"], info["gold_units_without_phones"]
        print("gold disfluent tokens with NO recognized phone in their span: "
              + ", ".join(f"{k} {miss[k]}/{tot[k]}" for k in tot))
    print(f"NPTEL files: {info['files']}")
    print(f"G2P: {g2p.n_lex} words from lexicon, {g2p.n_espeak} via espeak-ng")
    print(f"gold token labels: {dict(info['gold_tokens'])}")
    print(f"examples: {Counter(e['kind'] for e in nptel)}  units: "
          f"{dict(Counter(u['label'] for e in nptel for u in e['units']))}")

    sets = {
        "nptel_clean": (nptel, None),
        "nptel_noisy": (nptel, PhoneNoise(seed=5)),
        "switchboard_dev_noisy": (load_jsonl(Path(a.gaps) / "dev.jsonl.gz", a.dev_n), PhoneNoise(seed=1)),
    }
    results, sample_lines = {}, []
    for name, (exs, noise) in sets.items():
        ds = GapDataset(exs, tok, rest_vocab, noise=noise)
        items = [ds[i] for i in range(len(ds))]
        preds = predict(model, items, dev, rest_vocab)
        res = unit_prf([it["gold_units"] for it in items], preds)
        res["n_examples"] = len(exs)
        results[name] = res
        if name == "nptel_clean":
            sample_lines = show_samples(items, exs, preds, rest_vocab, a.samples)
        print(f"\n=== {name}  (examples={len(exs)})  macroF1 strict={res['macro_f1_strict']} "
              f"lenient={res['macro_f1_lenient']}  restoration acc on matched units={res['restoration_acc_on_strict_matches']} "
              f"(n={res['strict_matches']})")
        print(f"  {'type':20s}{'support':>8s}{'pred':>7s}   strict P/R/F1        lenient P/R/F1")
        for t, v in res["per_type"].items():
            s, l = v["strict"], v["lenient"]
            print(f"  {t:20s}{v['support']:8d}{v['predicted']:7d}   {s['p']:.2f}/{s['r']:.2f}/{s['f1']:.2f}        "
                  f"{l['p']:.2f}/{l['r']:.2f}/{l['f1']:.2f}")

    print("\n--- NPTEL samples (clean phones) ---")
    print("\n".join(sample_lines))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    dest = out / f"nptel_oracle_{a.phones}_{ckpt.name}.json"
    dest.write_text(json.dumps(
        {"ckpt": str(ckpt), "phones": a.phones, "nptel_files": info["files"], "gold_tokens": dict(info["gold_tokens"]),
         "dropped_tags": dict(info["dropped_tags"]), "gold_units_without_phones": dict(info["gold_units_without_phones"]),
         "results": results}, indent=2))
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
