"""Sequence-level restoration: fluent transcript + its G2P phones + the recognized phones -> restored transcript.

One example is a *window* of consecutive words of the disfluency-removed transcript::

    {"id", "sid",
     "words": [w_1 .. w_n],              # the transcript an ASR would give (disfluencies absent)
     "exp":   [[phones of w_1], ...],    # G2P phones of every word
     "rec":   [phone, ...],              # what a phone recognizer hears: *every* spoken token of the
                                         # window, disfluent ones included (noise is added at load time)
     "slots": [[unit, ...] * (n + 1)],   # slot i = "before word i" (slot n = after the last word);
                                         # None = not a target (that gap belongs to the next window)
     "wtags": [0|1|2 * n],               # per word: fluent / prolonged / mispronounced
     "span": [lo, hi],                   # token index range of the side the window covers (evaluation on real audio)
     "rec_role": [ROLES index * len(rec)], "rec_ptr": [word index * len(rec)],   # auxiliary labels of the recognized phones
     "rec_tok": [token index * len(rec)], "tok_lab": [label of every token]}     # token structure of ``rec`` (ProfileNoise)
    unit  = "LABEL|TARGET"               # e.g. "FILLER_HESITATION|FIL:um", "REPETITION|COPY_R1"

The BERT input is ``[CLS] w_1 <g2p of w_1> w_2 <g2p of w_2> ... [END] [SEP] <recognized phones> [SEP]``.
The model reads the hidden state of every word's first piece (and of ``[END]``) and classifies the slot in
front of it: up to ``K_UNITS`` units per slot, each a class of ``unit_vocab``. No alignment is computed:
the model has to find, in the recognized phones, what the transcript's own phones do not explain.
"""
from __future__ import annotations

import random
from collections import Counter, defaultdict
from typing import Callable, Iterator

import torch

from phones.noise import PhoneNoise

from .dataset import IGNORE, PHONE_TOKENS
from .gaps import FILLER_LABELS, SKELETON_LABELS, restore_target

K_UNITS = 4                              # units predicted per slot
WTAGS = ["O", "PROLONGATION", "MISPRONUNCIATION"]
# Auxiliary, training-time-only supervision of the recognized phones (dense, unlike the per-slot labels): what
# each phone is -- explained by a transcript word (O) or a disfluency of this type -- and which word it belongs to
# (explained) or precedes (extra). Without it the model has to discover the phone-to-word alignment from the
# sparse slot labels alone, which did not get off the label prior in a local 10k-window run.
ROLES = ["O", "FILLER_HESITATION", "FILLER_BACKCHANNEL", "FILLER_DM", "REPETITION", "REPAIR", "FALSE_START",
         "PARTIAL_WORD", "NOISE"]
MAX_PTR = 25                             # pointer classes: word 0..24 (windows have <= 24 words; slot n = after the last)
NONE, UNK = "<none>", "<unk>"
SEQ_SPECIAL = ["[END]"] + PHONE_TOKENS
# Disfluent tokens that an ASR sometimes writes anyway (Whisper kept ~40% of them on NPTEL); the training
# transcript keeps each of them with probability ``keep_p`` so the model also sees such transcripts.
KEEP_LABELS = FILLER_LABELS | {"REPETITION", "REPAIR", "FALSE_START"}
BUDGET = 215.0                           # window size limit, in estimated tokens (see word_cost)
MAX_LEN = 320


def word_cost(n_phones: int) -> float:
    """Estimated BERT tokens of a word: ~2 pieces + its G2P phones + ~1.1x as many recognized phones."""
    return 2.0 + 2.1 * n_phones


def split_windows(costs: list[float], size: Callable[[], int], budget: float = BUDGET) -> Iterator[tuple[int, int]]:
    """Consecutive ``[a, b)`` word windows of at most ``size()`` words and ``budget`` estimated tokens."""
    a, n = 0, len(costs)
    while a < n:
        limit, b, c = size(), a, 0.0
        while b < n and b - a < limit and (b == a or c + costs[b] <= budget):
            c += costs[b]
            b += 1
        yield a, b
        a = b


def unit_string(label: str, target: str) -> str:
    return f"{label}|{target}"


def unit_label(unit: str) -> str:
    return unit.split("|", 1)[0]


def _skeleton_word(t: dict) -> str:
    return t["intended"] if t["label"] == "MISPRONUNCIATION" and t["intended"] else t["spoken"]


def _g2p(lex, word: str, fallback: list[str] | None) -> list[str]:
    prons = lex.get(word) if lex is not None else []
    return list(prons[0]) if prons else list(fallback or [])


def build_seq_side(side: dict, lex, rng: random.Random, noise: PhoneNoise, keep_p: float = 0.0,
                   sizes: tuple[int, int] = (10, 24), noise_p: float = 0.5) -> Iterator[dict]:
    """Windows of one labelled Switchboard side. ``sizes`` = (min, max) words per window (equal -> fixed)."""
    toks, sid = side["tokens"], side["sid"]

    # --- pass 1: transcript words, and the disfluent tokens sitting in front of each of them -------------
    skel: list[dict] = []
    slot_toks: list[list[dict]] = []                     # slot_toks[p]: disfluent tokens before skel[p]
    pending: list[dict] = []
    for t in toks:
        lab = t["label"]
        if lab == "NOISE":
            if rng.random() < noise_p:                   # the recognizer "hears" junk where there was noise
                pending.append({"label": "NOISE", "spoken": "", "intended": "", "phones": noise.spurious(), "i": t["i"]})
            continue
        keep = lab in KEEP_LABELS and t["kind"] == "WORD" and rng.random() < keep_p
        if lab in SKELETON_LABELS or keep:
            word = _skeleton_word(t)
            skel.append({"word": word, "exp": _g2p(lex, word, t["phones"]), "rec": list(t["phones"] or []),
                         "wtag": WTAGS.index(lab) if lab in WTAGS else 0, "i": t["i"], "lab": lab})
            slot_toks.append(pending)
            pending = []
        elif t["phones"]:
            pending.append({"label": lab, "spoken": t["spoken"], "intended": t["intended"],
                            "phones": list(t["phones"]), "i": t["i"]})
    slot_toks.append(pending)                            # trailing disfluent tokens
    if not skel:
        return

    # --- pass 2: windows -----------------------------------------------------------------------------------
    lo, hi = sizes
    assert hi < MAX_PTR, "windows must have < MAX_PTR words"
    costs = [word_cost(len(w["exp"])) for w in skel]
    for k, (a, b) in enumerate(split_windows(costs, lambda: rng.randint(lo, hi))):
        final = b == len(skel)
        words = [skel[p]["word"] for p in range(a, b)]
        slots: list[list[str] | None] = []
        rec: list[str] = []
        role: list[int] = []
        ptr: list[int] = []
        tok: list[int] = []                              # which token of the window every recognized phone belongs to
        tok_lab: list[str] = []                          # original label of every token ("O" = fluent word)

        def emit(phones: list[str], lab: str, role_idx: int, word_idx: int) -> None:
            rec.extend(phones)
            role.extend([role_idx] * len(phones))
            ptr.extend([word_idx] * len(phones))
            tok.extend([len(tok_lab)] * len(phones))
            tok_lab.append(lab)

        for i, p in enumerate(range(a, b)):
            left, right = words[:i], words[i:]
            units = []
            for u in slot_toks[p]:
                tgt = "DROP" if u["label"] == "NOISE" else restore_target(u["label"], u["spoken"], u["intended"], left, right)
                units.append(unit_string(u["label"], tgt))
                emit(u["phones"], u["label"], ROLES.index(u["label"]), i)
            slots.append(units)
            emit(skel[p]["rec"], skel[p]["lab"], 0, i)
        if final:
            slots.append([unit_string(u["label"], "DROP" if u["label"] == "NOISE" else
                                      restore_target(u["label"], u["spoken"], u["intended"], words, []))
                          for u in slot_toks[b]])
            for u in slot_toks[b]:
                emit(u["phones"], u["label"], ROLES.index(u["label"]), b - a)
        else:
            slots.append(None)
        lo_i = min([u["i"] for u in slot_toks[a]] + [skel[a]["i"]])
        hi_i = max([skel[b - 1]["i"]] + ([u["i"] for u in slot_toks[b]] if final else [])) + 1
        yield {"id": f"{sid}:s{k}", "sid": sid, "words": words, "exp": [skel[p]["exp"] for p in range(a, b)],
               "rec": rec, "slots": slots, "wtags": [skel[p]["wtag"] for p in range(a, b)], "span": [lo_i, hi_i],
               "rec_role": role, "rec_ptr": ptr, "rec_tok": tok, "tok_lab": tok_lab}


def count_units(ex: dict, counts: Counter) -> None:
    for slot in ex["slots"]:
        counts.update(slot or [])


def make_unit_vocab(counts: Counter, top_k: int = 6000) -> list[str]:
    """``[<none>, <unk>, LABEL|UNK ..., most frequent unit strings]`` (built from training windows only)."""
    labels = sorted({unit_label(u) for u in counts})
    fixed = [NONE, UNK] + [unit_string(l, "UNK") for l in labels]
    rest = [u for u, _ in counts.most_common() if u not in fixed][:top_k]
    return fixed + rest


# ------------------------------------------------------------------------------------------ dataset

class SeqDataset(torch.utils.data.Dataset):
    def __init__(self, examples: list[dict], tokenizer, unit_vocab: list[str], max_len: int = MAX_LEN,
                 noise: PhoneNoise | None = None) -> None:
        self.ex, self.tok, self.max_len, self.noise = examples, tokenizer, max_len, noise
        self.u2id = {u: i for i, u in enumerate(unit_vocab)}
        self.sid = {t: tokenizer.convert_tokens_to_ids(t) for t in SEQ_SPECIAL}
        self.cache: dict[str, list[int]] = {}
        self.truncated = 0

    def __len__(self) -> int:
        return len(self.ex)

    def _wp(self, w: str) -> list[int]:
        c = self.cache.get(w)
        if c is None:
            c = self.cache[w] = self.tok.encode(w, add_special_tokens=False) or [self.tok.unk_token_id]
        return c

    def unit_id(self, u: str) -> int:
        i = self.u2id.get(u)
        if i is None:
            i = self.u2id.get(unit_string(unit_label(u), "UNK"), self.u2id[UNK])
        return i

    def __getitem__(self, i: int) -> dict:
        ex = self.ex[i]
        ids = [self.tok.cls_token_id]
        word_pos: list[int] = []
        for w, exp in zip(ex["words"], ex["exp"]):
            word_pos.append(len(ids))
            ids += self._wp(w) + [self.sid[f"<ph:{p}>"] for p in exp]
        end_pos = len(ids)
        ids += [self.sid["[END]"], self.tok.sep_token_id]
        n_a = len(ids)
        if self.noise is not None and ex["rec"]:
            rec, src = self.noise.corrupt_idx(ex["rec"], ex.get("rec_tok"), ex.get("tok_lab"))
        else:
            rec, src = ex["rec"], list(range(len(ex["rec"])))
        room = self.max_len - n_a - 1
        if len(rec) > room:
            rec, src, self.truncated = rec[:room], src[:room], self.truncated + 1
        role, ptr = ex.get("rec_role"), ex.get("rec_ptr")
        assert role is None or (len(role) == len(ex["rec"]) and len(ptr) == len(ex["rec"])), \
            f"{ex.get('id')}: rec_role / rec_ptr are out of sync with rec (drop them if rec was replaced)"
        rec_pos = list(range(n_a, n_a + len(rec))) if role is not None else []
        rec_role = [role[k] for k in src] if role is not None else []
        rec_ptr = [ptr[k] for k in src] if ptr is not None else []
        ids += [self.sid[f"<ph:{p}>"] for p in rec] + [self.tok.sep_token_id]
        types = [0] * n_a + [1] * (len(ids) - n_a)

        slot_pos, slot_lab = [], []
        for s, units in enumerate(ex["slots"]):
            slot_pos.append(word_pos[s] if s < len(word_pos) else end_pos)
            if units is None:
                slot_lab.append([IGNORE] * K_UNITS)
            else:
                lab = [self.unit_id(u) for u in units[:K_UNITS]]
                slot_lab.append(lab + [0] * (K_UNITS - len(lab)))       # 0 = <none>
        return {"ids": ids, "types": types, "slot_pos": slot_pos, "slot_lab": slot_lab,
                "word_pos": word_pos, "word_tag": list(ex["wtags"]),
                "rec_pos": rec_pos, "rec_role": rec_role, "rec_ptr": rec_ptr}


def collate_seq(batch: list[dict]) -> dict[str, torch.Tensor]:
    T, B = max(len(b["ids"]) for b in batch), len(batch)
    ids = torch.zeros(B, T, dtype=torch.long)
    types = torch.zeros(B, T, dtype=torch.long)
    att = torch.zeros(B, T, dtype=torch.long)
    sb, sp, sl, wb, wp, wt, rb, rp, rr, rq = [], [], [], [], [], [], [], [], [], []
    for k, b in enumerate(batch):
        n = len(b["ids"])
        ids[k, :n], types[k, :n], att[k, :n] = torch.tensor(b["ids"]), torch.tensor(b["types"]), 1
        sb += [k] * len(b["slot_pos"]); sp += b["slot_pos"]; sl += b["slot_lab"]
        wb += [k] * len(b["word_pos"]); wp += b["word_pos"]; wt += b["word_tag"]
        rb += [k] * len(b["rec_pos"]); rp += b["rec_pos"]; rr += b["rec_role"]; rq += b["rec_ptr"]
    L = torch.long
    return {"input_ids": ids, "token_type_ids": types, "attention_mask": att,
            "slot_batch": torch.tensor(sb, dtype=L), "slot_pos": torch.tensor(sp, dtype=L),
            "slot_lab": torch.tensor(sl, dtype=L).reshape(-1, K_UNITS),
            "word_batch": torch.tensor(wb, dtype=L), "word_pos": torch.tensor(wp, dtype=L),
            "word_tag": torch.tensor(wt, dtype=L),
            "rec_batch": torch.tensor(rb, dtype=L), "rec_pos": torch.tensor(rp, dtype=L),
            "rec_role": torch.tensor(rr, dtype=L), "rec_ptr": torch.tensor(rq, dtype=L)}


# ------------------------------------------------------------------------------------------ decoding

def decode_slot(logits: torch.Tensor, unit_vocab: list[str], none_bias: float = 0.0) -> list[str]:
    """Units of one slot from its ``[K, V]`` logits: stop at the first ``<none>``.
    ``none_bias`` is subtracted from the ``<none>`` logit (> 0 trades precision for recall)."""
    out: list[str] = []
    for j in range(logits.size(0)):
        row = logits[j].clone()
        row[0] -= none_bias
        k = int(row.argmax())
        if k == 0:
            break
        out.append(unit_vocab[k])
    return out


@torch.no_grad()
def predict_seq(model, examples: list[dict], tokenizer, unit_vocab: list[str], dev: torch.device,
                none_biases: tuple[float, ...] = (0.0,), noise: PhoneNoise | None = None, bs: int = 32,
                ) -> dict[float, list[dict]]:
    """Predictions for every ``none_bias``: ``{bias: [{"slots": [[unit, ..] * (n+1)], "wtags": [0|1|2 * n]}, ..]}``
    (one forward pass; no gold labels needed). ``noise`` corrupts the recognized phones (evaluation on clean data)."""
    model.eval()
    ds = SeqDataset(examples, tokenizer, unit_vocab, noise=noise)
    out: dict[float, list[dict]] = {b: [] for b in none_biases}
    for lo in range(0, len(ds), bs):
        items = [ds[i] for i in range(lo, min(len(ds), lo + bs))]
        b = {k: v.to(dev) for k, v in collate_seq(items).items()}
        res = model(b, compute_loss=False)
        slot_logits = res["slot_logits"].float().cpu()
        wtag = res["word_logits"].argmax(-1).cpu().tolist()
        s = w = 0
        for it in items:
            ns, nw = len(it["slot_pos"]), len(it["word_pos"])
            for bias in none_biases:
                out[bias].append({"slots": [decode_slot(slot_logits[s + i], unit_vocab, bias) for i in range(ns)],
                                  "wtags": wtag[w:w + nw]})
            s, w = s + ns, w + nw
    return out


# ------------------------------------------------------------------------------------------ metrics

def slot_prf(gold: list[list[list[str] | None]], pred: list[list[list[str]]]) -> dict:
    """Slot-level evaluation. ``gold[w][s]`` / ``pred[w][s]``: units of slot ``s`` of window ``w``
    (``None`` gold slots are skipped).

    * per type: a unit counts if the slot holds the same type in gold and prediction (multiset match per slot);
    * ``exact``: the full unit string (type + restoration) must match too;
    * ``slot``: binary "is there any disfluency before this word".
    """
    tp, n_p, n_g = defaultdict(int), defaultdict(int), defaultdict(int)
    ex = [0, 0, 0]                                       # tp, pred, gold (exact)
    sl = [0, 0, 0]
    for g_win, p_win in zip(gold, pred):
        for g, p in zip(g_win, p_win):
            if g is None:
                continue
            g = g[:K_UNITS]
            gl, pl = Counter(map(unit_label, g)), Counter(map(unit_label, p))
            for t in set(gl) | set(pl):
                tp[t] += min(gl[t], pl[t]); n_g[t] += gl[t]; n_p[t] += pl[t]
            ex[0] += sum((Counter(g) & Counter(p)).values()); ex[1] += len(p); ex[2] += len(g)
            sl[0] += bool(g) and bool(p); sl[1] += bool(p); sl[2] += bool(g)

    def prf(a: int, p: int, g: int) -> dict:
        pr, rc = a / p if p else 0.0, a / g if g else 0.0
        return {"p": round(pr, 4), "r": round(rc, 4), "f1": round(2 * pr * rc / (pr + rc), 4) if pr + rc else 0.0}

    per_type = {t: {"support": n_g[t], "predicted": n_p[t], **prf(tp[t], n_p[t], n_g[t])} for t in sorted(set(n_g) | set(n_p))}
    present = [t for t in per_type if n_g[t]]
    return {"per_type": per_type,
            "macro_f1": round(sum(per_type[t]["f1"] for t in present) / max(1, len(present)), 4),
            "exact": {**prf(*ex), "tp": ex[0]}, "slot": prf(*sl),
            "restoration_acc_on_type_matches": round(ex[0] / max(1, sum(tp.values())), 4)}
