"""End-to-end disfluency restoration on one audio file.

    audio -> VAD chunks -> ASR words (Whisper) ----------------> G2P expected phones --+
                       \\-> phone recognizer (wav2vec2 CTC) -----> recognized phones ---+-> align -> gaps
    gaps + word context -> BERT gap model -> restored tokens -> restored transcript

Each expensive stage is cached under ``out_dir`` so that re-running with different gap-extraction
or model settings does not repeat VAD, Whisper or the phone recognizer.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

from model.dataset import GapDataset
from model.infer import load_gap_model, predict
from phones.g2p import G2P

from .asr import DEFAULT_MODEL as WHISPER, MlxWhisper, Word
from .audio import Chunk, SR, chunk_speech, read_wav, to_16k_mono, vad_spans
from .extract import GapCand, GapParams, build_examples, find_gaps
from .phonerec import MODEL as PHONE_MODEL, PhoneRecognizer, RecPhone
from .restore import Token, restore, to_text


@dataclass
class Config:
    ckpt: Path
    lexicon: Path
    phonemap: Path
    whisper: str = WHISPER
    phone_model: str = PHONE_MODEL
    bert: str = "bert-base-uncased"
    ctx: int = 16
    max_chunk: float = 28.0
    limit_s: float = 0.0                 # only process the first N seconds (0 = all)
    gap: GapParams = field(default_factory=GapParams)
    # Detected but not inserted into the transcript. Partial words are restored as the *full* word the
    # speaker was heading for, which is wrong in a verbatim transcript, and they were the main source of
    # false insertions on english_001 (see README). They stay in restored.json.
    skip_types: frozenset[str] = frozenset({"PARTIAL_WORD"})
    force: bool = False                  # recompute cached stages


def _slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_")


def _cached(path: Path, force: bool, make):
    if path.exists() and not force:
        return json.loads(path.read_text())
    data = make()
    path.write_text(json.dumps(data))
    return data


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def run(audio: Path, out_dir: Path, cfg: Config) -> tuple[dict, list[Token]]:
    """Run the pipeline; writes ``restored*.txt``, ``restored.json``, ``asr.txt`` and ``summary.json`` to
    ``out_dir`` and returns ``(summary, restored tokens)``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    wav = audio if _is_16k(audio) else to_16k_mono(audio, out_dir / f"{audio.stem}.16k.wav")
    x = read_wav(wav)
    if cfg.limit_s:
        x = x[: int(cfg.limit_s * SR)]
    total = len(x) / SR
    _log(f"audio {total:.1f}s")

    # --- VAD ---------------------------------------------------------------------------
    def make_vad():
        spans = vad_spans(x, max_len=cfg.max_chunk)
        return [[c.start, c.end] for c in chunk_speech(spans, total, max_len=cfg.max_chunk)]
    vad = _cached(out_dir / f"vad_{int(total)}s.json", cfg.force, make_vad)
    chunks = [Chunk(i, s, e) for i, (s, e) in enumerate(vad)]
    _log(f"VAD: {len(chunks)} chunks, {sum(c.end - c.start for c in chunks):.0f}s of speech")

    # --- ASR ---------------------------------------------------------------------------
    def make_asr():
        asr = MlxWhisper(cfg.whisper)
        res = []
        for c in chunks:
            res.append([w.to_json() for w in asr.transcribe(c.samples(x), c.start)])
            if (c.idx + 1) % 20 == 0:
                _log(f"  ASR {c.idx + 1}/{len(chunks)}")
        return res
    asr_json = _cached(out_dir / f"asr_{_slug(cfg.whisper)}_{int(total)}s.json", cfg.force, make_asr)
    chunk_words: list[list[Word]] = [[Word(w["text"], w["start"], w["end"], w["prob"]) for w in ws] for ws in asr_json]
    words = [w for ws in chunk_words for w in ws]
    _log(f"ASR: {len(words)} words")

    # --- phone recognition -----------------------------------------------------------------
    g2p = G2P.from_files(cfg.lexicon, cfg.phonemap)

    def make_phones():
        rec = PhoneRecognizer(g2p.ipa, cfg.phone_model)
        res = []
        for c in chunks:
            res.append([p.to_json() for p in rec.recognize(c.samples(x), c.start)])
            if (c.idx + 1) % 20 == 0:
                _log(f"  phones {c.idx + 1}/{len(chunks)}")
        return res
    ph_json = _cached(out_dir / f"phones_{_slug(cfg.phone_model)}_{int(total)}s.json", cfg.force, make_phones)
    chunk_rec: list[list[RecPhone]] = [[RecPhone(**p) for p in ps] for ps in ph_json]
    _log(f"phones: {sum(len(r) for r in chunk_rec)} recognized")

    # --- gaps ------------------------------------------------------------------------------
    gaps: list[GapCand] = []
    totals: dict[str, int] = {}
    base = 0
    for ws, rec in zip(chunk_words, chunk_rec):
        g, st = find_gaps(ws, rec, g2p, base_word=base, params=cfg.gap)
        gaps += g
        base += len(ws)
        for k, v in st.items():
            totals[k] = totals.get(k, 0) + v
    n_cand = sum(totals.get(k, 0) for k in ("kept", "too_short", "too_long", "low_prob", "too_brief"))
    _log(f"gaps: {len(gaps)} kept of {n_cand} candidates")

    # --- BERT restoration ----------------------------------------------------------------------
    texts = [w.text for w in words]
    examples = build_examples(gaps, texts, sid=audio.stem, ctx=cfg.ctx)
    model, tok, rest_vocab, dev = load_gap_model(cfg.ckpt, bert=cfg.bert)
    ds = GapDataset(examples, tok, rest_vocab, noise=None)
    preds = predict(model, [ds[i] for i in range(len(ds))], dev, rest_vocab)
    tokens = restore(words, gaps, preds, ctx=cfg.ctx, skip_types=cfg.skip_types)
    n_restored = sum(t.source == "restored" for t in tokens)
    _log(f"restored {n_restored} tokens in {sum(bool(p) for p in preds)} gaps")

    # --- outputs ---------------------------------------------------------------------------
    (out_dir / "asr.txt").write_text(" ".join(texts) + "\n")
    (out_dir / "restored.txt").write_text(to_text(tokens) + "\n")
    (out_dir / "restored_annotated.txt").write_text(to_text(tokens, annotate=True) + "\n")
    (out_dir / "restored.json").write_text(json.dumps({
        "tokens": [t.to_json() for t in tokens],
        "gaps": [{**g.to_json(), "left": e["left"][-6:], "right": e["right"][:4],
                  "pred": [{"s": u["s"], "e": u["e"], "type": u["type"], "rest": u["restoration"]} for u in p]}
                 for g, e, p in zip(gaps, examples, preds)],
    }, ensure_ascii=False, indent=1))
    by_type: dict[str, int] = {}
    for t in tokens:
        if t.source == "restored":
            by_type[t.label] = by_type.get(t.label, 0) + 1
    summary = {"audio": str(audio), "seconds": round(total, 1), "chunks": len(chunks), "asr_words": len(words),
               "gap_stats": totals, "gaps": len(gaps), "restored_tokens": n_restored, "restored_by_type": by_type,
               "ckpt": str(cfg.ckpt), "whisper": cfg.whisper, "elapsed_s": round(time.time() - t_start, 1)}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary, tokens


def _is_16k(path: Path) -> bool:
    import wave
    try:
        with wave.open(str(path), "rb") as w:
            return w.getframerate() == SR and w.getnchannels() == 1 and w.getsampwidth() == 2
    except (wave.Error, EOFError):
        return False
