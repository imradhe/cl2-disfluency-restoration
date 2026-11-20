"""End-to-end disfluency restoration on one audio file.

    audio -> VAD chunks -> ASR words (Whisper) -> transcript ---> G2P phones of the transcript --+
                       \\-> phone recognizer (wav2vec2 CTC) -> recognized phone sequence ---------+-> BERT -> restored transcript

The BERT model (src/model/seq.py) is trained on exactly these three inputs (disfluency-removed transcript, its
G2P phones, simulated recognized phones) and predicts the disfluencies to restore in front of every word.
Each expensive stage is cached under ``out_dir`` so that re-running with another checkpoint or threshold does
not repeat VAD, Whisper or the phone recognizer.
"""
from __future__ import annotations

import json
import re
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path

from model.bert_seq import load_seq_model
from model.seq import predict_seq
from phones.g2p import G2P

from .asr import DEFAULT_MODEL as WHISPER, MlxWhisper, Word, normalize_fillers
from .audio import SR, Chunk, chunk_speech, read_wav, to_16k_mono, vad_spans
from .phonerec import MODEL as PHONE_MODEL, PhoneRecognizer, RecPhone
from .restore import Token, restore_windows, to_text
from .windows import Window, make_windows


@dataclass
class Config:
    ckpt: Path
    lexicon: Path
    phonemap: Path
    whisper: str = WHISPER
    phone_model: str = PHONE_MODEL
    bert: str = "bert-base-uncased"
    max_chunk: float = 28.0
    max_words: int = 20                  # words per BERT window (training used 10-24)
    none_bias: float = 0.0               # > 0: restore more (subtracted from the "nothing here" logit)
    limit_s: float = 0.0                 # only process the first N seconds (0 = all)
    # Detected but not inserted into the transcript (they stay in restored.json). Empty = insert everything.
    skip_types: frozenset[str] = field(default_factory=frozenset)
    norm_fillers: bool = True            # respell Whisper's filler "ah" as "uh" before windowing (see asr.FILLER_SPELLING)
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


def _is_16k(path: Path) -> bool:
    try:
        with wave.open(str(path), "rb") as w:
            return w.getframerate() == SR and w.getnchannels() == 1 and w.getsampwidth() == 2
    except (wave.Error, EOFError):
        return False


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
    raw_words = [w for ws in chunk_words for w in ws]
    n_norm = 0
    if cfg.norm_fillers:
        fixed = [normalize_fillers(ws) for ws in chunk_words]
        chunk_words, n_norm = [ws for ws, _ in fixed], sum(n for _, n in fixed)
    words = [w for ws in chunk_words for w in ws]
    _log(f"ASR: {len(words)} words ({n_norm} filler spellings normalized)")

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

    # --- windows -> BERT ---------------------------------------------------------------------
    windows: list[Window] = []
    for c, ws, rec in zip(chunks, chunk_words, chunk_rec):
        windows += make_windows(ws, rec, g2p, c.start, c.end, max_words=cfg.max_words)
    model, tok, vocab, dev = load_seq_model(cfg.ckpt, bert=cfg.bert)
    examples = [w.example(audio.stem, k) for k, w in enumerate(windows)]
    preds = predict_seq(model, examples, tok, vocab, dev, none_biases=(cfg.none_bias,))[cfg.none_bias]
    slot_preds = [p["slots"] for p in preds]
    tokens = restore_windows(windows, slot_preds, cfg.skip_types)
    n_restored = sum(t.source == "restored" for t in tokens)
    n_units = sum(len(s) for sp in slot_preds for s in sp)
    _log(f"{len(windows)} windows; {n_units} units predicted, {n_restored} inserted")

    # --- outputs ---------------------------------------------------------------------------
    (out_dir / "asr_raw.txt").write_text(" ".join(w.text for w in raw_words) + "\n")     # Whisper as written
    (out_dir / "asr.txt").write_text(" ".join(w.text for w in words) + "\n")             # what the windows were built on
    (out_dir / "restored.txt").write_text(to_text(tokens) + "\n")
    (out_dir / "restored_annotated.txt").write_text(to_text(tokens, annotate=True) + "\n")
    (out_dir / "restored.json").write_text(json.dumps({
        "tokens": [t.to_json() for t in tokens],
        "windows": [{"t0": round(w.t0, 3), "t1": round(w.t1, 3), "words": [x.text for x in w.words], "rec": w.rec,
                     "units": [[s, u] for s, u in enumerate(sp) if u]} for w, sp in zip(windows, slot_preds)],
    }, ensure_ascii=False, indent=1))
    by_type: dict[str, int] = {}
    for sp in slot_preds:
        for slot in sp:
            for u in slot:
                by_type[u.split("|", 1)[0]] = by_type.get(u.split("|", 1)[0], 0) + 1
    summary = {"audio": str(audio), "seconds": round(total, 1), "chunks": len(chunks), "asr_words": len(words),
               "windows": len(windows), "predicted_units": n_units, "predicted_by_type": by_type,
               "inserted_tokens": n_restored, "skipped_types": sorted(cfg.skip_types), "none_bias": cfg.none_bias,
               "normalized_fillers": n_norm,
               "ckpt": str(cfg.ckpt), "whisper": cfg.whisper, "elapsed_s": round(time.time() - t_start, 1)}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary, tokens
