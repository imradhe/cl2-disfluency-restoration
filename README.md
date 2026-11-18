# Disfluency Restoration (Switchboard → NPTEL)

Restore the disfluencies that ASR drops — fillers, repetitions, repairs, partial
words, prolongations, false starts, mispronunciations — by using **phones** as
evidence.

* **Train (text only):** free Switchboard (ISIP/Mississippi State) transcripts. Every
  disfluent token becomes a *gap* in the fluent word sequence; BERT sees the
  surrounding words plus the gap's phoneme sequence and predicts the disfluency
  tag(s) and the restored token(s).
* **Test (audio):** `audio → VAD → ASR → G2P` and `audio → phone recognizer`;
  forced alignment of the expected phones against the recognized phones; phones with
  no corresponding word are gaps, which BERT then restores.
* **Out-of-domain test:** NPTEL lectures whose `.rsml` segments are all verified.

The full plan lives in the project's planning notes; status is tracked below.

## Status

| Phase | What | State |
|---|---|---|
| 0 | Repo + environment | done (torch 2.15 + MPS, espeak-ng, uv) |
| 1 | Download Switchboard | done (4,876 sides; checksums in `data/MANIFEST.md`) |
| 2 | Parse + label | done (gold + silver rules; spot-check precision still to be measured) |
| 3 | Phoneme layer | lexicon + IPA map + noise simulator done; recognizer-calibrated noise in Phase 6 |
| 4 | Gap datasets | done (262k train / 16k dev / 13k test examples) |
| 5 | BERT training | full run done on Colab L4 (3 epochs, 261,868 examples, two replicate runs; dev unit-type acc 0.889, restoration acc 0.901). Notebook: `notebooks/train_colab.ipynb` |
| 6 | NPTEL ingestion | `english_001` ingested (audio, verified rsml, aligned rsml, recognized phones); other `en/` lectures pending |
| 7 | Inference pipeline | done: `scripts/run_pipeline.py` runs audio → VAD → Whisper + phone recognizer → alignment → gaps → BERT → restored transcript (`src/pipeline/`, see below). Works end to end, but on `english_001` it only breaks even on verbatim WER (0.0954 vs 0.0963 ASR-only); gap precision is the bottleneck |
| 8 | Evaluation | todo |
| 9 | Report | todo |

## Setup

```bash
brew install uv espeak-ng
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -e .            # text path (Phases 1-5)
uv pip install -e ".[audio]"   # audio path (Phases 6-7), heavy
scripts/check_disk.sh          # keep an eye on free space
```

## Phase 7 pipeline

```bash
HF_HOME=.cache/hf .venv/bin/python -I scripts/run_pipeline.py \
    --audio data/nptel/wav16k/english_001.wav --ckpt ckpt/colab_main \
    --out results/pipeline/english_001 \
    --reference data/nptel/raw/english_001/english_001.aligned.rsml   # optional WER sanity check
```

Stages (each cached in `--out`, so re-running with other gap settings takes seconds):
Silero VAD → chunks ≤ 28 s → Whisper `large-v3-turbo` (mlx, word timestamps, no prompt) and
wav2vec2 phone recognizer per chunk → expected phones (sw-ms98 lexicon / espeak, pronunciation
variant chosen by what was heard) aligned to recognized phones (`src/pipeline/align.py`; cost =
phone similarity + time mismatch + extra cost for insertions inside a word) → recognized phones
that no word explains, sitting on a word boundary, are **gaps** → BERT tags and restores each gap
→ `restored.txt`, `restored_annotated.txt`, `restored.json` (per-gap phones, times, predictions).
The 32-minute lecture takes ~8 min (almost all Whisper) on an M-series Mac.

Findings on `english_001` (350 gold disfluent tokens; a gap "hits" if it overlaps one in time):

* Of the 350, Whisper already writes ~149 itself (it keeps many "uh"/"ah" and some repetitions).
  104 are overlapped by a gap before the duration filter, 46 after it. For 75 of the 171 hesitations
  the phone recognizer emits no phone in their span at all (Phase 6), so they cannot be found:
  recall is limited by the recognizer, not by BERT.
* Untuned (every insertion run is a gap): 310 gaps, 27% overlap a gold disfluency, 225 of 332
  restored tokens are PARTIAL_WORD guesses, and verbatim WER rises from 0.096 to 0.139.
* Two filters fixed that, both chosen on this same lecture (so treat the numbers as optimistic):
  gaps shorter than 0.15 s are dropped (those under 0.1 s hit gold 13% of the time, longer ones
  35-60%), and PARTIAL_WORD units are detected but not inserted (the model restores the *full*
  word, which is wrong in a verbatim transcript). Result: 139 gaps, 91 restored tokens, 49% on gold,
  WER 0.0954.
* A CTC blank penalty (more phones from the recognizer) was tried at 1, 2, 3: recall rose
  (48 → 69 gold hits) but precision fell faster and WER got worse, so it was not kept.
* Pronunciation-variant selection made no measurable difference on this lecture.

## Data

Raw data is never committed. See `data/MANIFEST.md` for sources and checksums.

* Switchboard (free): https://openslr.org/5/ — `switchboard_word_alignments.tar.gz`,
  `sw-ms98-dict.text` (no license restrictions).
* NPTEL: private Drive folder "BhashaCheck Data" → `en/` (audio + `.rsml`).
