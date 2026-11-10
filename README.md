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
| 5 | BERT training | smoke run locally; full run on Colab pending |
| 6 | NPTEL ingestion | todo |
| 7 | Inference pipeline | todo |
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

## Data

Raw data is never committed. See `data/MANIFEST.md` for sources and checksums.

* Switchboard (free): https://openslr.org/5/ — `switchboard_word_alignments.tar.gz`,
  `sw-ms98-dict.text` (no license restrictions).
* NPTEL: private Drive folder "BhashaCheck Data" → `en/` (audio + `.rsml`).
