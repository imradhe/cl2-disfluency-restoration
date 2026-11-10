# Data manifest

Raw data is **not** committed (`data/raw/` is gitignored). Everything below is reproducible with the scripts named.

## Switchboard transcripts (free ISIP / Mississippi State release, no audio)

| Item | Value |
|---|---|
| Source | OpenSLR SLR5, https://openslr.org/5/ (mirror of isip.piconepress.com/projects/switchboard, release 01/29/03 "manually corrected word alignments") |
| License | none ("released without any license restrictions" per OpenSLR) |
| Files | `switchboard_word_alignments.tar.gz` (49,651,293 B), `sw-ms98-dict.text` (902,296 B) |
| SHA-256 tar | `7bbc4dd5edc82b000078a028cc0a7f705e764da17cdf5027fe7888907578b106` |
| SHA-256 lexicon | `c5283e1d11127af443230f120a4f77dd7a76081e276578957975432db82a340c` |
| Contents | 4,876 speaker sides (2,438 conversations): `*-ms98-a-trans.text` (utterance level) and `*-ms98-a-word.text` (word level with `[silence]`), plus the lexicon (also inside the tarball, byte-identical) |
| Location | `data/raw/switchboard_ms98/` (tarball + lexicon) and `.../extracted/swb_ms98_transcriptions/` |

Download note: the OpenSLR host served ~60 KB/s (≈14 min); the elda.org / magicdatatech mirrors timed out or returned 504 on 2026-11-10.

### What the markup contains (verified on the data)

* Partial words `an[y]-`, `i['m]-`, `-[a]b[out]-`; the lexicon gives the pronunciation of the **spoken** part only.
* Mispronunciations `[spoken/intended]`, possibly nested with partial words (`[cam[pany]-/company]`).
* `[laughter] [noise] [vocalized-noise] [silence]`, `[laughter-word]`, `{coinage}`, `word_1` variants.
* Lexicon: 39,038 entries, 38,888 distinct keys, **42 phones** (ARPAbet-style lower case, includes `ax en el`, no stress), reduced forms (`um` → `ah m`).
* Not marked: repetitions, repairs, false starts, prolongations, clicks (`tsk`), `ugh`/`ooh`.

### Derived data (scripts/build_processed.py, scripts/build_gaps.py)

| File | Content |
|---|---|
| `data/processed/swb_tokens.jsonl.gz` | per-side labelled tokens + phones (3,131,723 tokens) |
| `data/processed/label_stats.json` | counts per label / token kind, lexicon coverage (100% on every kind) |
| `data/processed/spotcheck/<LABEL>.txt` | 40 random context windows per label for manual precision checks |
| `data/gaps/{train,dev,test}.jsonl.gz` | gap + word-span examples (see `src/model/gaps.py`) |
| `data/gaps/restoration_vocab.json` | classes of the restoration head (built from train only) |

Split by conversation id (Charniak & Johnson convention): test = sw4010–4153, dev = sw4519–4936, train = rest.

## NPTEL lectures (out-of-domain test)

Private Google Drive folder "BhashaCheck Data" → `en/` (`.wav` + `.rsml`). Not yet ingested (Phase 6).
Selection rule (decided with the project owner): keep `.rsml` files whose segments are **all** flagged verified.
