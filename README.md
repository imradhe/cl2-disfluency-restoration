# Disfluency Restoration (Switchboard → NPTEL)

Restore the disfluencies that ASR drops — fillers, repetitions, repairs, partial
words, prolongations, false starts, mispronunciations — by using **phones** as
evidence.

* **Model:** one BERT classifier with three inputs — the **disfluency-removed transcript**, the
  **G2P phones** of that transcript, and the **phoneme sequence** a phone recognizer hears. For every
  word position it predicts the disfluencies to restore in front of it (e.g.
  `FILLER_HESITATION|FIL:um`, `REPETITION|COPY_R1`); the output is the restored transcript. There is
  no explicit alignment: BERT has to find, in the recognized phones, what the transcript's own phones
  do not explain.
* **Train (text only):** free Switchboard (ISIP/Mississippi State) transcripts. Windows of the fluent
  word sequence (the labelled disfluent tokens removed) + their G2P phones + the phones of
  *everything that was said*, corrupted by a recognizer-error simulator.
* **Test (audio):** `audio → VAD → ASR → transcript → G2P` and `audio → VAD → phone recognizer →
  phoneme sequence`, both into the same BERT.
* **Out-of-domain test:** NPTEL lectures whose `.rsml` segments are all verified.

The full plan lives in the project's planning notes; status is tracked below.

## Status

| Phase | What | State |
|---|---|---|
| 0 | Repo + environment | done (torch 2.15 + MPS, espeak-ng, uv) |
| 1 | Download Switchboard | done (4,876 sides; checksums in `data/MANIFEST.md`) |
| 2 | Parse + label | done (gold + silver rules; spot-check precision still to be measured) |
| 3 | Phoneme layer | lexicon + IPA map + iid noise simulator, plus a recognizer-error profile fitted on `english_001` (`src/phones/channel.py`, `scripts/calibrate_noise.py`, `configs/noise_profile.json`) |
| 4 | Training windows | `scripts/build_seq.py` → `data/seq`: 150k train / 7.2k dev / 5.8k test windows (transcript, G2P, recognized phones, units per slot; 5,064 unit classes). The earlier gap-level datasets (`data/gaps`) are kept |
| 5 | BERT training | sequence model (`src/model/seq.py`, `bert_seq.py`, `train_seq.py`) trained on Colab (L4, 3 epochs, 14,082 steps, ~45 min; `notebooks/train_colab.ipynb`) -> `ckpt/seq_colab`: dev slot F1 0.885, exact-unit F1 0.676. Retrained with the fitted recognizer noise (`--noise-profile configs/noise_profile.json`, same Colab recipe) -> `ckpt/seq_cal`: dev slot F1 0.851 under iid noise, 0.796 under the fitted channel. The gap-level model trained earlier (`ckpt/colab_main`) is superseded |
| 6 | NPTEL ingestion | `english_001` ingested (audio, verified rsml, aligned rsml, recognized phones); other `en/` lectures pending |
| 7 | Inference pipeline | done: `scripts/run_pipeline.py` (see below); needs the trained sequence model |
| 8 | Evaluation | `scripts/eval_nptel_seq.py` (gold-segmented NPTEL + Switchboard dev) and the end-to-end disfluency F1 (`scripts/eval_pipeline_f1.py`) done for `english_001`; results below. The fitted noise closes much of the real-recognizer gap (held-out slot F1 0.46 -> 0.62) and lifts end-to-end disfluency F1 over the first model (0.25 -> 0.31), but not clearly above Whisper alone (0.29) |
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

## Sequence model and pipeline

Training example (`src/model/seq.py`) = a window of 10-24 consecutive transcript words:

```
[CLS] so <s ow> the <dh ax> cat <k ae t> ... [END] [SEP] <s ow ah m dh ax dh ax k ae t ...> [SEP]
       transcript words, each followed by its G2P phones        recognized phones: "so um the the cat"
```

The hidden state of each word's first piece (and of `[END]`) feeds `K=4` classifiers over the
5,064 unit classes (class 0 = nothing) and a word tagger (fluent / prolonged / mispronounced).
Targets come from the labelled Switchboard tokens: `FIL:<word>` for fillers, `COPY_R<k>` /
`COPY_L<k>` pointers into the window's own words for repetitions, repairs and false starts,
`W:<word>`, `DROP` for noise. Training windows keep each fillable disfluent token in the transcript
with probability 0.3, because an ASR (Whisper) writes some of them itself. The recognized phones
are corrupted afresh every epoch (`src/phones/noise.py`); the G2P phones are the lexicon's.

```bash
scripts/build_seq.py                                   # data/seq/*.jsonl.gz, unit_vocab.json
scripts/train_seq.py --max-train 10000 --epochs 2 --save --out ckpt/seq_local10k   # local smoke run (too short to learn anything)
# full run: notebooks/train_colab.ipynb, then scripts/install_colab_ckpt.sh ~/Downloads/main.zip
HF_HOME=.cache/hf .venv/bin/python -I scripts/eval_nptel_seq.py --ckpt ckpt/seq_colab
HF_HOME=.cache/hf .venv/bin/python -I scripts/calibrate_noise.py        # fit configs/noise_profile.json on the first 60% of english_001
HF_HOME=.cache/hf .venv/bin/python -I scripts/eval_nptel_seq.py --ckpt ckpt/seq_cal --from-frac 0.6   # score the held-out last 40%
HF_HOME=.cache/hf .venv/bin/python -I scripts/run_pipeline.py \
    --audio data/nptel/wav16k/english_001.wav --ckpt ckpt/seq_colab \
    --out results/pipeline/english_001 \
    --reference data/nptel/raw/english_001/english_001.aligned.rsml   # optional WER sanity check (not the metric)
.venv/bin/python -I scripts/eval_pipeline_f1.py --dir results/pipeline/english_001_cal_norm --from-frac 0.6 --per-type   # disfluency F1
```

Pipeline stages (`src/pipeline/`, each cached in `--out`): Silero VAD → chunks ≤ 28 s → Whisper
`large-v3-turbo` (mlx, word timestamps, no prompt; its filler spelling "ah" is respelled "uh") and the wav2vec2 phone recognizer per chunk →
windows of ≤ 20 words (cut at the longest pause; the recognized phones are split between windows by
time, which is the only use of time) → BERT → restored tokens → `restored.txt`,
`restored_annotated.txt`, `restored.json`. `--none-bias` trades precision for recall.

**Earlier design (commit `f1c5efa`, removed):** the expected phones of the ASR words were aligned
to the recognized phones, unexplained phones became gaps, and a gap-level BERT restored them. On
`english_001` it only broke even on verbatim WER (the check used at the time). Two facts from that work: Whisper
keeps only a minority of the disfluencies (in its `english_001` transcript: 47 filler words, 45 of them "ah" and
2 "um", against 171 gold hesitations; 15 immediate word repeats against 70 gold repetitions; no partial words.
An earlier estimate of "~149 of 350" matched on text and time overlap, so a single surviving "the" counted as
present, and overcounted), and the phone recognizer emits no phone inside the span of 83 of the 171 hesitations
(75 with a 30 ms tolerance); counting only a written vowel, it misses 94 of the 170 "uh" tokens (55%). That caps
recall for any model that relies on it.

## Results (`ckpt/seq_colab`)

**Switchboard dev** (7,219 windows, simulated recognizer noise), by epoch of the Colab run:

| | ep 1 | ep 2 | ep 3 |
|---|---|---|---|
| slot F1 (is a disfluency in front of this word?) | 0.53 | 0.84 | **0.885** (P 0.93, R 0.84) |
| exact-unit F1 (type and restored token) | 0.35 | 0.62 | **0.676** |
| restoration accuracy when the type is right | 0.885 | 0.938 | **0.941** |

Per-type F1 at epoch 3: FILLER_DM 0.86, FILLER_BACKCHANNEL 0.77, FILLER_HESITATION 0.71, REPETITION 0.68, NOISE 0.56,
PARTIAL_WORD 0.32, REPAIR 0.24, FALSE_START 0.04. The word tagger never predicts PROLONGATION / MISPRONUNCIATION
(F1 0): the inputs carry no durations. Without alignment the model needs far more steps than a local run allows:
the local 10k-window runs stayed at F1 0.000 (no auxiliary labels: 313 steps; with them: 626 steps), and the same
code reached 0.53 after one full Colab epoch (4,694 steps). A 64-window overfit test confirmed the labels and
positions are right (train-set slot F1 0.97). The auxiliary per-phone labels (role, word pointer) are on by default
(`--aux-weight 0.5`); a no-aux run on Colab was not done, so their benefit at scale is untested.

**NPTEL `english_001`** (gold segmentation, 208 windows of 20 words, 171 hesitations / 65 repetitions / 28 DM /
36 false starts / 21 partial words / 11 repairs in gold), slot P / R / F1 at `none_bias` 0:

| recognized-phone stream | slot P / R / F1 | notes |
|---|---|---|
| clean G2P phones of everything said | 0.95 / 0.88 / 0.91 (0.94 at bias 1) | hesitation F1 0.92 |
| G2P + the training noise | 0.84 / 0.70 / 0.76 | hesitation 0.69, repetition 0.50 |
| real wav2vec2 phones (by gold time span) | 0.35 / 0.49 / **0.41** | hesitation recall 0.22, 64 spurious NOISE |

**End-to-end metric: disfluency F1** (`src/pipeline/score.py`, `scripts/eval_pipeline_f1.py`; WER is not used, the
pipeline script only prints it as a sanity check). The pipeline's output transcript is aligned word by word to the
verbatim gold. The system's *disfluent tokens* are every token the model restored (with its type) plus the
disfluencies Whisper already wrote, found by two rules that every system gets alike: lexical fillers (uh, um, ...)
and the first of an immediate repeat. A system token is **correct** if it is aligned to a gold disfluent token with the
same word; precision = correct / system disfluent tokens, recall = correct / gold disfluent tokens ("typed" also needs
the same type). Whisper alone, scored by the same rules, is the baseline. A second score, **restoration F1** (the
model alone), counts only restored tokens, against the gold disfluent tokens that Whisper's words do not contain.
Gold disfluent types: hesitation, backchannel, DM filler, repetition, repair, false start, partial word.

**End to end on `english_001`, first model** (Whisper words + real phones; the whole lecture is a clean test for this
checkpoint; 350 gold disfluent tokens: 171 hesitations, 70 repetitions, 48 false starts, 28 DM, 22 partial words,
11 repairs; Whisper's spelling as written):

| setting | tokens inserted | restoration P / R / F1 | end-to-end F1 |
|---|---|---|---|
| Whisper alone | | | 0.049 |
| `--none-bias 0` (default), everything inserted | 279 | 0.093 / 0.110 / 0.101 | 0.108 |
| ... without PARTIAL_WORD | 204 | 0.128 / 0.110 / 0.118 | 0.122 |
| ... without PARTIAL_WORD, REPETITION | 106 | 0.151 / 0.068 / 0.093 | 0.105 |
| hesitations only | 42 | 0.357 / 0.063 / 0.107 | 0.117 |
| `--none-bias 1` / `2` | 381 / 522 | 0.081 / 0.131 / 0.100 ; 0.065 / 0.143 / 0.090 | 0.107 ; 0.099 |

Only about 1 in 10 restored tokens is correct. The model is strong where its input matches training (dev, clean G2P
phones) and weak on the real recognizer's output, which `PhoneNoise` (iid; 8% deleted / 14% substituted / 4%
inserted per phone) does not resemble. That motivated the calibration below.

## Recognizer-calibrated noise (`ckpt/seq_cal`)

`scripts/calibrate_noise.py` aligns, per utterance, the G2P phones of the gold tokens of `english_001` with the
wav2vec2 phones recognized in the same time span, and fits a channel (`src/phones/channel.py`): whole-word drops by
word length, per-phone deletion / substitution / insertion with a confusion table, stray phones between words, and a
separate channel for the one-phone "uh". `ProfileNoise` replays it token by token on the training windows (70% of the
windows; the other 30% keep the iid noise as a hedge, because the profile comes from one speaker). It is fitted on the
**first 60%** of the lecture; everything below marked *held-out* uses only the **last 40%** (1,747 gold words, 83
windows: 59 hesitations, 33 repetitions, 29 false starts, 16 DM, 9 repairs, 7 partial words).

| measured on the lecture | fit part | held-out | iid defaults |
|---|---|---|---|
| phones substituted (words that were recognized at all) | 21.8% | 21.4% | 14% |
| phones deleted (same) | 4.3% | 4.0% | 8% |
| one-phone words with no phone at all | 7.4% | 1.7% | |
| stray phones after a word | 2.5% | 3.0% | |
| "uh" **not** written as a vowel (nothing, or a neighbour's consonant) | 58.6% (n=111) | 49.2% (n=59) | 8% |

When an "uh" is written it is mostly `ax`, `aa` or `ah`. Replaying the fitted channel on the held-out reference
phones reproduces these statistics (deleted 3.9% vs 4.0%, substituted 21.4% vs 21.4%, "uh" 51.7% vs 49.2%), and at
window level the phone error rate against the clean G2P phones is 28.0% for the noised stream and 28.6% for the real
recognizer. A first version aligned each word against the phones inside its own time span; that books a phone that
the recognizer places a few tens of ms across a word boundary as a deletion in one word and an insertion in the
next, and overstated deletions about 2x and strays about 8x. Alignment per utterance fixed that.

**Held-out slot F1** (`none_bias` 0), the same 83 windows for both checkpoints:

| phone stream | `seq_colab` (iid noise) | `seq_cal` (fitted noise) |
|---|---|---|
| clean G2P phones of everything said | 0.92 | 0.91 |
| G2P + iid noise | 0.74 | 0.77 |
| G2P + the fitted channel | 0.40 | 0.55 |
| **real wav2vec2 phones** | 0.46 (P 0.37 / R 0.59) | **0.62** (P 0.78 / R 0.52) |
| real phones, exact-unit F1 | 0.25 | 0.28 |
| real phones, spurious NOISE predictions | 29 | 2 |
| Switchboard dev, iid noise (1,500 windows) | 0.88 | 0.85 |

The gain on real phones is almost all precision (spurious predictions dropped across types), not recall: "uh"
precision 0.61 -> 0.93 but recall 0.29 -> 0.24, since the recognizer gives the model nothing for about half of them.
FALSE_START is never predicted (0 of 21 gold units), REPAIR F1 0.17, PARTIAL_WORD 0.29.

**Whisper spells its fillers "ah".** In the `english_001` transcript Whisper writes the filler 45 times as "ah",
never "uh"; the gold writes "uh" (170 of 171 hesitations) and never "ah". The pipeline therefore respells "ah" as "uh"
before windowing (`pipeline.asr.normalize_fillers`, on by default, `--no-norm-fillers` turns it off; `asr_raw.txt`
keeps Whisper as written), so the model sees "uh" as in its training transcripts. All end-to-end numbers below use
it; without it Whisper alone scores F1 0.061 on the held-out part, with it 0.292. The mapping rests on this one
lecture, where the gold has no genuine "ah"; in other material "ah" can be a real word.

**End to end, held-out part** (Whisper words + real phones; 153 gold disfluent tokens: 59 hesitations, 33
repetitions, 29 false starts, 16 DM, 9 repairs, 7 partial words):

| system | disfluent tokens output | correct | P | R | **end-to-end F1** | restoration P / R / F1 (model alone) |
|---|---|---|---|---|---|---|
| Whisper alone (rules) | 32 | 27 | 0.844 | 0.176 | 0.292 | |
| `seq_colab` (iid noise) | 141 | 37 | 0.262 | 0.242 | 0.252 | 0.092 / 0.156 / 0.116 |
| `seq_cal`, `--none-bias -1` | 74 | 36 | 0.486 | 0.235 | **0.317** | 0.214 / 0.141 / 0.170 |
| `seq_cal`, `--none-bias 0` (default) | 81 | 36 | 0.444 | 0.235 | 0.308 | 0.184 / 0.141 / 0.159 |
| `seq_cal`, `--none-bias 1` | 94 | 38 | 0.404 | 0.248 | 0.308 | 0.177 / 0.172 / 0.175 |
| `seq_cal`, `--none-bias 2` | 127 | 41 | 0.323 | 0.268 | 0.293 | 0.147 / 0.219 / 0.176 |

By type (`seq_cal`, bias 0): hesitation F1 0.58 (23 of the 34 hesitation tokens are Whisper's own respelled "uh", 22
of them correct; the model restored 11 more, 5 correct), repetition 0.21 (3 of the model's 15 restored tokens
correct), DM filler 0.07, partial word 0, and repair and false start are never predicted. In total the model
restored 49 tokens, 9 correct. 61 of the 153 gold disfluent tokens are already in Whisper's words as ordinary words (mostly DM fillers
such as "so" or "you know") that nothing tags as disfluent; the model only inserts, it does not tag existing words.
Of the 11 hesitations the model restored, 6 are an "uh" or "um" where the gold has none (false positives, not a
scoring artifact). Without respelling: Whisper alone 0.061, `seq_colab` 0.111, `seq_cal` 0.133.

Calibration clearly helps: `seq_cal` beats `seq_colab` by +0.056 end-to-end F1 (paired bootstrap over 20 s blocks of
the held-out part, 95% CI +0.023 to +0.095). Against Whisper alone the gain is +0.017 (CI -0.033 to +0.073), which this
held-out part cannot tell apart from zero. Whole-lecture numbers for `seq_cal` include the part the noise was fitted
on and should not be used.

**Caveats.** The fit and the held-out part are the same lecture and speaker, so this shows generalization within a
lecture, not across lectures or speakers (only one verified lecture is local; the other `.rsml` files have no audio).
153 gold disfluent tokens is a small test set. The rules that tag Whisper's own words are ours, and the Whisper-alone
baseline (and so the end-to-end F1 of every system) depends on them. The noise is also not the only limit: when the
recognizer writes nothing for an "uh", no phone-based model can restore it.

Next: calibrate on more NPTEL lectures once their audio is available, and look for filler evidence that does not
depend on this recognizer (e.g. voiced energy in a stretch with no phone).

## Why the end-to-end F1 is low (`scripts/disfluency_funnel.py`)

Each of the 153 gold disfluent tokens of the held-out part was traced through the pipeline (`seq_cal`, bias 0,
"ah" respelled). The phone recognizers are only one of three places where tokens are lost, and not the biggest:

| type | gold | written by Whisper, tagged by rules | written by Whisper, **not tagged** | restored by the model | missing from the output | missing, but heard by espeak / Telugu / either |
|---|---|---|---|---|---|---|
| hesitation | 59 | 22 | 0 | 5 | 32 | 15 / 24 / 25 |
| DM filler | 16 | 0 | 11 | 1 | 4 | 2 / 2 / 2 |
| repetition | 33 | 3 | 20 | 3 | 7 | 3 / 2 / 3 |
| repair | 9 | 1 | 6 | 0 | 2 | 1 / 1 / 1 |
| false start | 29 | 1 | 24 | 0 | 4 | 4 / 2 / 4 |
| partial word | 7 | 0 | 0 | 0 | 7 | 5 / 5 / 5 |
| **all** | **153** | 27 | **61** | 9 | 56 | 30 / 36 / 40 |

"Heard" = the recognizer's phones in the token's forced-aligned span match at least half of its G2P phones in order
(for the one-phone "uh": any vowel).

1. **40% of the gold disfluent tokens (61) are not dropped at all.** Whisper wrote them as ordinary words: multi-word
   false starts and repairs ("that is", "they are called"), phrase repeats ("from these ... from these", "so so"), DM
   fillers ("you know"). The pipeline can only *insert* tokens; nothing tags a word that is already there, so these
   count as misses however good the model is. Whisper also writes "yeah" or "you know" where the speaker says "uh".
2. **The phone recognizers do lose the "uh"s, but little else.** Of the 32 hesitations missing from the transcript,
   the espeak recognizer heard 15 (47%), the Telugu one 24 (75%), either 25 (78%). Of the 24 missing tokens of the
   other types, espeak heard 15 (62%).
3. **The model recovers few of what the recognizers heard**: 40 of the 56 missing tokens were heard by at least one
   recognizer, 9 were restored. With a perfect gold transcript and clean G2P phones of everything said, the model's own
   slot F1 per type is (real recognizer phones in brackets):

   | type | gold units | perfect inputs | real phones |
   |---|---|---|---|
   | hesitation | 59 | 0.86 | 0.38 |
   | DM filler | 16 | 0.59 | 0.44 |
   | repetition | 29 | 0.42 | 0.39 |
   | repair | 9 | 0.14 | 0.17 |
   | false start | 21 | 0.00 | 0.00 |
   | partial word | 6 | 0.00 | 0.29 |

   Only hesitations are learned well. Even with perfect inputs the model finds no false starts and almost no
   repairs: those are not recognizer failures. One likely contributor, not tested: in Switchboard they are
   heuristic (silver) labels, so their definition may differ from the NPTEL gold.
4. **Precision is low too**: of the 49 tokens the model restored, 9 are correct (6 of 11 restored hesitations are
   an "uh" where the gold has none).

What this implies: better phone evidence can add at most about 25 correct hesitations (the missing "uh"s that
either recognizer heard), taking recall from 0.24 (36 of 153) to about 0.40, and only if the model then restores
every one of them. Tagging words that are already in the transcript is the larger pool (61 tokens), but it needs a
different output (a disfluency tag per word) and a model that finds false starts and repairs, which the current one
does not.

## Other phone recognizers (`scripts/recognizer_shootout.py`)

The wav2vec2/espeak recognizer leaves out about half of the "uh"s (it was trained on phonemized written text). The
lecture's 170 "uh" tokens (111 in the fit part, 59 held-out, spans from the forced alignment) were run through other
CTC recognizers over the same padded segments; each cell is the share of "uh" spans that get a *vowel* token
(fit / held-out), and the last columns show whether the model is just noisy: fluent words with any token, long pauses
(>= 0.4 s between gold tokens) with a vowel token.

| recognizer | writes | "uh" with a vowel | fluent words with a token | pauses with a vowel |
|---|---|---|---|---|
| current: `facebook/wav2vec2-xlsr-53-espeak-cv-ft` | IPA | 47% / 49% | 99% / 100% | 3% / 7% |
| `Harveenchadha/vakyansh-wav2vec2-telugu-tem-100` (378 MB) | Telugu script | 2% / 0% | 70% / 73% | 1% / 2% |
| **`anuragshas/wav2vec2-large-xlsr-53-telugu`** (1.26 GB) | Telugu script | **58% / 68%** | 98% / 99% | 2% / 5% |
| `facebook/mms-1b-all`, adapter `tel` (3.86 GB) | Telugu script | 0% / 0% | 98% / 98% | 0% / 1% |
| `facebook/mms-1b-all`, adapter `eng` | Latin letters | 1% / 2% | 99% / 99% | 1% / 2% |

No Telugu *phoneme* recognizer was found on the Hub: the Telugu models are character-level, and MMS (also character-
level) is trained to write clean orthography, which here means it writes nothing for a filler. Only the Telugu XLSR-53
model hears the "uh"s (usually as `ఆ`). Mapped to Switchboard phones (`src/phones/telugu.py`, coarse: retroflex and
aspirated consonants collapse onto the nearest English phone) and aligned per utterance with the G2P phones
(`scripts/recognizer_phone_quality.py`), it is a much noisier stream for ordinary words:

| phone stream vs G2P (fit / held-out) | current | XLSR-53 Telugu |
|---|---|---|
| "uh" not written as a vowel | 58.6% / 49.2% | **30.6% / 22.0%** |
| phone error on recognized words (deleted + substituted) | **26.0% / 25.5%** | 46.9% / 44.5% |
| stray phones per 100 words | **8.9 / 9.2** | 32.6 / 27.6 |

The two hear different "uh"s: a vowel from at least one of them for 71% / 80% of the spans, from both for 45 / 24.
This is evidence about the recognizers only. A model that reads both streams was then trained
(`train_colab_2streams.ipynb`, `ckpt/seq_2streams`: input `... [SEP] <espeak phones> [REC2] <Telugu phones> [SEP]`,
each stream with its own fitted error profile, `--two-streams --noise-profile2 configs/noise_profile_tel.json`).
On the Switchboard dev windows with simulated noise it improves on the single-stream model (iid noise: slot F1
0.851 -> 0.909, exact-unit 0.632 -> 0.676; fitted-channel noise: 0.796 -> 0.834; hesitation F1 under the fitted
channel 0.426 -> 0.509). In that simulation the second stream is an *independently* corrupted copy of the same clean
phones, which favours a second stream by construction; real recognizer errors are correlated, so the gain on real
phones is expected to be smaller and has not been measured. The Telugu mapping is a first approximation.

## Omitted phonemes on Switchboard (`scripts/swb_omitted_phonemes.py`)

Switchboard has no audio and no phone-level transcription, only word transcripts and the lexicon. Per utterance, the
phones of every spoken token (disfluencies included; exact, or corrupted by a simulated recognizer) are aligned with the
G2P phones of the transcript with the disfluencies ignored. The phones of the first sequence that no G2P phone is
aligned to are the *omitted phonemes*. All 4,876 sides (249,624 utterances, 3.1M tokens); "found" = at least half of a
disfluent token's surviving phones are omitted phonemes.

| phone sequence | omitted phonemes (% of phones) | of the omitted: disfluent / fluent-token / inserted noise | disfluent tokens found |
|---|---|---|---|
| exact | 10.3% | 97.7% / 2.3% / 0% | 98.9% |
| noise del/sub/ins 4/7/2% | 11.4% | 77.8% / 8.5% / 13.7% | 88.7% |
| noise 8/14/4% (training strength) | 12.0% | 66.0% / 13.1% / 20.9% | 79.8% |
| noise 16/28/8% | 11.6% | 53.7% / 21.3% / 25.0% | 61.9% |

By type at 8/14/4% noise (found / absorbed by the G2P): backchannel 98.9 / 1.0, false start 91.1 / 7.8, DM filler
89.1 / 10.1, repair 79.1 / 19.5, hesitation 70.0 / 23.9 (6.0% lost), partial word 68.4 / 27.2, repetition 63.9 / 34.7
(the aligner can pair the transcript's single copy with either copy). The noise is a simulation without whole-token
dropout, so hesitations are optimistic here; repetition, repair and false-start labels are heuristic (silver) in
Switchboard. The same test on the lecture with real recognizer phones is `scripts/omitted_phonemes.py` (sequence
test) and `scripts/phoneme_omission.py` (by gold time spans); results in `results/omitted_phonemes.json` and
`results/phoneme_omission.json`.

## Data

Raw data is never committed. See `data/MANIFEST.md` for sources and checksums.

* Switchboard (free): https://openslr.org/5/ — `switchboard_word_alignments.tar.gz`,
  `sw-ms98-dict.text` (no license restrictions).
* NPTEL: private Drive folder "BhashaCheck Data" → `en/` (audio + `.rsml`).
