# Switchboard corpus audit — v2

## Scope and counting

Computed from the local ISIP/MS-State transcript archive and independently recounted from the existing project-labelled token file. No audio is included. This is this release's coverage, not a claim about every Switchboard release. Speaker sides are conversation channels, not unique people; unique speakers cannot be established from these files alone.

Occurrences below are **token counts**, not disfluency episode counts. A three-word repeated reparandum contributes three labelled tokens. Structural episode boundaries are not supplied by this release or by the cached primary labels. Each retained token has one primary project label, so overlapping phenomena can be suppressed by label precedence.

## Corpus statistics

| Measure | Count |
|---|---:|
| Conversations | 2,438 |
| Speaker sides | 4,876 |
| Utterance rows | 391,593 |
| Unique utterance IDs | 391,593 |
| Word-alignment rows (including silence/noise) | 4,051,206 |
| Speech tokens (including partial/laughing/mispronounced words) | 3,085,832 |
| Normalized speech vocabulary | 29,452 |
| Retained project tokens (speech + noise) | 3,131,723 |
| Conversation annotation spans | 259.147 hours |
| Speaker-side annotation spans | 518.293 hours |
| Summed speech-token alignment durations | 218.263 hours |

Conversation span = latest aligned end minus earliest aligned start across both channels, summed once per conversation. Side span uses each channel separately. These include internal silences and are transcript-derived coverage estimates, not measured recording lengths. Speech-duration sums can include overlapping speech and alignment artifacts.

## Native transcript annotations

| Parsed annotation | Tokens |
|---|---:|
| WORD | 3,043,167 |
| SILENCE | 919,155 |
| NOISE | 45,891 |
| PARTIAL | 28,529 |
| NOISE_WORD | 13,423 |
| COINAGE | 370 |
| MISPRON | 343 |
| ASIDE | 328 |

Noise and partial-word subtypes (subsets of the counts above):

| Subtype | Tokens |
|---|---:|
| SILENCE:silence | 919,155 |
| NOISE:laughter | 22,209 |
| NOISE:noise | 17,962 |
| NOISE_WORD:laughter | 13,423 |
| NOISE:vocalized-noise | 5,720 |
| PARTIAL:end | 28,050 |
| PARTIAL:start | 460 |
| MISPRON:end | 35 |
| PARTIAL:both | 19 |

`PARTIAL` is a marked word fragment. `MISPRON` is the project's name for `[spoken/intended]` substitutions; the spelling annotation alone does not establish the acoustic cause. Some such tokens also contain truncation. `NOISE_WORD` is speech through laughter/noise. Noise, laughter, silence, coinages, and pronunciation variants are contextual annotations, not automatically disfluencies.

## Lexical hesitation and backchannel forms

These are normalized lexical occurrences, not a contextual judgement that every occurrence is disfluent. Backchannels are often appropriate interactional responses. Zero means no matching spelling in this release; it does not establish absence in the audio. `oh` is shown separately because it is ambiguous and the existing labeler does not tag it as a hesitation.

| Category | Form | Tokens |
|---|---|---:|
| Hesitation candidate | uh | 69,864 |
| Hesitation candidate | um | 21,205 |
| Hesitation candidate | ah | 354 |
| Hesitation candidate | hm | 1,013 |
| Hesitation candidate | huh | 1,883 |
| Hesitation candidate | eh | 16 |
| Hesitation candidate | er | 0 |
| Hesitation candidate | hmm | 0 |
| Hesitation candidate | mm | 0 |
| Hesitation candidate | umm | 0 |
| Hesitation candidate | uhh | 0 |
| Hesitation candidate | ehh | 0 |
| Hesitation candidate | ugh | 22 |
| Hesitation candidate | ooh | 226 |
| Hesitation candidate | ohh | 0 |
| Hesitation candidate | ho | 80 |
| Hesitation candidate | uh-oh | 19 |
| Hesitation candidate | uhh-oh | 0 |
| Hesitation candidate | tsk | 0 |
| Backchannel form | uh-huh | 16,315 |
| Backchannel form | um-hum | 16,016 |
| Backchannel form | huh-uh | 269 |
| Backchannel form | hum-um | 97 |
| Ambiguous interjection | oh | 19,033 |

## Existing project labels

These counts audit the existing v1 labels as a starting point for v2; they are **not newly established gold annotations**. Raw tokens and rounded timestamps were matched on every side before counting. Label correctness has not been manually validated here. A match to raw tokens verifies coverage, not the accuracy of the inferred label.

| Type | Labelled tokens | % of retained tokens | Basis |
|---|---:|---:|---|
| O | 2,631,755 | 84.035% | Unlabelled by the rules; not verified fluent |
| FILLER_DM | 133,671 | 4.268% | Context and pause heuristics |
| FILLER_HESITATION | 94,533 | 3.019% | Lexical inventory; contextual ambiguity remains |
| REPETITION | 86,403 | 2.759% | Sequence-matching heuristic |
| FILLER_BACKCHANNEL | 68,152 | 2.176% | Lexical forms + short-utterance heuristic |
| NOISE | 45,891 | 1.465% | Explicit nonspeech markers; not a disfluency type |
| PARTIAL_WORD | 28,564 | 0.912% | Explicit fragments, including truncated substitutions |
| REPAIR | 17,310 | 0.553% | Near-copy/interruption heuristic |
| PROLONGATION | 16,017 | 0.511% | Word-duration outlier heuristic |
| FALSE_START | 9,119 | 0.291% | Truncation/restart heuristic |
| MISPRONUNCIATION | 308 | 0.010% | Explicit substitutions without truncation |

Repetitions, repairs, false starts, and prolongations have **no native gold count available in this release**. The table supplies the existing project's inferred token counts. Stuttering blocks, acoustically confirmed sound prolongations, individual click/tsk and ugh types, and other untranscribed events cannot be exhaustively counted from this transcript release; unavailable counts must not be reported as zero. Explicit silence intervals are not automatically hesitation pauses.

## Reproduction and checks

Run `.venv/bin/python v2/switchboard_stats.py` from the project root. The archive is streamed without extraction. Existing v1 files are read only.

Checked 4,876 complete A/B sides, identical retained raw tokens/times between archive and cache, and exact agreement with the earlier label summary. Malformed rows: 0; negative-duration word rows: 0; zero-duration word rows: 1172.

Machine-readable outputs: `switchboard_stats.json` (including source SHA-256 hashes), `disfluency_counts.csv`, `raw_annotation_counts.csv`, `lexical_fillers.csv`, and `label_token_forms.csv` (every normalized token form per non-O project label). The source paths and rule/config hashes describe the current workspace; the cached labels' original generation version is not established by those hashes.
