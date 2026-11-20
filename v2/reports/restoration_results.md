# V2 BERT disfluency restoration results

A fresh, compact BERT encoder (2 layers, hidden size 128, 4,946,620 parameters) was trained for 6 epochs on 151,606 windows from 2,122 Switchboard conversations. All model weights were randomly initialized; the cached BERT WordPiece tokenizer was reused. This is a BERT architecture trained for this task, not fine-tuning of pretrained BERT-base.

## Task and data

Input: `[CLS] flat transcript [END] [SEP] full phone sequence [SEP]`. No word-to-phone boundaries, gold slot/type labels, or auxiliary pointers enter the encoder. The flat transcript has every insertion-type disfluency removed. Phones retain the removed words. The model predicts insertion types and surface words/copy pointers at each transcript gap; rendering those insertions produces a restored transcript. The word tagger also predicts prolongation and mispronunciation labels.

Phone sequences come from the MS98 pronunciation dictionary, not an audio recognizer. 70% of training windows receive simulated phone corruption (8% deletion; 14% substitution, using the existing simulator's mutually exclusive probabilities; 4% insertion). Each noisy evaluation uses a fixed random seed. Clean phones are an oracle condition. Removing phones at test time is an input ablation of the same trained model, not a separately trained text-only baseline.

Train/dev/test split by conversation ID: test 4010–4153; dev 4519–4936; train remainder. Target vocabulary is built from training only. The best epoch is selected using typed restored-token F1 on a fixed random subset of development windows with simulated noise. Test conversations are first evaluated after training and model selection.

| Split | Conversations | Windows | Flat words | Reference words |
|---|---:|---:|---:|---:|
| train | 2,122 | 151,606 | 2,393,333 | 2,787,099 |
| dev | 173 | 8,953 | 140,665 | 164,696 |
| test | 143 | 7,261 | 114,082 | 134,037 |

The default saved model uses the development-WER-selected decoder. A separate F1-selected decoder is reported alongside it; both use the same trained encoder.

## Held-out results

| Condition | Typed restored-token precision | Recall | F1 | Restored WER | Flat baseline WER | Exact windows |
|---|---:|---:|---:|---:|---:|---:|
| dev_noisy | 0.5842 | 0.1288 | 0.2110 | 0.1311 | 0.1461 | 0.2697 |
| test_clean | 0.6091 | 0.1760 | 0.2731 | 0.1276 | 0.1490 | 0.2690 |
| test_noisy | 0.5961 | 0.1229 | 0.2038 | 0.1340 | 0.1490 | 0.2479 |
| test_no_phones | 0.0000 | 0.0000 | 0.0000 | 0.1490 | 0.1490 | 0.2231 |
| test_noisy_f1_decoder | 0.3712 | 0.2445 | 0.2948 | 0.1483 | 0.1490 | 0.2535 |

## Per-type test results with simulated phone noise

| Type | Gold tokens | Predicted tokens | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| FALSE_START | 353 | 1 | 0.0000 | 0.0000 | 0.0000 |
| FILLER_BACKCHANNEL | 2,714 | 1,604 | 0.4570 | 0.2701 | 0.3395 |
| FILLER_DM | 5,901 | 1,917 | 0.7016 | 0.2279 | 0.3441 |
| FILLER_HESITATION | 4,683 | 503 | 0.6481 | 0.0696 | 0.1257 |
| PARTIAL_WORD | 1,331 | 0 | 0.0000 | 0.0000 | 0.0000 |
| REPAIR | 702 | 1 | 0.0000 | 0.0000 | 0.0000 |
| REPETITION | 4,271 | 89 | 0.5506 | 0.0115 | 0.0225 |

Prolongation and mispronunciation word-tag results:

| Type | Gold tokens | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| PROLONGATION | 770 | 0.0000 | 0.0000 | 0.0000 |
| MISPRONUNCIATION | 17 | 0.0000 | 0.0000 | 0.0000 |

## Interpretation and limits

Typed restored-token scoring requires the insertion gap, type, and rendered token to match, using multiset matching within a gap; it does not enforce the ordering of multiple insertions within the same gap. Corpus WER and exact-window match assess the actual restored text, including insertion order. WER aggregates edit counts over all windows, rather than averaging per-window WER. Disfluency F1 counts only inserted disfluencies; it excludes unchanged fluent words.

The decoder supports at most 8 inserted tokens per gap. All gold tokens, including overflow, remain in scoring. Test overflow: 67 tokens. Rare unseen surface targets map to UNK at training/inference but are scored against their actual held-out surfaces. Word-internal prolongation is reported as a tag; phone identities without durations do not identify acoustic lengthening. Mispronunciation tags do not themselves replace a word's surface, so any unresolved spelling substitutions still count as transcript errors.

Training and evaluation use the existing project's labels, including inferred repetitions/repairs/false starts/discourse markers. They measure agreement with those rules, not independently validated human judgements. Existing duration-based labels were derived before this split and used corpus-wide word-duration statistics; prolongation tag results therefore are not a strictly train-only label-generation evaluation. Conversation separation does not guarantee unique-speaker separation because speaker metadata are unavailable here.

Nonspeech noise and laughter are excluded from text restoration; no fabricated noise phones are added. Speech through laughter remains ordinary speech. The reference preserves partial-word fragments with edge hyphens, rather than replacing them with their intended complete words. Oversize windows and sides lacking any flat words are excluded and counted in `data/stats.json`. Window boundaries use input lengths; no phone stream is silently truncated.

Device: cpu; measured run time including final evaluation: 20.25 minutes. Weights: `v2/checkpoints/bert_compact/model.pt`; training history/config and target vocabulary are saved alongside them. Complete predictions and machine-readable results are in `v2/reports/`.

## Decoder calibration

After epoch selection, the NONE/STOP class received a logit offset of +1.5, selected solely by minimum transcript WER on the same fixed development subset. A second decoder, with offset +0.5, was selected by maximum typed restored-token F1 and is reported as `test_noisy_f1_decoder`. The candidate offsets were -1, 0, 0.5, 1, 1.5, 2, and 3. Each offset was frozen before its test evaluation. The default model weights use the minimum-WER decoder; `model_f1_decoder.pt` stores the F1 decoder. Both raw and calibrated evaluations are retained in the JSON results. Calibration does not retrain the encoder.

Per-type restored-token F1 at the two development-selected operating points:

| Type | Test gold tokens | WER-selected decoder F1 | F1-selected decoder F1 |
|---|---:|---:|---:|
| FALSE_START | 353 | 0.0000 | 0.0051 |
| FILLER_BACKCHANNEL | 2,714 | 0.3395 | 0.3708 |
| FILLER_DM | 5,901 | 0.3441 | 0.4161 |
| FILLER_HESITATION | 4,683 | 0.1257 | 0.2430 |
| PARTIAL_WORD | 1,331 | 0.0000 | 0.0103 |
| REPAIR | 702 | 0.0000 | 0.0026 |
| REPETITION | 4,271 | 0.0225 | 0.1332 |

## Paired uncertainty estimates

95% percentile bootstrap intervals resample whole held-out conversations (10,000 resamples, seed 2026). Positive reductions mean lower WER than the unchanged flat transcript. These quantify uncertainty across conversations within this labelled transcript release, not uncertainty from training seeds, annotation errors, or real recognizer output.

| Test condition | WER reduction, percentage points | 95% CI |
|---|---:|---:|
| test_clean | 2.141 | [2.015, 2.268] |
| test_noisy | 1.497 | [1.380, 1.614] |
| test_noisy_f1_decoder | 0.075 | [-0.130, 0.274] |
