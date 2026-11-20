# Pretrained BERT: disfluency restoration F1

Fine-tuned `bert-base-uncased` (113,020,348 total parameters) on 151,606 training windows for 3 epochs using NVIDIA L4. Pretrained encoder weights were loaded; restoration heads and new phone-token embeddings were initialized for this task.

Input: flat words, each followed by dictionary-derived pronunciation guidance, plus the full unsegmented spoken-phone sequence in the second BERT segment. Pronunciation guidance is derived only from the flat input, not from gold alignments or missing words. The encoder learns alignment using auxiliary training-only phone-role and word-pointer losses. No gold alignment or disfluency label is an input.

The source dataset and conversation split are identical to the compact baseline. Phones are dictionary-derived; noisy conditions simulate recognizer errors. This does not establish performance on actual recognizer output. Existing project labels include heuristic annotations. Counts are restored tokens, not disfluency episodes.

Epoch and decoder STOP offset are selected exclusively by typed restored-token F1 on a fixed noisy development subset. All held-out gold insertions are scored, including those beyond the eight-token-per-gap decoder capacity. A true positive requires the correct gap, disfluency type, and rendered token; matching is a multiset within each gap. Original insertion order is preserved in saved text predictions but is not enforced by this F1 metric.

Overall disfluency F1 is micro-averaged over inserted tokens; it excludes unchanged fluent words.

## Test results

| Condition | Precision | Recall | Disfluency F1 | Gold disfluent tokens |
|---|---:|---:|---:|---:|
| dev_noisy | 0.7636 | 0.6856 | 0.7225 | 24,031 |
| test_clean | 0.8539 | 0.8025 | 0.8274 | 19,955 |
| test_noisy | 0.7671 | 0.6964 | 0.7300 | 19,955 |
| test_no_phones | 0.0000 | 0.0000 | 0.0000 | 19,955 |

## Per-type disfluency F1

| Type | Gold tokens | Clean-phone F1 | Noisy-phone precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| FALSE_START | 353 | 0.1584 | 0.1313 | 0.1105 | 0.1200 |
| FILLER_BACKCHANNEL | 2,714 | 0.8193 | 0.7525 | 0.7638 | 0.7581 |
| FILLER_DM | 5,901 | 0.9038 | 0.8753 | 0.8539 | 0.8645 |
| FILLER_HESITATION | 4,683 | 0.9271 | 0.7620 | 0.6938 | 0.7263 |
| PARTIAL_WORD | 1,331 | 0.3619 | 0.3687 | 0.1908 | 0.2515 |
| REPAIR | 702 | 0.3820 | 0.3541 | 0.2835 | 0.3149 |
| REPETITION | 4,271 | 0.8538 | 0.8029 | 0.7125 | 0.7550 |

Prolongation and mispronunciation word-tag scores are available separately in JSON. These labels are not inserted-token scores. Phone IDs contain no duration information, and the existing prolongation labels used corpus-wide duration statistics, so acoustic prolongation performance cannot be established here.

Selected epoch: 3. Selected STOP-logit offset: +1.50. Runtime including evaluation: 37.6 minutes. Checkpoint, tokenizer, model configuration, target vocabulary, pronunciation dictionary, full predictions, and development history are included with this report.

## Word-tag disfluency scores

These scores classify existing words and are separate from insertion-restoration F1.

| Type | Test gold tokens | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| PROLONGATION | 770 | 0.0000 | 0.0000 | 0.0000 |
| MISPRONUNCIATION | 17 | 0.0000 | 0.0000 | 0.0000 |

## Comparison with the compact baseline

Both rows below use decoder thresholds selected by noisy-development disfluency F1 and evaluate the same held-out noisy-phone windows. The new run changes both encoder initialization/size and pronunciation guidance; the comparison does not isolate those factors.

| Model | Test disfluency F1 |
|---|---:|
| Compact model trained from scratch | 0.2948 |
| Pretrained BERT with pronunciation guidance | 0.7300 |

Paired F1 difference: +0.4353; 95% conversation-bootstrap CI [+0.4262, +0.4444]. Bootstrap uses 10,000 resamples of whole conversations, seed 2026. It does not capture training-seed or annotation uncertainty.

| Condition | Disfluency F1 | 95% conversation-bootstrap CI |
|---|---:|---:|
| test_clean | 0.8274 | [0.8195, 0.8353] |
| test_noisy | 0.7300 | [0.7217, 0.7385] |
| test_no_phones | 0.0000 | [0.0000, 0.0000] |
