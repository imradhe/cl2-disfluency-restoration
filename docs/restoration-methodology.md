# Restoration methodology and data contract

The implementation uses transcripts, dictionary phones, existing text alignment times, and optional
text dependency parses. It does not open audio, run ASR, or run a phoneme recognizer.

## Sources and scope

The method is grounded in:

- [CL2_class_project](https://docs.google.com/document/d/1Hq2tNdsbh3rG28jTLDcmvMvbgPwo6QkifkVIXaErbFE/edit).
- [CL2 Project Proposal (Revised)](https://docs.google.com/document/d/1xtpreJPhNfsSNpj6aYadWsE3IoMpHcneUIrysRVeSwA/edit).
- [Dammalapati et al. (2021), *Effects of Duration, Locality, and Surprisal in Speech Disfluency Prediction in English Spontaneous Speech*](https://aclanthology.org/2021.scil-1.9.pdf), particularly §§2–3.

The user subsequently narrowed the implementation to transcripts and modelling, allowed additional
text embeddings, and removed BERT probing. NPTEL transfer, audio processing, and optional Gemma
experiments are outside the active implementation.

The paper predicts fillers/reparandums using surrounding fluent-word features and GLMMs on NXT
Switchboard. The local implementation extends this motivation to word tagging plus insertion slots.
Its corpus, weak labels, local language model, parser and GLMM estimator differ, so its scores are
not a numerical reproduction of the paper.

## Observation/target separation

Each line of input is one JSON object. Minimal labelled example:

```json
{
  "id": "sw2001A:u2:0",
  "conversation": "2001",
  "word_origin": "simulated",
  "phone_source": "simulated_lexicon",
  "label_source": "ms98_project_rules",
  "words": [
    {"text": "i", "canonical": ["ay"], "duration": 0.2},
    {"text": "go", "canonical": ["g", "ow"], "duration": 0.3}
  ],
  "observed_phones": ["ay", "ah", "m", "g", "ow"],
  "heads": [1, -1],
  "word_labels": ["O", "O"],
  "slots": [[], [{"label": "FILLER_HESITATION", "text": "um"}], []]
}
```

`words` and `observed_phones` are observations. `word_labels` and `slots` are targets and are
never used to construct model features or serialized input. Both target fields are omitted for prediction.
There are n word BIO labels and n+1 slots: before each observed word, plus the trailing boundary.
Phone lists must use the same symbol inventory. An empty canonical list explicitly indicates unknown G2P.

`heads` is an optional zero-based dependency-head list; roots are -1. Condition E requires it and
timed-transcript `duration` values in seconds. `--parser-model` adds dependency heads using an installed
spaCy pipeline over the observed tokens. Supplied `syntactic_surprisal` is an optional n-value numeric
list; it must describe the observed transcript and must not be fitted on held-out target annotations.

The preparer removes selected labelled disfluencies from original transcript windows, deriving
auxiliary phones from the original words and optionally adding synthetic phone errors.
It records `phone_source=simulated_lexicon`. Original phones remain privileged simulated evidence,
even when no random noise is added. Words and phone streams are never called audio observations.
The `--max-utterances` option limits examples/windows, not unique source utterance IDs.

## Word labels and insertion decoding

Supported primary types are hesitation, backchannel, discourse-marker filler, repetition, repair,
false start, partial word, prolongation, mispronunciation and noise. These retain their existing
project labels; type names are not re-estimated from phone mismatch alone.

A word head predicts BIO labels on retained words. A slot head predicts a sequence of typed insertions.
The class vocabulary for literal insertion sequences is constructed from the training fold only.
Repetition/repair/false-start units that match nearby observed words are encoded with relative copy
pointers (within four words), allowing lexical generalization. Other units use literal training templates.
Invalid predicted pointers abstain; their number is returned. There is no silently truncated slot length.

Mispronunciations remain supported primary tags. `[spoken](intended)` is the RSML notation used in
the statistics notebook; the restoration model does not invent intended-word corrections or claim
to infer pronunciation from text alone. Source mispronunciations with a cut-off keep their existing
primary `PARTIAL_WORD` label, preserving the earlier overlap accounting.

Output JSON contains restored `text`, `words`, `tags`, `inserted` flags and `observed_index` mappings.
It describes annotations directly rather than emitting an unverified RSML document.

## Linguistic features

For a word with observed dependency head h, integration cost is 1 plus the sum of intervening-word
distances for dependencies completed at that word. Storage cost is 1 plus the number of already-seen
dependents whose heads have not yet appeared. These reproduce the paper's illustrative costs for
“My dog also likes eating sausage”: IC = 1,1,1,2,1,1; SC = 2,2,3,1,1,1.

The local lexical language model uses preceding two-word context and add-0.1 smoothing. It and
word-duration means are fitted only on the outer training fold. Features are extracted for the
preceding/following words around a candidate boundary, including duration, duration/word mean,
and duration/canonical-phone vowel count. These boundaries are appropriate for insertion slots;
this is not the paper's special treatment of words following an annotated repair.

PCFG syntactic surprisal is not computed without a supplied PCFG resource; lexical surprisal is not
substituted under that name. Missing syntactic surprisal has explicit availability flags.
Unavailable parse/timing features cause E to fail with an actionable error.

The GLMM is a one-versus-rest binomial random-intercept model. Numeric features are fixed effects;
context-word identity supplies a Gaussian random intercept. Words observed fewer than 30 times in
the training entity rows share a rare-word group. Marginal likelihood is approximated using 12-point
Gauss–Hermite quadrature; coefficients have mild quadratic regularization. Known-word predictions
use posterior-mean intercepts, while unseen groups have zero random intercept. This differs from
the paper's frequentist lme4 estimation and its exact word-frequency counting convention.

## Shared folds and train-only transformations

Conversation-level grouping holds out both sides of each conversation together. A seeded greedy
allocation balances example counts across ten folds. The manifest records the full data fingerprint,
seed, fold sizes and conversation assignments, and rejects accidental reuse with changed data.

For each fold, lexical vocabulary, TF–IDF, corpus-trained PPMI embeddings, local LM, duration means,
scalers, optional SVD projections, restoration templates, sampling and classifier fitting use only
training examples. External pretrained vectors/encoders are frozen shared resources.

Balancing targets the complete model classes (word BIO class or insertion-template class), whose
imbalance can be more severe than type-only imbalance. SMOTE is run in a training-fitted continuous
space. Singleton classes are replicated and reported. Cluster undersampling picks real rows nearest
training-only centroids. These policies never modify evaluation examples.

BERT balanced variants follow encoder fine-tuning with resampling and a refitted classifier over
frozen training-entity embeddings; this is a two-stage comparison, not end-to-end token-ID SMOTE.

F is optional: each supplied gold/ASR transcript requires a unique `id`, a shared `pair_id`, identical
conversation grouping and `word_origin` equal to `gold` or `asr`. Both F variants train on gold-view
training conversations; evaluation changes only the held-out transcript view. No audio is required.

## Evaluation and interpretation

Exact spans require identical type and complete matching word-aligned token extent. Relaxed spans
require identical type and at least one correctly aligned overlapping token. Maximum bipartite matching
makes matches one-to-one. Incorrect restored words do not receive credit just because their tags match.

Main span scores include disfluencies already visible in the input plus restored material.
Restoration-only scores mask the observed-token tags and measure inserted material alone.
Tag-aware WER is Levenshtein distance over (word, BIO-label) pairs divided by reference word count.
Each fold also exports confusion counts, class sampling diagnostics, and predicted transcripts.

A small-run score only checks the implementation and provides preliminary comparisons. It does not
establish performance for the full corpus or recoverability of omitted speech from phones derived
from the incomplete text itself. Templates unseen in a training fold cannot be generated literally;
relative copy targets reduce that limitation for repetitions and repairs.
