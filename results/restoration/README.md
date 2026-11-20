# Local modelling validation

These are **small engineering runs**, not a full-corpus benchmark. All auxiliary phone
streams were derived from transcript lexicons, with synthetic noise. No audio was used.
The source labels are the project's MS98 rules, including silver categories.

## Main comparison

240 prepared windows from 20 conversations, with 10 shared conversation-held-out folds.
Conditions A–E were evaluated with visible-token rules and multinomial/SVM models over
TF–IDF and training-fold PPMI embeddings: 25 configurations, 250 fold runs.
Condition E used en_core_web_sm 3.8.0 dependency parses and existing transcript durations.

For orientation, the highest mean overall exact-span F1 within each condition is shown below.
Selecting these rows is descriptive; there is no separate model-selection validation set.

| Condition | Model / representation | Overall exact F1 (mean ± SD) | Inserted-material exact F1 |
|---|---|---|---|
| A | multinomial / tfidf | 0.206 ± 0.174 | 0.000 |
| B | svm / tfidf | 0.286 ± 0.144 | 0.000 |
| C | multinomial / tfidf | 0.203 ± 0.151 | 0.000 |
| D | svm / tfidf | 0.346 ± 0.138 | 0.146 |
| E | svm / tfidf | 0.338 ± 0.179 | 0.108 |

A–B use only the incomplete transcript. C–E have simulated auxiliary evidence derived from
the original verbatim text, so their scores do not demonstrate recovery from a lone transcript.
The small A/B baselines inserted no correctly matched missing spans; their overall F1 comes
from tagging disfluencies that remain visible. More data and model training are needed to
assess text-only restoration performance.

[Complete main summary](validation/summary.csv) · [Per-tag results](validation/summary.json)

## Additional checks

- All four balancing policies: 4 configurations × 10 folds on 24 windows.
- Logistic and MLP models with linguistic features: 2 configurations × 10 folds on 24 windows.
- GLMM with linguistic features: 1 configuration × 10 folds on 24 windows; convergence recorded per class.
- Frozen bert-base-uncased embeddings: trained a multinomial classifier and reloaded it for target-free inference.
- Pretrained bert-base-uncased fine-tuning: one epoch on 24 windows, checkpoint reload and target-free inference.
- Tiny random BERT: numeric features, balanced classifier, serialization and checkpoint tests.

The 320 classical fold runs are software validation. They are too small for conclusions about
rare disfluency types or which representation is best. Pretrained encoder checks are not
cross-validated BERT benchmark results.

[Verification metadata](verification.json) contains fold/data fingerprints and run counts.
Every run directory retains per-fold metrics, predictions, confusion counts and sampling diagnostics.
Model checkpoints are local under `ckpt/restoration/` and excluded from Git.
