# Methods, representations and execution coverage

Implementation coverage and full-corpus experiment coverage are different. The initial full-corpus
report is averaged SGD logistic regression on hashed word/bigram contexts, with an optional timing
control. It is not an experiment over all the methods or embeddings below.

The embedding extension has now completed frozen-versus-fine-tuned GloVe-50 + MLP comparisons on
all 3,085,832 lexical rows, for binary and ten-class tasks, across the same ten folds (40 fold fits).
Final models are refitted on every corpus row and saved locally, including adapted vector tables.
See [GloVe results](../results/classification/glove50/README.md). The full-corpus variational GLMM
comparison has also completed on all ten folds; end-to-end BERT fine-tuning is now running on
the Apple GPU with resumable optimizer/RNG checkpoints. SentenceTransformer comparisons remain
unrun. A matched-GloVe comparison of OvR logistic, multinomial logistic, hinge-loss linear SVM,
MLP and variational GLMM is exported in
[the technique comparison](../results/classification/model_comparison/README.md).

## Methods

| Method | Implemented | Execution before the embedding extension |
|---|---|---|
| Visible-word rules | Yes, restoration pipeline | 240-window pilot |
| One-versus-rest logistic regression | Yes, restoration pipeline | 24-window integration run |
| Multinomial logistic regression | Yes, restoration pipeline | 240-window pilot |
| Linear SVM | Yes, restoration pipeline | 240-window pilot |
| MLP | Yes, restoration pipeline | 24-window integration run |
| Gaussian random-intercept binomial GLMM | Yes, one-versus-rest approximation | 24-window integration run |
| Fine-tuned BERT word/slot classifier | Yes, encoder and head updated jointly | Small integration checks, not a full-corpus benchmark |
| Averaged SGD logistic regression | Yes, full-corpus classification pipeline | All lexical rows; ten grouped folds |

The custom GLMM, local lexical language model and project labels differ from the paper.
An automatically computed PCFG syntactic-surprisal model is **not** implemented: the restoration
pipeline accepts externally computed syntactic surprisal and marks missing values. No BERT probing,
audio processing or Gemma experiment is included in the active implementation.

## Representations

| Representation | Existing support | Task adaptation in the full-corpus embedding runner |
|---|---|---|
| Word/character TF–IDF | Training-fold vocabulary and IDF | No neural parameters; refit per training fold |
| PPMI/SVD | Trained on training-fold contexts | Corpus-trained static baseline; no pretrained encoder fine-tuning |
| Word2Vec/fastText text vectors | Import and frozen pooling | Frozen or update the imported word lookup table |
| GloVe text vectors | Headerless text import | Frozen or update the imported word lookup table |
| Generic Hugging Face transformer | Frozen mean-pooled context features | Frozen or update the complete encoder, including input embeddings |
| SentenceTransformer checkpoint | Frozen sentence features | Frozen or update its complete encoder/pooling model using differentiable forward |
| Pretrained BERT restoration model | Already fine-tuned end to end | Separate word/slot restoration task |
| Random word embeddings | New full-corpus baseline | Frozen random features versus task training from scratch |

Importing Word2Vec/fastText files does not implement those original unsupervised training algorithms.
The static fine-tuning comparison updates the imported word table, including OOV initializations;
it does not reconstruct a fastText model's subword-composition mechanism. Generic encoders must be
compatible with `AutoModel` and expose a token hidden state. Sentence models must expose text
tokenization, a sentence embedding and a transformer input-embedding layer.

## Full-corpus frozen versus updated embeddings

Use `scripts/classify_embeddings.py`. It defaults to every lexical row, all four tasks and all ten
conversation folds. It never reads timing/phone fields as features. It reuses the existing candidate
context construction: the target word is withheld and source labels identify surrounding fluent
words. These are contextual classification experiments, not unknown-position restoration.

`--classifier logistic|multinomial|svm|mlp|glmm` selects an independent sigmoid/OvR linear head,
joint softmax linear head, one-vs-rest hinge-loss linear head, nonlinear MLP, or variational GLMM. All use the same
minibatch training loop. The SVM option is a stochastic hinge-loss fit, not liblinear; these
fixed-budget runs are separate from the sklearn estimators in the restoration pipeline.

The full-corpus GLMM uses frozen GloVe coordinates as fixed effects, plus Gaussian random
intercepts by preceding context word. Training counts determine frequent groups and the pooled
rare group; unseen groups use the population distribution. Factorized Gaussian variational
posteriors are fitted with expected-log-likelihood Gauss-Hermite quadrature, Gaussian KL shrinkage
and a half-normal population-scale prior. This Bayesian approximation is distinct from the
small restoration runner's direct marginal-likelihood GLMM and the paper's lme4 fit.
Run it with `--classifier glmm --modes frozen`; posterior diagnostics accompany each fold.

For the full BERT run and exact resume command, see [the BERT record](../results/classification/bert_full/README.md).
`--training-states`, `--checkpoint-every` and `--resume` restore model, optimizer, scaler, permutation,
batch offset and RNG state within a fold. `--amp` enables float16 scaling on CUDA/MPS.
`--save-last-fold` limits stored encoder copies when disk space is constrained.

For each fold/task, frozen and updated modes have the same initial encoder and MLP head, fixed
epoch count, batch ordering and training-only class weights. No test fold is used to choose epochs
or hyperparameters. A training-only vocabulary maps unseen static words to UNK; transformer
tokenizers retain their external pretrained vocabulary. Every training example participates in
every epoch; there is no window/sample cap.

The updated static model learns lookup weights and an MLP over four positional context vectors.
The updated transformer/sentence model learns the full pretrained encoder and a classification
head. Random initialization is called **training from scratch**, not fine-tuning. Reports record
SHA-256 checks before/after training for encoder and embedding changes, trainable parameter counts,
training-row counts, loss, held-out precision/recall/F1 and confusion matrices.

```bash
# Corpus-trained word embeddings, including an identical frozen-initialization control:
.venv/bin/python scripts/classify_embeddings.py \
  --embedding random --out results/classification/random_embeddings \
  --fold-manifest results/classification/full/settings.json --resume

# GloVe/Word2Vec/fastText text vectors, frozen versus supervised fine-tuning:
.venv/bin/python scripts/classify_embeddings.py \
  --embedding static --vectors /path/to/vectors.txt \
  --out results/classification/static_embeddings \
  --fold-manifest results/classification/full/settings.json --resume

# Full pretrained transformer fine-tuning, not just the classifier head:
.venv/bin/python scripts/classify_embeddings.py \
  --embedding transformer --encoder bert-base-uncased --batch-size 32 \
  --out results/classification/bert_embeddings \
  --fold-manifest results/classification/full/settings.json --resume

# Preserve the sentence checkpoint's own pooling/normalization during task fine-tuning:
.venv/bin/python scripts/classify_embeddings.py \
  --embedding sentence --encoder sentence-transformers/all-MiniLM-L6-v2 \
  --allow-download --batch-size 64 --out results/classification/minilm_embeddings \
  --fold-manifest results/classification/full/settings.json --resume
```

Use `--encoder-lr` independently of `--head-lr`. Defaults are 0.001 for static encoders/heads and
0.00002 for pretrained transformer encoders. `--save-checkpoints` exports complete reloadable
models under ignored `ckpt/`; no checkpoint upload is performed. Downloads are opt-in.
`--fit-final` refits the fixed recipe on all eligible corpus rows after cross-validation is complete
and saves usable final models and adapted embeddings. Those refits receive no training-set
performance claims; the separate out-of-fold report remains the evaluation.
`--fold-ids` selects full-sized folds for staged execution; reports are explicitly incomplete
until every requested mode/task has all ten folds. `--resume` reuses only matching settings.
The default random/static batch size is 1,024; use the smaller examples above for transformers.

Implementation references: [PyTorch embeddings](https://docs.pytorch.org/docs/2.15/generated/torch.nn.Embedding.html),
[Transformers fine-tuning](https://huggingface.co/docs/transformers/training),
[SentenceTransformer training](https://www.sbert.net/docs/sentence_transformer/training_overview.html).
