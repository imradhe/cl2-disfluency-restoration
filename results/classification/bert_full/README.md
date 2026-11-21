> Training has moved to Colab (NVIDIA L4). The Mac process is no longer active. See [Colab run metadata](../bert_colab/remote_run.json) and [live Drive progress](https://drive.google.com/file/d/1RoNT_yAbUdghG80Fnkuno143xrK_Lwca/view). The description below records the earlier Mac run.

# Full-corpus fine-tuned BERT run

Training has started on the local Apple GPU. **No held-out BERT result is available yet.**
Use [progress.json](progress.json) for the latest completed training-batch progress, and
[launch.json](launch.json) for the launch record. The log is `/tmp/bert-full.log`.

The experiment uses all 3,085,832 lexical examples from 2,438 conversations and exactly the same
ten conversation-separated folds as the full-corpus logistic/GloVe comparisons. Binary and ten-class
models are trained separately. Each receives two complete passes over its outer training fold.
No sample/window limit is applied. The target word is withheld, with four fluent context words and
a mask marking the known candidate. Source annotations identify fluent contexts. There are no
timing, phone or audio features. This is contextual classification, not unknown-position restoration.

The pretrained `bert-base-uncased` encoder and a 64-unit MLP head are updated jointly. The encoder
learning rate is 0.00002 and head learning rate 0.001; loss uses training-fold inverse-frequency
class weights. Batches contain 128 rows, with float16 autocast/gradient scaling and float32 model
parameters. No outer test fold selects epochs, hyperparameters or thresholds.

Measured local throughput projects roughly 5–7 days for all 20 fold/task fits, before interruptions
or significant throttling. This is a compute requirement, not a claim that the corpus is too small.
The Mac GPU is available outside the tool sandbox; the earlier sandbox-only check was insufficient.

## Checkpoints and resume

An atomic checkpoint is saved after the first training step, every 500 steps, and each epoch.
It preserves model, optimizer, gradient scaler, RNG state, batch permutation/offset and loss history.
Resume validates the training recipe and initialization. CPU interruption tests reproduce uninterrupted
weights exactly; GPU kernels may have their usual numerical nondeterminism.

Completed fold metrics are saved before moving to the next task/fold. Completed optimizer states are
removed after the report is written, limiting disk usage. Final-fold model checkpoints are retained
for each task; earlier completed folds retain metrics rather than large encoder copies.

Run the following command from the repository root in a normal Mac terminal, or outside the
agent sandbox so MPS is available. `--resume` continues both an interrupted training batch sequence
and completed fold reports. Do not run a second copy while the current process is active.

```bash
.venv/bin/python scripts/classify_embeddings.py \
  --embedding transformer --encoder bert-base-uncased --classifier mlp \
  --modes updated --tasks binary multiclass \
  --fold-manifest results/classification/full/settings.json \
  --epochs 2 --batch-size 128 --hidden 64 \
  --head-lr .001 --encoder-lr .00002 --device mps --amp --threads 4 \
  --out results/classification/bert_full \
  --training-states ckpt/bert_full_training \
  --checkpoint-every 500 --progress-every 25 \
  --save-last-fold --checkpoints ckpt/classification_bert_full --resume
```

No paid cloud GPU was started. Another training machine can use the same prepared corpus/fold
manifest and code with `--device cuda`; use a separate output directory if changing the batch size
or training recipe. A portable in-progress checkpoint includes optimizer state; copying only an
encoder checkpoint does not preserve the exact training continuation.
