# Colab continuation

The notebook has been created in Dia and saved to the user private Drive folder.
**Training is running on Colab, using an NVIDIA L4 GPU.** CUDA, the full 3,085,832-example corpus, training progress and a 1.33 GB rolling checkpoint in Drive have been verified. The first observed checkpointed run reached step 1,000 / 128,000 training examples at roughly 1,245 examples/second.

The Colab run initializes from pretrained BERT and uses the original ten conversation folds and two epochs for each binary/ten-class model. The earlier Mac checkpoint remains local. The Mac process was no longer running when checked (exit code 137); it was not restarted.

- [Colab notebook](https://colab.research.google.com/drive/1GoAW49oGxYgYWhQbmgBqS-WXo_iYrRpa)
- [Private Drive folder](https://drive.google.com/drive/folders/1xkvZQfcvd0ICbYV4D2wIWFQv3kweup06)

The folder is not shared; its only listed permission is the signed-in user's owner permission. No public sharing or third-party collaborator access is added.

## Transfer payload

| File | Size | Contents |
|---|---:|---|
| project.tar.gz | 97 MB | Full existing Switchboard transcript dataset, compact arrays, source code, corpus counts, original conversation fold map |
| state.pt | 1.33 GB | BERT model and optimizer, gradient scaler, RNG state, shuffled row order and batch offset |
| transfer.json | Small JSON | File sizes, hashes, archive inventory and checkpoint provenance |

The preserved local snapshot contains binary fold 1, epoch 1, step 9,500: 1,216,000 already-processed training examples. This checkpoint has not been transferred to Colab. No sample/window cap is introduced. The corpus has 3,085,832 lexical examples across 2,438 conversations.

The notebook verifies the archive digest, checks CUDA, mounts Drive and starts the fixed two-epoch/ten-fold recipe. Runtime-local arrays avoid repeated Drive reads. Rolling model/optimizer snapshots are copied atomically to Drive in the background; small fold reports are persistent directly. Completed rolling optimizer copies are removed after their fold reports are saved.

The CUDA continuation uses --portable-resume: the actual saved weights/optimizer are restored even if CPU initialization sequences differ between PyTorch versions. Dataset, row assignment, class counts, batch size, epochs, learning rates, seed, balancing and classifier checks remain strict. Cross-device dropout streams and floating-point kernels are not expected to be bit-identical.

No new Colab subscription or compute-unit purchase has been made. The prepared notebook itself contains no corpus rows, checkpoint tensors, account credentials or tokens.


## Live Colab status

- [Live progress JSON](https://drive.google.com/file/d/1RoNT_yAbUdghG80Fnkuno143xrK_Lwca/view)
- [Reports folder](https://drive.google.com/drive/folders/1nfzxa7MxWEuY9ekGBj3P8X05yDv_ImV0)
- [Run metadata](../results/classification/bert_colab/remote_run.json)

The local bert_colab JSON files are verification snapshots. Query the linked Drive progress file for current status; the old local bert_full/progress.json describes the stopped Mac process.
