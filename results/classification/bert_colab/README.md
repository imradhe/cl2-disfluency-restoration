# BERT training on Colab

The full-corpus BERT experiment is running on an NVIDIA L4 GPU. All 3,085,832 lexical examples, the original ten conversation-separated folds, two epochs, and separate binary/ten-class tasks are retained.

The verified live run reached step 1,000 and 128,000 processed training examples; its rolling model/optimizer checkpoint is saved in Drive. Local JSON files are snapshots, not live counters.

- [Colab notebook](https://colab.research.google.com/drive/1GoAW49oGxYgYWhQbmgBqS-WXo_iYrRpa)
- [Live progress](https://drive.google.com/file/d/1RoNT_yAbUdghG80Fnkuno143xrK_Lwca/view)
- [Reports and final results](https://drive.google.com/drive/folders/1nfzxa7MxWEuY9ekGBj3P8X05yDv_ImV0)

The run starts each missing fold from pretrained BERT. It resumes its own Colab checkpoints if restarted. The interrupted Mac run and its first binary-fold score are preserved separately; they are not combined into the Colab cross-validation average.
