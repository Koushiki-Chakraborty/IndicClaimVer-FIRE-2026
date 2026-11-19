# MuRIL Veracity Classifier

A binary `SUPPORTS` / `REFUTES` classifier shared by both subtasks. It was fine-tuned only on Subtask 1 claim and evidence pairs and reused in Subtask 2 without retraining.

* Base model: `google/muril-base-cased`
* Checkpoint: [nirnit-13/muril-supports-refutes](https://huggingface.co/nirnit-13/muril-supports-refutes)
* Training: class-weighted cross entropy, hyperparameters chosen with Optuna
* Label mapping: `{0: SUPPORTS, 1: REFUTES}`. This is the reverse of `config.LABEL2ID` in the pipelines; both pipelines read labels from the checkpoint's `id2label`.

The pipelines load the checkpoint from the Hugging Face Hub (`MURIL_CLASSIFIER_MODEL` in each `config.py`). Nothing in this folder is needed at run time.

## Contents

```
classifier/
├── notebooks/
│   ├── 00_hyperparameter_search.ipynb   Optuna search
│   ├── 01_train_muril.ipynb             final fine-tuning
│   └── 02_predict_on_kg_top3.ipynb      predictions on the Subtask 1 KG top 3
├── model_selection/
│   ├── notebooks/                       MuRIL, IndicBERTv2, mBERT, XLM-R with the same settings
│   └── results/                         per-seed results
└── tokenizer_config/                    config, tokenizer and training arguments of the checkpoint
```

The notebooks were written for Google Colab and read data from Google Drive. Change the paths in the first cells before running them elsewhere.

## Model selection

All four encoders were fine-tuned with the same settings on Subtask 1 data.

| Model | Dev Macro-F1 |
| ----- | -----------: |
| MuRIL | 0.813 |
| IndicBERTv2 | 0.801 |
| mBERT | 0.730 |
| XLM-R | 0.692 |
