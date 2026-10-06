# Ablation 2 — Without Transformer Encoder

## Purpose

This experiment tests the contribution of the Transformer encoder after convolutional feature extraction.

## Implementation

The variant is implemented in `models/ablations/InceptSADNet_no_transformer.py`.

- It inherits the original model and configuration from `models/InceptSADNet.py`.
- It replaces `model.transformer` with `torch.nn.Identity()`.
- The convolutional embedding still produces the sequence of embedded EEG features; this sequence is passed unchanged to the classification head.
- The multi-scale temporal branches, spatial convolution, SE block, channel projection, and classification head remain enabled.
- The original `models/InceptSADNet.py` is not edited.
- The variant's `model_name` is `InceptSADNet_no_Transformer`, keeping checkpoints/results separate.

This tests removal of Transformer processing, not removal of the sequence itself. The classifier still receives the flattened convolutional sequence.

## Run

Run from the repository root after activating the project's Python environment and preparing the data described in `SETUP_AND_RUN.md`.

Intra-subject mode (the default):

```powershell
python run.py --model models.ablations.InceptSADNet_no_transformer
```

Leave-one-subject-out mode:

```powershell
python run.py --model models.ablations.InceptSADNet_no_transformer --mode True
```

Add `--resume` to resume an interrupted run, or `--compile` to request `torch.compile` where supported.

## Expected outputs

Model checkpoints are written under:

```text
data/saved_dict/InceptSADNet_no_Transformer/
```

The training code also appends a summary to `InceptSADNet_no_Transformer_result.txt` in the repository's current working directory. WandB behavior follows the existing runner configuration.

## Comparison protocol

Compare against the full model using the same evaluation mode, subjects, preprocessing, prepared data/splits, training schedule, and hyperparameters. Record accuracy, weighted F1, and AUC for the same subjects. Run each model independently; do not combine this change with another ablation when estimating the Transformer's individual contribution.

## Important evaluation caveat

The existing `train_test.py` selects best checkpoints using test-set F1/AUC observed at every epoch. That is test-set model selection. For a paper-quality result, select checkpoints with validation data and evaluate on the test set only after training/model selection. This ablation does not alter that procedure.
