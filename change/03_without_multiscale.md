# Ablation 3 — Without Multi-Scale Temporal Convolution

## Purpose

This experiment compares the original parallel temporal-convolution branches with a single temporal receptive field.

## Implementation

The variant is implemented in `models/ablations/InceptSADNet_no_multiscale.py`.

- It inherits the original `ConvEmbedding` and replaces its three temporal branches (kernel sizes 15, 31, and 63) with one branch.
- The replacement branch uses a temporal kernel size of 31, 24 output filters, batch normalization, and ELU.
- The original branches provide 8 filters each (24 total), so the replacement retains 24 temporal feature channels before the downstream spatial convolution.
- The downstream spatial convolution, SE block, channel projection, Transformer, and classifier remain enabled.
- The original `models/InceptSADNet.py` is not edited.
- The variant's `model_name` is `InceptSADNet_no_MultiScale`, keeping checkpoints/results separate.

This is specifically a single-scale 31-sample-kernel comparison. It does not show that every possible single kernel would perform the same.

## Run

Run from the repository root after activating the project's Python environment and preparing the data described in `SETUP_AND_RUN.md`.

Intra-subject mode (the default):

```powershell
python run.py --model models.ablations.InceptSADNet_no_multiscale
```

Leave-one-subject-out mode:

```powershell
python run.py --model models.ablations.InceptSADNet_no_multiscale --mode True
```

Add `--resume` to resume an interrupted run, or `--compile` to request `torch.compile` where supported.

## Expected outputs

Model checkpoints are written under:

```text
data/saved_dict/InceptSADNet_no_MultiScale/
```

The training code also appends a summary to `InceptSADNet_no_MultiScale_result.txt` in the repository's current working directory. WandB behavior follows the existing runner configuration.

## Comparison protocol

Compare this variant with the full model under the same evaluation mode, subjects, preprocessing, prepared data/splits, training schedule, and hyperparameters. Record accuracy, weighted F1, and AUC for the same subjects. Do not disable another block in this run.

## Important evaluation caveat

The existing `train_test.py` chooses best checkpoints using test-set F1/AUC collected every epoch. That leaks test information into model selection. For a paper-quality comparison, select checkpoints with validation data, then evaluate the selected checkpoint on the test set once. This ablation does not alter the training/evaluation procedure.
