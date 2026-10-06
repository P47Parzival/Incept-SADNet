# Ablation 4 — Without Learned Electrode-Spatial Convolution

## Purpose

This experiment tests the effect of learned spatial filtering across the electrode axis in the original embedding.

## Implementation

The variant is implemented in `models/ablations/InceptSADNet_no_spatial.py`.

- It replaces the original grouped convolution whose kernel spans all 30 electrodes.
- A grouped 1×1 convolution expands each of the 24 temporal feature channels to two channels without mixing information between electrode positions.
- Batch normalization and ELU are applied, followed by a mean across the electrode axis, average pooling, and dropout.
- The existing SE block and subsequent channel projection, Transformer, and classifier remain enabled.
- The original `models/InceptSADNet.py` is not edited.
- The variant's `model_name` is `InceptSADNet_no_Spatial`, keeping checkpoints/results separate.

This is a replacement-based ablation, not a literal deletion: downstream layers require a compatible tensor shape. In particular, electrode averaging still aggregates electrodes, but it is fixed averaging rather than learned spatial filtering. State this exact replacement when describing the experiment.

## Run

Run from the repository root after activating the project's Python environment and preparing the data described in `SETUP_AND_RUN.md`.

Intra-subject mode (the default):

```powershell
python run.py --model models.ablations.InceptSADNet_no_spatial
```

Leave-one-subject-out mode:

```powershell
python run.py --model models.ablations.InceptSADNet_no_spatial --mode True
```

Add `--resume` to resume an interrupted run, or `--compile` to request `torch.compile` where supported.

## Expected outputs

Model checkpoints are written under:

```text
data/saved_dict/InceptSADNet_no_Spatial/
```

The training code also appends a summary to `InceptSADNet_no_Spatial_result.txt` in the repository's current working directory. WandB behavior follows the existing runner configuration.

## Comparison protocol

Compare with the full model using the same evaluation mode, subjects, preprocessing, prepared data/splits, training schedule, and hyperparameters. Record accuracy, weighted F1, and AUC for the same subjects. Run this by itself to estimate the effect of learned electrode-spatial filtering.

## Important evaluation caveat

The current training loop picks best checkpoints using test-set F1/AUC observed every epoch. This uses test data in model selection. For paper-quality reporting, select checkpoints on validation data and reserve test data for final evaluation. This ablation does not change the existing training/evaluation procedure.
