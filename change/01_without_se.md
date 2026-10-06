# Ablation 1 — Without Squeeze-and-Excitation (SE)

## Purpose

This experiment asks whether the SE channel-recalibration operation contributes to performance, while retaining the rest of InceptSADNet.

## Implementation

The variant is implemented in `models/ablations/InceptSADNet_no_se.py`.

- It inherits the original model and configuration from `models/InceptSADNet.py`.
- It replaces `model.conv.se_block` with `torch.nn.Identity()`.
- The SE operation normally scales feature channels using weights learned from globally pooled activations. In this variant, features pass directly to the next projection layer without that scaling.
- No sequence, channel, or classifier dimensions are intentionally changed.
- The multi-scale branches, spatial convolution, projection, Transformer, and classification head remain enabled.
- The original `models/InceptSADNet.py` is not edited.
- The variant's `model_name` is `InceptSADNet_no_SE`, which keeps its saved outputs separate from the baseline.

## Run

Run commands from the repository root after activating the project's Python environment and preparing the data described in `SETUP_AND_RUN.md`.

Intra-subject mode (the default):

```powershell
python run.py --model models.ablations.InceptSADNet_no_se
```

Leave-one-subject-out mode:

```powershell
python run.py --model models.ablations.InceptSADNet_no_se --mode True
```

To resume an interrupted run, add `--resume`. To request `torch.compile` where supported, add `--compile`.

## Expected outputs

Model checkpoints are written under:

```text
data/saved_dict/InceptSADNet_no_SE/
```

The training code also appends a summary to `InceptSADNet_no_SE_result.txt` in the repository's current working directory. WandB behavior follows the existing runner configuration: it prompts for a key if one is not already configured, otherwise it uses offline mode.

## Comparison protocol

Compare this model with the full InceptSADNet using the same evaluation mode, subject list, input preprocessing, prepared data/splits, number of epochs, optimizer, learning rate, batch size, and other settings. Run baseline and ablation as separate experiments; do not disable SE in an already modified ablation.

Record accuracy, weighted F1, and AUC for the same subjects. Use repeated seeds if the study requires uncertainty estimates.

## Important evaluation caveat

The current `train_test.py` evaluates the test set every epoch and saves the best checkpoints according to test F1/AUC. That uses test results for model selection and can make reported test performance optimistic. For a paper-quality ablation, change the training/evaluation procedure so checkpoint selection uses validation data and reserve the test set for final evaluation. This ablation implementation does not change that training behavior.
