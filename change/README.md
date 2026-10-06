# InceptSADNet Ablation Changes

This folder records the ablation variants added alongside the original model. The full model in `models/InceptSADNet.py` remains unchanged.

## What was changed

### Model variants

Four separate implementations live under `models/ablations/`:

| Experiment | Module | Change from full model | Config/model name |
|---|---|---|---|
| Without SE | `InceptSADNet_no_se` | Replaces the SE block with `nn.Identity()` | `InceptSADNet_no_SE` |
| Without Transformer | `InceptSADNet_no_transformer` | Replaces the Transformer encoder with `nn.Identity()` | `InceptSADNet_no_Transformer` |
| Without multi-scale | `InceptSADNet_no_multiscale` | Replaces kernels 15/31/63 with one kernel-31 branch and 24 filters | `InceptSADNet_no_MultiScale` |
| Without learned spatial convolution | `InceptSADNet_no_spatial` | Replaces the learned electrode-spanning convolution with per-channel expansion and fixed electrode mean | `InceptSADNet_no_Spatial` |

Each variant subclasses/reuses the original model components as appropriate. Each has a separate model name so saved weights and training summaries are not placed in the original model's output folder/name. These variants are independent: run one at a time against the baseline for single-block comparisons.

### Runner

`run.py` now uses the `--model` argument to import the selected module. The default is `InceptSADNet`, so invoking `python run.py` continues to choose the original model. A short module name such as `InceptSADNet` is interpreted as `models.InceptSADNet`; a dotted module path such as `models.ablations.InceptSADNet_no_se` is imported as written.

The runner continues to support the existing `--mode`, `--resume`, and `--compile` options. It does not automatically train all four ablations in one invocation.

## Before running

1. Activate the project's configured Python environment and install its dependencies, including PyTorch. See `SETUP_AND_RUN.md`.
2. Prepare the input data and required split files. Intra-subject mode reads each subject's preprocessed `_x.npy` and `_y.npy` files under `data/raw/sXX/`. Leave-one-subject-out mode reads prebuilt cross-subject arrays under `data/cross/`.
3. Run from the repository root, since paths such as `data`, model result text files, and WandB's local output are relative to the current directory.
4. The runner filters the subjects to `s01`, `s05`, `s09`, `s13`, `s22`, `s31`, `s35`, `s40`, `s41`, `s42`, `s43`, `s44`, `s45`, `s49`, `s50`, and `s53`.

## Run commands

### Original model

The default selects the baseline:

```powershell
python run.py
```

Explicitly selecting it is also supported:

```powershell
python run.py --model InceptSADNet
```

### One ablation at a time

Default intra-subject experiment:

```powershell
python run.py --model models.ablations.InceptSADNet_no_se
python run.py --model models.ablations.InceptSADNet_no_transformer
python run.py --model models.ablations.InceptSADNet_no_multiscale
python run.py --model models.ablations.InceptSADNet_no_spatial
```

Leave-one-subject-out experiment (the existing runner checks for the exact string `True`):

```powershell
python run.py --model models.ablations.InceptSADNet_no_se --mode True
python run.py --model models.ablations.InceptSADNet_no_transformer --mode True
python run.py --model models.ablations.InceptSADNet_no_multiscale --mode True
python run.py --model models.ablations.InceptSADNet_no_spatial --mode True
```

Add `--resume` to resume a matching in-progress checkpoint. Add `--compile` to request `torch.compile` when supported by the installed PyTorch. Do not add `--resume` to a fresh run unless you intend to use the checkpoints/results already present for that model name.

## Where outputs go

For each model, checkpoint files are saved beneath:

```text
data/saved_dict/<model_name>/
```

Examples include `data/saved_dict/InceptSADNet/` for the original and `data/saved_dict/InceptSADNet_no_SE/` for no-SE. The training routine also appends a result summary to `<model_name>_result.txt` in the current working directory. The runner prompts for a WandB API key if none is configured; pressing Enter switches it to offline mode.

## Fair comparison checklist

- Compare each ablation to the full model independently.
- Use the same subjects, train/dev/test examples, preprocessing, evaluation mode, and hyperparameters.
- Record the selected checkpoint's test accuracy, weighted F1, and AUC consistently for every variant.
- Keep logs of model name, run command, seed, and any settings changed.
- Repeat with multiple seeds if reporting variability.
- Document that the multi-scale comparison uses a single 31-sample kernel.
- Document that the spatial comparison uses fixed electrode averaging after channel expansion; it is not a literal no-aggregation model.

## Evaluation limitation that must be addressed

In the current `train_test.py`, `test_model(...)` is called every epoch and the best F1/AUC checkpoints are selected from those test metrics. Thus, the test set is used for model selection. A fair scientific final result should select the checkpoint using validation data and evaluate on the test set only after model selection. The ablation code and runner changes described here do not fix this pre-existing behavior; do not present current best-test scores as an untouched final holdout evaluation.

## Individual implementation notes

- [RUN_ABLATIONS.md](./RUN_ABLATIONS.md) — commands, prerequisites, and output locations
- [01_without_se.md](./01_without_se.md)
- [02_without_transformer.md](./02_without_transformer.md)
- [03_without_multiscale.md](./03_without_multiscale.md)
- [04_without_spatial.md](./04_without_spatial.md)
