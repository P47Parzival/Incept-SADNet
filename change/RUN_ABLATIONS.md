# Running InceptSADNet Ablation Models

This guide shows how to launch the ablation variants and where their source code and training outputs are stored. Run commands from the repository root:

```powershell
cd "E:\VISHAL PRAJAPATI\research paper\code\Incept-SADNet"
```

## Source-code locations

The original model remains in `models/InceptSADNet.py`. The ablation implementations are separate files under `models/ablations/`:

| Ablation | Python module | Model name used for output folders |
|---|---|---|
| Without SE | `models/ablations/InceptSADNet_no_se.py` | `InceptSADNet_no_SE` |
| Without Transformer | `models/ablations/InceptSADNet_no_transformer.py` | `InceptSADNet_no_Transformer` |
| Without multi-scale temporal convolution | `models/ablations/InceptSADNet_no_multiscale.py` | `InceptSADNet_no_MultiScale` |
| Without learned spatial convolution | `models/ablations/InceptSADNet_no_spatial.py` | `InceptSADNet_no_Spatial` |

`run.py` accepts a dotted Python module path through `--model`. The default is `InceptSADNet`, so `python run.py` still selects the original model.

## Before running

- Activate the Python environment with the repository's dependencies, including PyTorch. See `SETUP_AND_RUN.md` for environment and data setup.
- Run from the repository root because the runner uses paths relative to the current directory.
- Intra-subject training needs each subject's preprocessed `_x.npy` and `_y.npy` files under `data/raw/sXX/`.
- Leave-one-subject-out (LOSO) training needs the prepared arrays under `data/cross/`.
- The runner selects the configured 16 subjects listed in `run.py`.

## Train one ablation

Each command runs one model. Run the ablations separately so each result can be compared with the baseline without combining block removals.

### Intra-subject evaluation

This is the default mode:

```powershell
python run.py --model models.ablations.InceptSADNet_no_se
python run.py --model models.ablations.InceptSADNet_no_transformer
python run.py --model models.ablations.InceptSADNet_no_multiscale
python run.py --model models.ablations.InceptSADNet_no_spatial
```

### Leave-one-subject-out evaluation

Pass `--mode True` to select LOSO mode:

```powershell
python run.py --model models.ablations.InceptSADNet_no_se --mode True
python run.py --model models.ablations.InceptSADNet_no_transformer --mode True
python run.py --model models.ablations.InceptSADNet_no_multiscale --mode True
python run.py --model models.ablations.InceptSADNet_no_spatial --mode True
```

Optional existing runner flags:

- `--resume`: resume from an in-progress checkpoint for that model and evaluation mode. Do not use for a fresh run if you want training to start from the beginning.
- `--compile`: request `torch.compile` where supported by the installed PyTorch.

For example, to resume the no-SE intra-subject experiment:

```powershell
python run.py --model models.ablations.InceptSADNet_no_se --resume
```

## Where training outputs are saved

The model's `Config.model_name` determines the checkpoint folder. The training code in `train_test.py` writes model checkpoints and any active recovery checkpoint to:

```text
data/saved_dict/<model_name>/
```

For example:

```text
data/saved_dict/InceptSADNet_no_SE/
data/saved_dict/InceptSADNet_no_Transformer/
data/saved_dict/InceptSADNet_no_MultiScale/
data/saved_dict/InceptSADNet_no_Spatial/
```

The original model uses `data/saved_dict/InceptSADNet/`, so the ablation checkpoints do not overwrite its checkpoints.

The training code also appends a summary line to a model-specific file in the repository's current working directory:

```text
InceptSADNet_no_SE_result.txt
InceptSADNet_no_Transformer_result.txt
InceptSADNet_no_MultiScale_result.txt
InceptSADNet_no_Spatial_result.txt
```

WandB run files are stored according to the existing WandB configuration. If no `WANDB_API_KEY` is configured, `run.py` prompts for one; pressing Enter selects offline mode.

## Baseline comparison

Run the original model with:

```powershell
python run.py --model InceptSADNet
```

For a controlled comparison, use the same evaluation mode, prepared data/splits, subjects, and training configuration for the baseline and each ablation. The spatial ablation replaces learned electrode-spanning filtering with a specified shape-compatible operation, and the single-scale ablation uses a 31-sample kernel; see the individual notes in this folder for details.

> **Evaluation caution:** The current training loop evaluates test metrics every epoch and uses test F1/AUC to save the best checkpoints. For final paper results, checkpoint selection should be based on validation data, with the test set reserved for final evaluation.
