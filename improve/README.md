# Evaluation Protocol Improvements

This document explains the changes on the `improved-evaluation-protocol` branch. The goal is to make evaluation consistent and prevent the test set from influencing checkpoint selection before producing new baseline or ablation results.

## Quick summary

The branch changes the training/evaluation support code in `run.py`, `utils.py`, `train_test.py`, `make_datasets.py`, and `dataset_contract.py`. It also adds synthetic tests under `tests/` and implementation notes in this folder.

The original network implementation and its four ablation implementations are not changed. Model architecture, optimizer, learning rate, batch size, epoch count, dropout settings, and split seed remain as before. A follow-up class-aware split change is documented below; it changes intra-subject split membership relative to older runs, so all baseline and ablation models must use the same new split.

| Change | What it addresses |
|---|---|
| Consistent train/dev/test ordering | Prevents the runner from assigning the wrong arrays/loaders to dev and test, including in LOSO. |
| Complete, ordered evaluation loaders | Ensures dev/test evaluation includes final partial batches and sees examples in a stable order. |
| Dev-selected checkpoint; one final test evaluation | Prevents test metrics from choosing the reported model checkpoint. |
| Per-subject seed and seed-aware paths | Makes a subject's initialization reproducible and separates non-default seed outputs. |
| Class-aware intra-subject splitting | Uses every sample for unequal class counts instead of selecting/truncating samples based on class 0. |
| Training-only oversampling | Balances training classes after splitting; dev/test examples are never duplicated. |
| Explicit Xavier initialization | Applies the documented initializer to Conv/Linear layers for every run while preserving normalization-layer defaults. |
| Cohort and EEG input contract | Fails on incomplete subject directories, missing/ambiguous EEG channels, incorrect epoch dimensions, or inconsistent labels. |
| Shared response-time label mapping | Applies the same documented threshold-to-class mapping in both preprocessing label paths. |
| Explicit undefined-AUC error | Stops when weighted one-vs-rest AUC cannot be computed instead of writing a fallback score. |
| Synthetic protocol and model tests | Checks data routing, evaluation coverage, checkpoint selection, seed-aware skipping, and five model forward paths. |

## Why these changes were needed

Before these changes, the train/dev/test split ordering was inconsistent across the split function, loader builder, and runner. This made LOSO use the held-out subject incorrectly for evaluation, and the standalone all-subject evaluator could receive dev/test in the opposite order.

The training loop also measured test F1/AUC at every epoch and saved whichever checkpoint scored best on those test metrics. That uses the test set for model selection and can bias the final reported test result.

Finally, dev/test DataLoaders shuffled and dropped their final partial batches. As a result, evaluation could use a different subset/order between passes and omit some examples. The previous intra-subject splitter also based all class slices on the class-0 count, which could omit samples or select the wrong rows when class counts were unequal.

## What changed

### 1. Loader ordering

`split_data` returns train, dev, test. The loader-order correction made its result names and runner call sites consistent; the later class-aware update changed how intra-subject sample membership is created while retaining the fixed split seed 27.

- `utils.build_datasets` now assigns those outputs as `train_data`, `dev_data`, and `test_data`, then builds loaders with the same ordering.
- The intra-subject and LOSO call sites in `run.py` unpack loaders as `train_iter`, `dev_iter`, and `test_iter`.
- `build_cross_datasets` and `evaluate_all_subjects.py` already used train/dev/test order, so no change was needed there.
- For intra-subject data, the corrected construction preserves the split rows previously reaching the dev/test roles through the old double-swap, while subsequent evaluation now includes any last partial batch.

### 2. Class-aware intra-subject splitting

The later update in `utils.split_data` independently shuffles the indices for each label with fixed random state 27, then assigns approximately 80%/10%/10% of each present class to train/dev/test. It retains all samples even when class counts differ; it no longer assumes class 0 is the smallest or truncates other classes. A present class must have at least three samples so each split gets at least one; otherwise, the function raises a descriptive `ValueError`.

This deliberately changes the split membership from historical runs. Use the updated function for the baseline and every ablation, and do not mix old results with new results.

### 3. Complete and deterministic validation/test evaluation

- Dev/test loaders use `shuffle=False` and `drop_last=False`.
- Training loaders retain their prior shuffle/drop behavior.
- Evaluation loss targets are flattened in a way that handles a final batch containing one example.
- Batch logits are concatenated, so evaluation works when the final batch is smaller than the configured batch size.

### 4. Checkpoint selection and final test

At each epoch, the loop calculates validation loss and dev F1/AUC. It saves the best-F1 and best-AUC checkpoints based on **dev** metrics. When training ends, it reloads the best-dev-F1 checkpoint and computes test metrics once. Test F1/AUC therefore refer to the same selected checkpoint.

Each completed subject/run appends one line to `<model_name>_result.txt`:

```text
subject:{id}, mode:{mode}, seed:{seed}, best_epoch:{epoch}, best_dev_f1:{score}, test_acc:{score}, best_test_f1:{score}, best_test_auc:{score}
```

`best_test_auc` remains the final field for compatibility with the existing parser in `evaluate_all_subjects.py`.

WandB logs train loss, validation loss, and dev metrics per epoch; final test metrics and the selected epoch are logged after training. Recovery checkpoints store dev metrics. If an old-format checkpoint is encountered, its weights are not restored; that subject starts from scratch under the new protocol.

### 5. Seed handling and output folders

`run.py` accepts `--seed` (default `27`) and resets the random seeds at the start of each subject iteration before model initialization. Intra-subject split generation remains fixed at `random_state=27`; the run seed controls model/training randomness, not which rows are assigned to splits.

For seed 27, existing output folder names are retained. For other seeds, a suffix distinguishes outputs:

```text
data/saved_dict/<model_name>/            # seed 27
data/saved_dict/<model_name>_seed28/     # seed 28
```

Checkpoint writing, resume loading, and completed-subject skipping use the same seed-aware directory. This prevents one seed from accidentally resuming or skipping based on another seed's files.

## How to run

Run commands from the repository root with the project's dependencies installed and the relevant data prepared. See `SETUP_AND_RUN.md` for environment and dataset setup.

The original model remains the default:

```powershell
python run.py
```

Select a model explicitly and set a seed:

```powershell
python run.py --model InceptSADNet --seed 27
python run.py --model models.ablations.InceptSADNet_no_se --seed 28
```

Use LOSO by passing the string `True` to the existing `--mode` option:

```powershell
python run.py --model InceptSADNet --mode True --seed 27
python run.py --model models.ablations.InceptSADNet_no_se --mode True --seed 28
```

Add `--resume` to resume a matching incomplete run, or `--compile` to request `torch.compile` where supported:

```powershell
python run.py --model models.ablations.InceptSADNet_no_se --seed 28 --resume
```

Run baseline and each ablation separately using the same subjects, mode, and experimental settings. Do not compare fresh results from this protocol against old test-selected results as if they used the same evaluation procedure.

## Output locations

- Model checkpoints and recovery checkpoints: `data/saved_dict/<model_name-or-seed-suffixed-name>/`
- Text summary: `<model_name>_result.txt` in the current working directory
- WandB run data: according to the existing WandB configuration (the runner prompts for a key if one is not configured; pressing Enter selects offline mode)

The `model_name` is supplied by each model module's `Config`, so each ablation has its own checkpoint folder and summary file.

## Tests and validation status

The synthetic suite is:

```powershell
python -m pytest tests\test_eval_protocol.py -q
```

It checks:

1. LOSO marker data reaches the test loader in order and all held-out examples are retained.
2. Intra-subject dev/test loaders contain the expected split rows, including partial final batches.
3. Evaluation handles a singleton final batch.
4. Dev metrics are used for checkpoint selection, and test evaluation happens once after training.
5. Non-default seed paths and completed-subject resume/skip checks are isolated from seed 27.

At the time of implementation, static Pylance checks and `git diff --check` passed. Runtime tests could not be executed in the available project environment because it lacked PyTorch and pytest. The real-data smoke test and per-subject class-count check also remain to be run locally because `data/raw` and `data/cross` were not present in the workspace.

## Before generating new paper numbers

1. Inspect per-subject class counts. Confirm every present class has at least three samples so each can contribute to train/dev/test. The updated splitter retains unequal class counts rather than requiring class 0 to be the smallest.
2. Run the documented per-subject label-count check and verify label meanings and subject selection against the dataset documentation.
3. Run the synthetic tests in an environment with the project dependencies and pytest.
4. Run the local smoke test on the real dataset.
5. Do not mix checkpoints or summaries from the old protocol/split with results from the corrected protocol. Archive old outputs before rerunning.
6. Report results consistently across baseline and ablations; for multiple seeds, report mean/standard deviation and paired per-subject differences.

## Scope and deferred work

The four ablation architectures are unchanged. The class-aware splitter changes intra-subject split membership; train-only oversampling and explicit Xavier initialization also change training relative to historical runs. Rerun every model/seed using the same pipeline; see [07_class_aware_split.md](./07_class_aware_split.md) and [12_training_configuration_and_metrics.md](./12_training_configuration_and_metrics.md). Subject-selection rationale and out-of-list historical results are unresolved, although the code now fails fast if the configured cohort is incomplete; see [08_subject_selection_and_exclusions.md](./08_subject_selection_and_exclusions.md). The source task and reaction-time-derived target semantics are documented in [09_label_semantics_and_task.md](./09_label_semantics_and_task.md). Preprocessing and loaders now enforce the configured sampling/epoch/channel input contract, but actual recording compatibility remains unverified without the local data; see [10_sampling_epoch_channels.md](./10_sampling_epoch_channels.md). Synthetic smoke tests for all five variants are present but not executed in this dependency-limited environment; see [11_model_variant_smoke_tests.md](./11_model_variant_smoke_tests.md). `cam.py` remains unmodified and needs a separate update before generating CAM figures because it directly indexes the Transformer, which is incompatible with the no-Transformer variant.

For step-by-step implementation notes, see:

- [06_wandb_credential_hygiene.md](./06_wandb_credential_hygiene.md)
- [01_loader_order.md](./01_loader_order.md)
- [02_complete_evaluation_loaders.md](./02_complete_evaluation_loaders.md)
- [03_dev_checkpoint_test_once.md](./03_dev_checkpoint_test_once.md)
- [04_reproducible_seed_paths.md](./04_reproducible_seed_paths.md)
- [05_protocol_tests_and_validation.md](./05_protocol_tests_and_validation.md)
- [07_class_aware_split.md](./07_class_aware_split.md)
- [08_subject_selection_and_exclusions.md](./08_subject_selection_and_exclusions.md)
- [09_label_semantics_and_task.md](./09_label_semantics_and_task.md)
- [10_sampling_epoch_channels.md](./10_sampling_epoch_channels.md)
- [11_model_variant_smoke_tests.md](./11_model_variant_smoke_tests.md)
- [12_training_configuration_and_metrics.md](./12_training_configuration_and_metrics.md)
