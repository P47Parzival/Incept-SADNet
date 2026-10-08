# Improvement 4 — Per-Subject Seeds and Isolated Seed Outputs

## PRD coverage

Implements C4 and C5 (defect D5 and the `--seed` requirement) from `prd.md`.

## Changes

- `utils.set_seed(seed=27)` applies the requested seed to NumPy, PyTorch CPU/CUDA, Python `random`, and the existing `PYTHONHASHSEED` setting.
- `run.py` adds `--seed` with default `27`; the default command retains the old default seed.
- The runner resets the seed at the beginning of each subject iteration before model creation in both intra-subject and LOSO modes. The initial call in `__main__` is retained.
- `train_MAG` accepts `seed=27` and uses a shared `get_save_dir(config, seed)` helper for its checkpoint directory.
- Both resume/completed-subject checks in `run.py` use the same helper, so they inspect the same location that `train_MAG` writes.
- Seed 27 retains the pre-existing checkpoint directory name. Other seeds receive distinct folders such as `<model_name>_seed28`.
- The seed is included in each result line.

## Run examples

```powershell
python run.py --model InceptSADNet --seed 27
python run.py --model models.ablations.InceptSADNet_no_se --seed 28
python run.py --model models.ablations.InceptSADNet_no_se --seed 28 --resume
```

## Scope

Data split generation remains fixed at `random_state=27`, as required by the PRD. The run seed affects model initialization and shuffled training batches; the validation/test loaders are deterministic from Improvement 2.

## Verification

Synthetic tests will check the seed-27 default path, non-default seed-specific checkpoint paths, and that resume/skip logic uses the shared path helper.
