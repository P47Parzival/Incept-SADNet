# Improvement 10 — Sampling Rate, Epoch Shape, and Channel Order

## Facts from the current source

- `make_datasets.py` sets `freq = 500` and calls `raw.resample(freq)` before epoching.
- Epochs are created from `tmin = -2` through `tmax = 0`. At 500 Hz, inclusive endpoints imply 1001 time samples per epoch.
- `dataset_contract.py` holds the shared ordered 30-electrode list, 1001-sample epoch length, and array validation.
- The sample rate and epoch bounds are also defined in the shared contract and used by preprocessing, preventing the loader shape contract from drifting from resampling and epoch settings.
- Before converting Epochs to a DataFrame, preprocessing resolves channel names case-insensitively, selects exactly the configured EEG electrodes in model order, and raises a clear error for missing or ambiguous names.
- Generated arrays are validated for shape `(trials, 1001, 30)`, nonempty trials, and one label per trial. Intra-subject and LOSO loaders also validate saved arrays before training/evaluation.
- Session file enumeration is sorted, and LOSO source-subject class shuffles use a stable subject-specific seed, so repeated dataset generation uses repeatable ordering.
- The public dataset metadata describes 32 recorded channels, including 30 EEG electrodes and two mastoid references.

## Required local verification

The dataset is absent from this workspace, so actual raw channel names and the existing arrays' provenance cannot be established here. The code now enforces the intended shape and channel order when preprocessing/using data. Before full training:

1. Confirm that each raw recording has the 30 required electrode names (the preprocessing will stop if it does not).
2. Regenerate the per-subject arrays so that no stale arrays bypass the channel selection and ordering fix.
3. Run a model forward-pass smoke test on a real batch and verify the intended channels correspond to the source recording.

The exact electrode set is kept unchanged from the existing intended list. If the dataset uses different names, resolve the alias mapping based on verified recording metadata rather than silently substituting channels.

## Status

**Implementation added; real-data validation remains pending.** Source-level synthetic contract tests pass. Actual recording compatibility, regenerated arrays, and a real-batch smoke test remain a stop condition before full training.
