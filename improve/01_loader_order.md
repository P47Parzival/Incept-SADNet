# Improvement 1 — Consistent Train/Dev/Test Loader Ordering

## PRD coverage

Implements C1 (defects D1 and D2) from `prd.md`.

## Changes

- `utils.build_datasets` now assigns the split outputs in their documented order: train, dev, test.
- Intra-subject and LOSO call sites in `run.py` now unpack train, dev, test in that same order.
- `build_cross_datasets` already returned train, dev, test and needed no modification.
- `evaluate_all_subjects.py` already unpacks train, dev, test and remains unchanged.

## Intra-subject split preservation

At the time of this step, the old loader construction accidentally swapped dev/test when naming them; the old `run.py` unpacking swapped them back. Updating both sides made the API's names match the actual split contents and preserved the original split membership.

**Follow-up:** [07_class_aware_split.md](./07_class_aware_split.md) later changes the intra-subject split algorithm to retain all samples across unequal class counts. Therefore, newly generated intra-subject split membership is no longer identical to the old split. Re-run the baseline and all ablations with the updated split; do not compare new runs directly with numbers produced from the old split.

## Verification

Synthetic assertions for row-level loader identity and LOSO marker separation are included in the evaluation-protocol test suite and will be run with the complete protocol validation. No model, hyperparameter, or split-generation code was changed.
