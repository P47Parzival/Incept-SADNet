# Improvement 1 — Consistent Train/Dev/Test Loader Ordering

## PRD coverage

Implements C1 (defects D1 and D2) from `prd.md`.

## Changes

- `utils.build_datasets` now assigns the split outputs in their documented order: train, dev, test.
- Intra-subject and LOSO call sites in `run.py` now unpack train, dev, test in that same order.
- `build_cross_datasets` already returned train, dev, test and needed no modification.
- `evaluate_all_subjects.py` already unpacks train, dev, test and remains unchanged.

## Intra-subject split preservation

`split_data` still uses the same split implementation and `random_state=27`. The old loader construction accidentally swapped dev/test when naming them; the old `run.py` unpacking swapped them back. Updating both sides preserves which split rows are supplied as dev and test to `train_MAG`, while making the API's names match the actual split contents. The separate `evaluate_all_subjects.py` path now receives the correctly identified dev and test data as well.

## Verification

Synthetic assertions for row-level loader identity and LOSO marker separation are included in the evaluation-protocol test suite and will be run with the complete protocol validation. No model, hyperparameter, or split-generation code was changed.
