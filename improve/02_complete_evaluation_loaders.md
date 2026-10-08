# Improvement 2 — Complete, Deterministic Evaluation Loaders

## PRD coverage

Implements C2 (defect D4) from `prd.md`.

## Changes

- Intra-subject dev and test DataLoaders now use `shuffle=False` and `drop_last=False`.
- LOSO dev and test DataLoaders now use `shuffle=False` and `drop_last=False`.
- Train loaders retain their existing shuffle/drop behavior.
- Evaluation loss targets are flattened with `y.view(-1)` in `eval_epoch` and `test_epoch`, so a final batch containing one example remains a batch-shaped target.
- `test_epoch` concatenates per-batch logits with `np.concatenate(..., axis=0)`, supporting a smaller final batch.

## Effect

Validation and test metrics now cover every example in their respective splits in a stable order. This intentionally includes examples from the final partial batch that were previously dropped. The split membership and split generation remain unchanged.

## Verification

Synthetic checks for all-example coverage, stable repeated iteration, and single-example final batches are part of the evaluation-protocol test suite and will be run with the complete protocol validation.
