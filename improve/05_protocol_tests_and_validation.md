# Improvement 5 — Synthetic Protocol Tests and Validation Status

## PRD coverage

Adds the automated checks from Section 6 of `prd.md` using CPU-compatible synthetic fixtures where local dataset files are not required.

## Tests added

`tests/test_eval_protocol.py` covers:

- LOSO split markers, all 130 held-out test rows, and repeatable test-loader order.
- Intra-subject row identity against `split_data`, including preservation of final partial batches.
- Evaluation loss and logits for a final singleton batch.
- Dev-metric evaluation each epoch, dev-selected F1 checkpoint restoration, exactly one final test evaluation, and result-line format/parser compatibility.
- Default and non-default seed paths, including completed-subject resume/skip behavior.
- Unequal per-class counts, sample conservation across splits, deterministic repeated split generation, and a clear error when a present class cannot populate train/dev/test.

The two-epoch real-subject smoke check remains a local integration check because it requires the user's preprocessed dataset.

## Validation attempted

- Pylance diagnostics reported no errors for `run.py`, `utils.py`, `train_test.py`, or `tests/test_eval_protocol.py`.
- `git diff --check` passed.
- `tests/test_dataset_contract.py` passes all ten tests.
- `tests/test_eval_protocol.py` could not be collected: the selected project virtual environment has no NumPy module. The protocol and five-model smoke tests in that file therefore remain unverified.
- The real-subject smoke test and per-subject class-count gate were not run: `data/raw` and `data/cross` are not present in this workspace.

Run the protocol suite in an environment with the project's requirements plus pytest:

```powershell
python -m pytest tests\test_eval_protocol.py -q
```

Before any full training rerun, inspect class counts and confirm each present class has at least three samples per subject. The updated class-aware splitter preserves all samples when classes have different counts, but rejects any present class with fewer than three samples because it cannot place at least one example in each split. Do not mix old checkpoints/results with results from the corrected protocol.
