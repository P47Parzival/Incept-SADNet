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

The two-epoch real-subject smoke check remains a local integration check because it requires the user's preprocessed dataset.

## Validation attempted

- Pylance diagnostics reported no errors for `run.py`, `utils.py`, `train_test.py`, or `tests/test_eval_protocol.py`.
- `git diff --check` passed.
- The focused `pytest` command could not run: the selected project virtual environment has no `pytest` module. Earlier runtime probing also found no `torch` module in that environment. Dependencies were not installed as part of this implementation.
- The real-subject smoke test and per-subject class-count gate were not run: `data/raw` and `data/cross` are not present in this workspace.

Run the synthetic suite in an environment with the project's requirements plus pytest:

```powershell
python -m pytest tests\test_eval_protocol.py -q
```

Before any full training rerun, follow the class-count gate in `prd.md`. Do not mix old checkpoints/results with results from the corrected protocol.
