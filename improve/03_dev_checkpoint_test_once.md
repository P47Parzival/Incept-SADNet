# Improvement 3 — Select Checkpoints on Dev, Test Once

## PRD coverage

Implements C3 (defect D3) from `prd.md`.

## Changes

- Each epoch now calculates loss and classification metrics on the dev loader.
- Best-F1 and best-AUC checkpoint files are selected using dev F1/AUC, not test metrics.
- The model selected by dev F1 is loaded after the training loop, then test metrics are calculated once.
- The result file receives one summary line for the run. Its final field remains `best_test_auc:` for compatibility with `evaluate_all_subjects.py` parsing.
- WandB records train loss, dev loss, dev F1/AUC per epoch; test accuracy/F1/AUC and selected epoch are logged at the end.
- New recovery checkpoints store dev metrics and the selected epoch. An old-format checkpoint lacking `dev_f1s` is detected before model/optimizer state is restored; the subject starts fresh rather than inheriting old test-selected weights.

## Checkpoint selection details

The F1-selected and AUC-selected checkpoints retain their existing filenames. Test metrics are all measured from the F1-selected checkpoint, so reported test F1 and AUC refer to the same trained model.

## Result format

Each completed subject/run appends one line with the format:

```text
subject:{id}, mode:{mode}, seed:{seed}, best_epoch:{epoch}, best_dev_f1:{score}, test_acc:{score}, best_test_f1:{score}, best_test_auc:{score}
```

`best_test_auc` is intentionally last so the existing parser can read it.

## Verification

Synthetic protocol tests will assert one evaluation on the test loader after training, one dev-metric evaluation per epoch, validation-based checkpoint selection, and one correctly formatted result line.
