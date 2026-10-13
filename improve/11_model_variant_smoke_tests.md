# Improvement 11 — Synthetic Smoke Tests for All Model Variants

## Change made

Added a forward-pass test that instantiates the full InceptSADNet model and its four ablations (without SE, Transformer, multi-scale convolution, and spatial convolution), sends each a synthetic batch shaped `(2, 1001, 30)`, and checks that each returns finite logits shaped `(2, 3)`.

The model architectures were not modified. The runner now applies the documented deterministic Xavier initialization to Conv/Linear layers before training.

## Validation status

The test was added but could not be executed in this workspace. The selected project environment lacks NumPy and PyTorch, and test collection fails before reaching the smoke test. No real-dataset smoke test has been run. Run `python -m pytest tests/test_eval_protocol.py -q` in the project's fully provisioned environment before starting full training.
