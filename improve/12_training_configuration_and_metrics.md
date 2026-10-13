# Improvement 12 — Training Configuration and Evaluation Metrics

## Changes made

- The runner now calls `init_network()` for intra-subject and LOSO runs. Xavier-normal initialization is applied to convolutional and linear weights, their biases are zeroed, and normalization-layer defaults are preserved. Initialization is seeded from `--seed`.
- The initialization helper rejects unsupported method names rather than silently using a different initializer.
- Intra-subject oversampling, previously commented out despite the `oversample=True` call, is now performed only on the training split.
- LOSO oversampling is likewise applied only to the combined training split. Dev and held-out test arrays are unchanged.
- Oversampling is deterministic with seed 27 and duplicates minority-class training indices until each training class reaches the majority count.
- Intra-subject splits use the fixed shared split seed 27. LOSO session files are processed in sorted order, and each training subject's class-wise shuffle uses a stable name-derived seed.
- Weighted one-vs-rest ROC AUC remains the configured AUC metric. If the evaluation data does not support that metric, evaluation now raises a descriptive error instead of recording a fallback value of `0.5`.

## Current run settings

For InceptSADNet and its four ablations, the inherited configuration is 3 classes, Adam with learning rate `0.002`, 100 epochs, batch size 128, and the existing dropout settings (0.5 in convolutional blocks and the first head layer, 0.3 in the second head layer). The split and oversampling seed is 27; the training seed defaults to 27 and can be changed with `--seed`. The training loop selects its final checkpoint by dev weighted F1 and evaluates the test split once. Reported final metrics include accuracy, weighted F1, and weighted one-vs-rest ROC AUC. The deterministic train/dev/test cohort split and initialization are shared across model variants for a given seed.

These settings now match the documented initialization and training-only balancing behavior. Do not compare new runs with previous result files: the class-aware split, oversampling, channel selection, and initialization pipeline differ from prior runs.

## Validation status

The pure deterministic label/cohort/channel/oversampling contract suite passes. The protocol tests for train-only balancing, undefined AUC, initialization, and five-model forward passes are present but could not be run here because NumPy/PyTorch are missing from the selected environment. Dataset-backed runs and final metric review remain pending.
