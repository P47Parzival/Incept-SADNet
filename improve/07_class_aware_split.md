# Improvement 7 — Class-Aware Intra-Subject Splitting

## Why this was needed

The previous `utils.split_data` sorted labels and used the number of class-0 samples to slice the class groups. If classes had unequal counts, those slices could omit samples or mix class boundaries. The splitter did not conserve every input sample.

## Change made

`utils.split_data` now:

1. Checks that the number of EEG examples matches the number of labels.
2. Finds the indices for each label that is present.
3. Requires at least three samples per present class so that train, dev, and test can each contain at least one sample of that class.
4. Shuffles each class's indices deterministically with random state 27.
5. Assigns approximately 80%/10%/10% of each class to train/dev/test, with integer allocation that guarantees at least one example in each split.
6. Shuffles the combined split indices deterministically and returns the same three-DataFrame API: train, dev, test.

No examples are oversampled or discarded by this splitter. It supports unequal counts across the labels present in the subject data.

## Experimental consequence

This changes intra-subject split membership relative to earlier runs. Re-run the full InceptSADNet baseline and every ablation using this same splitter before comparing their results. Do not combine results from the old split with results from this one.

The seed for the split remains fixed at 27 regardless of the model/training `--seed`, so each model and training seed sees identical train/dev/test membership. Oversampling is performed only after this split and only on the training portion; see [12_training_configuration_and_metrics.md](./12_training_configuration_and_metrics.md).

## Checks added

`tests/test_eval_protocol.py` includes synthetic data with unequal class counts. It checks that:

- every original sample occurs exactly once across train/dev/test,
- no samples are duplicated or lost,
- the split assigns each present class to every split,
- repeated calls produce identical split contents, and
- a class with fewer than three samples triggers a descriptive error.

These splitter tests are part of `tests/test_eval_protocol.py`, which could not be collected in this workspace because NumPy is unavailable in the selected project environment. The separate dataset-contract tests pass, but they do not exercise the splitter. Real subject label counts remain unverified because the dataset is not available here. Before training, inspect `sXX_y.npy` for each subject and confirm each present class has at least three examples.

Example local count check from the repository root:

```python
from pathlib import Path
from collections import Counter
import numpy as np

for label_path in sorted(Path("data/raw").glob("s*/s*_y.npy")):
    labels = np.load(label_path).reshape(-1)
    counts = Counter(labels.tolist())
    print(label_path.parent.name, dict(sorted(counts.items())))
    if any(count < 3 for count in counts.values()):
        print("STOP: at least one class cannot populate all three splits.")
```

This only verifies counts. It does not validate that the dataset's label meanings, subject exclusions, or the sample-level split strategy are appropriate for the paper; those require checking the dataset documentation and experimental design.
