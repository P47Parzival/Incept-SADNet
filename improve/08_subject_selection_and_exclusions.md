# Improvement 8 — Subject Selection and Exclusions

## Current implementation

`dataset_contract.SUBJECT_IDS` is the single fixed 16-subject allowlist:

`s01, s05, s09, s13, s22, s31, s35, s40, s41, s42, s43, s44, s45, s49, s50, s53`

`run.get_all_subjects()` uses that contract for preprocessing, intra-subject training, and LOSO training. Matching is case-insensitive, and the returned paths use the configured order.

The implementation now fails explicitly if `data/raw` is absent, any configured subject directory is missing, or duplicate case-insensitive subject directories are found. This prevents silently training/evaluating on a partial cohort.

## Audit findings

- The repository does not state the scientific inclusion or exclusion criteria that produced this 16-subject cohort. The code comment says it follows the SADNet paper, but the paper citation and participant-selection rationale are not recorded here.
- The old `get_all_subjects()` behavior silently skipped absent subjects. That was replaced with a fail-fast completeness check in `run.py` and `dataset_contract.py`.
- `intersubject.txt` contains results for IDs outside the configured allowlist (including `s02`, `s04`, `s06`, `s11`, `s12`, `s14`, and `s23`). These results are not consistent with the current 16-subject cohort and must not be combined with or presented as results from this configuration unless their provenance and protocol are established.
- The raw dataset is not present in this workspace, so participant availability and per-subject file completeness could not be checked.

## Decision before full training

Do not change the 16-subject list based only on the result text file. Before training, record the dataset/paper source and the exact inclusion/exclusion rule, then verify that every included subject is present and complete. Confirm whether the extra-subject results are obsolete, belong to another cohort/configuration, or require a protocol correction. Update the manuscript and result artifacts consistently after that decision.

## Status

**Code guard implemented; dataset/methodology verification remains incomplete.** Synthetic contract tests pass, but the scientific rationale, actual dataset availability, raw-file completeness, and provenance of out-of-list historical results remain unresolved. No result files or cohort membership were changed.
