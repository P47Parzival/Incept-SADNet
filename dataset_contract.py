"""Shared dataset dimensions and cohort used by preprocessing and training."""

import random
import zlib

SUBJECT_IDS = (
    "s01", "s05", "s09", "s13", "s22", "s31", "s35", "s40",
    "s41", "s42", "s43", "s44", "s45", "s49", "s50", "s53",
)

EEG_CHANNELS = (
    "FP1", "FP2", "F7", "F3", "FZ", "F4", "F8", "FT7", "FC3", "FCZ",
    "FC4", "FT8", "T3", "C3", "CZ", "C4", "T4", "TP7", "CP3", "CPZ",
    "CP4", "TP8", "T5", "P3", "PZ", "P4", "T6", "O1", "OZ", "O2",
)

N_EEG_CHANNELS = len(EEG_CHANNELS)
SAMPLE_RATE = 500
EPOCH_TMIN = -2
EPOCH_TMAX = 0
N_EPOCH_SAMPLES = round((EPOCH_TMAX - EPOCH_TMIN) * SAMPLE_RATE) + 1
SPLIT_SEED = 27


def classify_response_time(global_rt, local_rt, alert_rt):
    """Map existing response-time thresholds to class IDs 0, 1, and 2."""
    if alert_rt <= 0:
        raise ValueError(f"Alert response-time threshold must be positive; got {alert_rt}.")
    if global_rt != 0:
        if global_rt < 1.5 * alert_rt and local_rt < 1.5 * alert_rt:
            return 2
        if global_rt > 2.5 * alert_rt and local_rt > 2.5 * alert_rt:
            return 1
        return 0
    if local_rt < 1.5 * alert_rt:
        return 2
    if local_rt > 2.5 * alert_rt:
        return 1
    return 0


def stable_seed_for_name(name, seed=SPLIT_SEED):
    """Derive a repeatable, name-specific seed independent of Python hash randomization."""
    return zlib.crc32(name.casefold().encode("utf-8"), seed)


def oversample_indices(labels, seed=SPLIT_SEED):
    """Balance class counts by deterministic resampling with replacement."""
    class_indices = {}
    for index, label in enumerate(labels):
        class_indices.setdefault(label, []).append(index)
    if not class_indices:
        raise ValueError("Cannot oversample an empty label array.")

    target_count = max(len(indices) for indices in class_indices.values())
    rng = random.Random(seed)
    sampled_indices = list(range(len(labels)))
    for indices in class_indices.values():
        sampled_indices.extend(
            rng.choices(indices, k=target_count - len(indices))
        )
    rng.shuffle(sampled_indices)
    return sampled_indices


def resolve_eeg_channel_names(available_names):
    """Return the requested EEG channel names in model order, case-insensitively."""
    names_by_key = {}
    for name in available_names:
        key = name.casefold()
        if key in names_by_key:
            raise ValueError(f"Ambiguous channel names differ only by case: {name!r}")
        names_by_key[key] = name

    missing = [name for name in EEG_CHANNELS if name.casefold() not in names_by_key]
    if missing:
        raise ValueError(
            "Dataset is missing required EEG channels: " + ", ".join(missing)
        )

    return [names_by_key[name.casefold()] for name in EEG_CHANNELS]


def resolve_subject_names(available_names):
    """Return configured subject directory names in cohort order or fail explicitly."""
    names_by_key = {}
    for name in available_names:
        key = name.casefold()
        if key in SUBJECT_IDS:
            if key in names_by_key:
                raise ValueError(f"Duplicate directories found for subject {key}.")
            names_by_key[key] = name

    missing = [subject_id for subject_id in SUBJECT_IDS if subject_id not in names_by_key]
    if missing:
        raise FileNotFoundError(
            "The configured 16-subject cohort is incomplete. Missing raw subject "
            "directories: " + ", ".join(missing)
        )

    return [names_by_key[subject_id] for subject_id in SUBJECT_IDS]


def validate_epoch_arrays(X, y):
    """Validate saved preprocessing arrays against the training input contract."""
    if X.shape[0] == 0:
        raise ValueError("Preprocessing produced no labeled EEG trials.")
    if X.ndim != 3:
        raise ValueError(
            f"Expected EEG data with shape (trials, samples, channels); got {X.shape}."
        )
    if X.shape[1] != N_EPOCH_SAMPLES:
        raise ValueError(
            f"Expected {N_EPOCH_SAMPLES} samples per epoch; got {X.shape[1]}."
        )
    if X.shape[2] != N_EEG_CHANNELS:
        raise ValueError(
            f"Expected {N_EEG_CHANNELS} EEG channels in the configured order; "
            f"got {X.shape[2]}."
        )
    if y.ndim != 1 or y.shape[0] != X.shape[0]:
        raise ValueError(
            f"Expected one label per EEG trial; got X={X.shape}, y={y.shape}."
        )
