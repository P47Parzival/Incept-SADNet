import pytest

from dataset_contract import (
    EEG_CHANNELS,
    N_EEG_CHANNELS,
    N_EPOCH_SAMPLES,
    SUBJECT_IDS,
    classify_response_time,
    oversample_indices,
    resolve_eeg_channel_names,
    resolve_subject_names,
    stable_seed_for_name,
    validate_epoch_arrays,
)


class ShapedArray:
    def __init__(self, shape):
        self.shape = shape
        self.ndim = len(shape)


def test_resolves_channels_in_model_order_case_insensitively():
    available = ["M1", *reversed([name.lower() for name in EEG_CHANNELS]), "M2"]

    assert resolve_eeg_channel_names(available) == [
        name.lower() for name in EEG_CHANNELS
    ]


def test_rejects_missing_or_ambiguous_required_channels():
    with pytest.raises(ValueError, match="missing required EEG channels"):
        resolve_eeg_channel_names(EEG_CHANNELS[:-1])

    with pytest.raises(ValueError, match="Ambiguous channel names"):
        resolve_eeg_channel_names([*EEG_CHANNELS, "fp1"])


def test_subject_cohort_is_complete_and_returned_in_fixed_order():
    available = ["unlisted", *reversed([subject.upper() for subject in SUBJECT_IDS])]

    assert resolve_subject_names(available) == [
        subject.upper() for subject in SUBJECT_IDS
    ]


def test_subject_cohort_fails_instead_of_silently_omitting_missing_ids():
    with pytest.raises(FileNotFoundError, match="configured 16-subject cohort is incomplete"):
        resolve_subject_names(SUBJECT_IDS[:-1])


def test_epoch_arrays_must_match_model_input_contract():
    x = ShapedArray((2, N_EPOCH_SAMPLES, N_EEG_CHANNELS))
    y = ShapedArray((2,))

    validate_epoch_arrays(x, y)

    with pytest.raises(ValueError, match="Expected 30 EEG channels"):
        validate_epoch_arrays(ShapedArray((2, N_EPOCH_SAMPLES, N_EEG_CHANNELS - 1)), y)

    with pytest.raises(ValueError, match="Expected 1001 samples"):
        validate_epoch_arrays(ShapedArray((2, N_EPOCH_SAMPLES - 1, N_EEG_CHANNELS)), y)

    with pytest.raises(ValueError, match="one label per EEG trial"):
        validate_epoch_arrays(x, ShapedArray((1,)))

    with pytest.raises(ValueError, match="no labeled EEG trials"):
        validate_epoch_arrays(ShapedArray((0,)), ShapedArray((0,)))


def test_response_time_classification_preserves_fast_slow_and_fallback_rules():
    alert_rt = 100
    assert classify_response_time(100, 100, alert_rt) == 2
    assert classify_response_time(300, 300, alert_rt) == 1
    assert classify_response_time(200, 100, alert_rt) == 0
    assert classify_response_time(0, 100, alert_rt) == 2
    assert classify_response_time(0, 300, alert_rt) == 1
    assert classify_response_time(0, 200, alert_rt) == 0


def test_response_time_classification_rejects_nonpositive_threshold():
    with pytest.raises(ValueError, match="must be positive"):
        classify_response_time(100, 100, 0)


def test_oversampling_balances_only_by_duplicating_training_indices_deterministically():
    labels = [0] * 5 + [1] * 2 + [2]
    first = oversample_indices(labels)
    second = oversample_indices(labels)
    counts = {
        label: sum(labels[index] == label for index in first)
        for label in set(labels)
    }

    assert first == second
    assert counts == {0: 5, 1: 5, 2: 5}
    assert len(first) == 15
    assert set(first) == set(range(len(labels)))


def test_oversampling_rejects_empty_labels():
    with pytest.raises(ValueError, match="empty label array"):
        oversample_indices([])


def test_subject_split_seed_is_stable_and_case_insensitive():
    assert stable_seed_for_name("s01") == stable_seed_for_name("S01")
    assert stable_seed_for_name("s01") != stable_seed_for_name("s05")
