from importlib import import_module
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

import train_test
from train_test import (
    eval_epoch,
    get_save_dir,
    has_completed_subject,
    test_epoch,
    test_model as evaluate_model,
    init_network,
    train_MAG,
)
from utils import build_cross_datasets, build_datasets, split_data


def _collect(loader):
    inputs = []
    labels = []
    for x, y in loader:
        inputs.append(x)
        labels.append(y)
    return torch.cat(inputs), torch.cat(labels)


def test_loso_test_loader_is_held_out_split_and_reproducible(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cross_dir = tmp_path / "data" / "cross"
    cross_dir.mkdir(parents=True)

    np.save(cross_dir / "train_s01_x.npy", np.ones((3, 1001, 30), dtype=np.float32))
    np.save(cross_dir / "train_s01_y.npy", np.array([0, 0, 1], dtype=np.int64))
    np.save(cross_dir / "dev_s01_x.npy", np.full((5, 1001, 30), 2, dtype=np.float32))
    np.save(cross_dir / "dev_s01_y.npy", np.ones(5, dtype=np.int64))
    test_labels = np.arange(130, dtype=np.int64) % 3
    np.save(cross_dir / "test_s01_x.npy", np.full((130, 1001, 30), 3, dtype=np.float32))
    np.save(cross_dir / "test_s01_y.npy", test_labels)

    config = SimpleNamespace(batch_size=128)
    train_loader, dev_loader, test_loader = build_cross_datasets(config, "s01")

    test_x_first, test_y_first = _collect(test_loader)
    test_x_second, test_y_second = _collect(test_loader)
    dev_x, _ = _collect(dev_loader)

    train_y = train_loader.dataset.tensors[1].view(-1)
    assert len(train_loader.dataset) == 4
    assert torch.equal(torch.bincount(train_y), torch.tensor([2, 2]))
    assert torch.all(dev_x == 2)
    assert len(dev_loader.dataset) == 5
    assert len(test_x_first) == 130
    assert len(test_loader.dataset) == 130
    assert torch.all(test_x_first == 3)
    assert torch.equal(test_y_first, torch.from_numpy(test_labels))
    assert torch.equal(test_x_first, test_x_second)
    assert torch.equal(test_y_first, test_y_second)


def test_intra_loaders_preserve_split_rows_and_keep_partial_batches(tmp_path):
    subject_dir = tmp_path / "s01"
    subject_dir.mkdir()
    x = np.arange(30 * 1001 * 30, dtype=np.float32).reshape(30, 1001, 30)
    y = np.repeat(np.arange(3, dtype=np.int64), 10)
    np.save(subject_dir / "s01_x.npy", x)
    np.save(subject_dir / "s01_y.npy", y)

    config = SimpleNamespace(data_path=str(subject_dir), batch_size=4)
    train_loader, dev_loader, test_loader = build_datasets(config, "s01")
    expected_train, expected_dev, expected_test = split_data(x, y)

    for loader, expected in (
        (train_loader, expected_train),
        (dev_loader, expected_dev),
        (test_loader, expected_test),
    ):
        actual_x, actual_y = _collect(loader)
        expected_x = expected.iloc[:, :-1].to_numpy().reshape(-1, 1001, 30)
        expected_y = expected.iloc[:, -1].to_numpy()
        assert len(actual_x) == len(expected)
        assert torch.equal(actual_x, torch.from_numpy(expected_x))
        assert torch.equal(actual_y.view(-1), torch.from_numpy(expected_y))

    assert len(dev_loader.dataset) % config.batch_size != 0
    assert len(test_loader.dataset) % config.batch_size != 0


def test_intra_oversampling_is_applied_only_to_train_split(tmp_path):
    subject_dir = tmp_path / "s01"
    subject_dir.mkdir()
    class_counts = {0: 30, 1: 9, 2: 6}
    y = np.concatenate(
        [np.full(count, label, dtype=np.int64) for label, count in class_counts.items()]
    )
    x = np.zeros((y.size, 1001, 30), dtype=np.float32)
    np.save(subject_dir / "s01_x.npy", x)
    np.save(subject_dir / "s01_y.npy", y)

    config = SimpleNamespace(data_path=str(subject_dir), batch_size=128)
    train_loader, dev_loader, test_loader = build_datasets(
        config, "s01", oversample=True
    )
    train_labels = train_loader.dataset.tensors[1].view(-1)

    assert torch.equal(torch.bincount(train_labels), torch.tensor([24, 24, 24]))
    assert len(dev_loader.dataset) == 5
    assert len(test_loader.dataset) == 5


def test_class_aware_split_preserves_all_unequally_distributed_samples():
    class_counts = {0: 23, 1: 11, 2: 7}
    expected_split_counts = {
        0: (19, 2, 2),
        1: (9, 1, 1),
        2: (5, 1, 1),
    }
    y = np.concatenate(
        [np.full(count, label, dtype=np.int64) for label, count in class_counts.items()]
    )
    sample_ids = np.arange(y.size, dtype=np.float32)
    x = np.repeat(sample_ids[:, None, None], 30, axis=2)

    splits = split_data(x, y)
    split_ids = []

    for split_index, split in enumerate(splits):
        values = split.iloc[:, 0].to_numpy().astype(np.int64)
        labels = split.iloc[:, -1].to_numpy().astype(np.int64)
        split_ids.extend(values.tolist())

        for label, count in class_counts.items():
            assert np.count_nonzero(labels == label) == expected_split_counts[label][split_index]

    assert sorted(split_ids) == list(range(y.size))
    assert len(set(split_ids)) == y.size

    repeated_splits = split_data(x, y)
    for first, repeated in zip(splits, repeated_splits):
        assert np.array_equal(first.to_numpy(), repeated.to_numpy())


def test_class_aware_split_rejects_classes_too_small_for_three_splits():
    x = np.zeros((7, 1, 30), dtype=np.float32)
    y = np.array([0, 0, 0, 1, 1, 2, 2], dtype=np.int64)

    try:
        split_data(x, y)
    except ValueError as exc:
        assert "at least 3 are required" in str(exc)
    else:
        raise AssertionError("Expected a clear error for a class with fewer than 3 samples")


def test_all_five_incept_sadnet_variants_accept_the_configured_input_shape():
    model_modules = (
        "models.InceptSADNet",
        "models.ablations.InceptSADNet_no_se",
        "models.ablations.InceptSADNet_no_transformer",
        "models.ablations.InceptSADNet_no_multiscale",
        "models.ablations.InceptSADNet_no_spatial",
    )
    inputs = torch.zeros(2, 1001, 30)

    with torch.no_grad():
        for module_name in model_modules:
            module = import_module(module_name)
            model = module.Model(SimpleNamespace(device=torch.device("cpu"))).eval()
            init_network(model, seed=27)
            outputs = model(inputs)

            assert outputs.shape == (2, 3), module_name
            assert torch.isfinite(outputs).all(), module_name


def test_eval_and_test_epochs_handle_final_single_example():
    x = torch.arange(3 * 30, dtype=torch.float32).reshape(3, 1, 30)
    y = torch.tensor([[0], [1], [2]], dtype=torch.int64)
    loader = DataLoader(TensorDataset(x, y), batch_size=2, shuffle=False, drop_last=False)
    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(30, 3))
    config = SimpleNamespace(device=torch.device("cpu"))

    valid_loss = eval_epoch(config, model, loader)
    predictions, labels, logits = test_epoch(config, model, loader)

    assert np.isfinite(valid_loss)
    assert predictions.shape == (3,)
    assert labels.shape == (3,)
    assert logits.shape == (3, 3)


def test_test_model_surfaces_undefined_auc_instead_of_returning_a_fake_score():
    inputs = torch.zeros(3, 1, 30)
    labels = torch.zeros(3, 1, dtype=torch.int64)
    loader = DataLoader(TensorDataset(inputs, labels), batch_size=2, shuffle=False)
    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(30, 3))
    config = SimpleNamespace(device=torch.device("cpu"), num_classes=3)

    try:
        evaluate_model(config, model, loader)
    except ValueError as exc:
        assert "Could not compute weighted one-vs-rest ROC AUC" in str(exc)
    else:
        raise AssertionError("Expected undefined ROC AUC to be reported explicitly")


def test_weight_initialization_is_deterministic_and_handles_norm_layers():
    model = torch.nn.Sequential(
        torch.nn.Linear(4, 3),
        torch.nn.LayerNorm(3),
    )
    init_network(model, seed=27)
    first_weight = model[0].weight.detach().clone()
    first_bias = model[0].bias.detach().clone()

    init_network(model, seed=27)

    assert torch.equal(first_weight, model[0].weight)
    assert torch.equal(first_bias, model[0].bias)
    assert torch.count_nonzero(model[0].bias) == 0
    assert torch.equal(model[1].weight, torch.ones_like(model[1].weight))


def test_train_selects_on_dev_and_tests_once(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = SimpleNamespace(
        model_name="TinyAblation",
        f1_save_path=str(tmp_path / "saved"),
        learning_rate=0.01,
        device=torch.device("cpu"),
        num_epoch=3,
    )
    model = torch.nn.Linear(1, 3)
    dev_iter = object()
    test_iter = object()
    dev_scores = [(0.3, 0.8), (0.7, 0.4), (0.5, 0.6)]
    calls = {"dev": 0, "test": 0}
    logged = []

    class WandbStub:
        config = None

        def __init__(self):
            self.config = self

        def init(self, **kwargs):
            return self

        def update(self, *args, **kwargs):
            pass

        def log(self, values):
            logged.append(values)

        def finish(self):
            pass

    monkeypatch.setattr(train_test, "wandb", WandbStub())

    epoch_counter = {"value": 0}

    def fake_train_epoch(config, model, train_iter, optimizer):
        epoch_counter["value"] += 1
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.fill_(epoch_counter["value"])
        return 0.1

    def fake_eval_epoch(config, model, loader):
        return 0.2

    def fake_test_model(config, model, loader):
        if loader is dev_iter:
            calls["dev"] += 1
            f1, auc = dev_scores[calls["dev"] - 1]
            return 0.6, 0.0, 0.0, f1, auc

        calls["test"] += 1
        assert all(torch.all(parameter == 2) for parameter in model.parameters())
        return 0.8, 0.0, 0.0, 0.77, 0.88

    monkeypatch.setattr(train_test, "train_epoch", fake_train_epoch)
    monkeypatch.setattr(train_test, "eval_epoch", fake_eval_epoch)
    monkeypatch.setattr(train_test, "test_model", fake_test_model)

    result = train_MAG(config, model, [], dev_iter, test_iter, "s01", seed=27)

    assert result == 0.77
    assert calls == {"dev": 3, "test": 1}
    assert len(logged) == 4
    lines = (tmp_path / "TinyAblation_result.txt").read_text().splitlines()
    assert len(lines) == 1
    assert lines[0].startswith(
        "subject:s01, mode:inter, seed:27, best_epoch:1, best_dev_f1:0.7, "
        "test_acc:0.8, best_test_f1:0.77, best_test_auc:0.88"
    )
    assert float(lines[0].split("best_test_auc:")[-1]) == 0.88


def test_seed_specific_output_and_resume_skip_paths(tmp_path):
    config = SimpleNamespace(model_name="InceptSADNet_no_SE", f1_save_path=str(tmp_path))

    assert get_save_dir(config, 27) == str(tmp_path / "InceptSADNet_no_SE")
    seed_28_dir = tmp_path / "InceptSADNet_no_SE_seed28"
    assert get_save_dir(config, 28) == str(seed_28_dir)

    seed_28_dir.mkdir()
    (seed_28_dir / "f1_s01_inter.ckpt").touch()
    (seed_28_dir / "auc_s01_inter.ckpt").touch()
    assert has_completed_subject(config, "s01", "inter", 28)

    (seed_28_dir / "checkpoint_s01_inter.pt").touch()
    assert not has_completed_subject(config, "s01", "inter", 28)

    seed_27_dir = tmp_path / "InceptSADNet_no_SE"
    seed_27_dir.mkdir()
    (seed_27_dir / "f1_s01_inter.ckpt").touch()
    (seed_27_dir / "auc_s01_inter.ckpt").touch()
    (seed_28_dir / "checkpoint_s01_inter.pt").unlink()
    (seed_28_dir / "f1_s01_inter.ckpt").unlink()
    (seed_28_dir / "auc_s01_inter.ckpt").unlink()
    assert not has_completed_subject(config, "s01", "inter", 28)
    assert has_completed_subject(config, "s01", "inter", 27)
