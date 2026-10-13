# coding: UTF-8
import os
import random

import torch
import numpy as np
import pandas as pd
import time
from datetime import timedelta
from dataset_contract import (
    N_EEG_CHANNELS,
    N_EPOCH_SAMPLES,
    SPLIT_SEED,
    oversample_indices,
    validate_epoch_arrays,
)


# 设置随机数种子，方便复现
def set_seed(seed=27):
    np.random.seed(seed)
    torch.manual_seed(seed)  # 设置CPU生成随机数的种子，方便下次复现实验结果
    torch.cuda.manual_seed_all(seed)  # 在GPU中设置生成随机数的种子。当设置的种子固定下来的时候，之后依次pytorch生成的随机数序列也被固定下来
    torch.backends.cudnn.deterministic = True  # 保证每次运行网络的时候相同输入的输出是固定的
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


# 获取时间差
def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


def split_data(X, y):
    X = np.asarray(X)
    y = np.asarray(y).reshape(-1)
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"X and y sample counts differ: {X.shape[0]} != {y.shape[0]}")
    if y.size == 0:
        raise ValueError("Cannot split an empty dataset.")

    X = X.reshape(X.shape[0], -1)
    rng = np.random.RandomState(SPLIT_SEED)
    split_indices = [[], [], []]

    for label in np.unique(y):
        class_indices = np.flatnonzero(y == label)
        if class_indices.size < 3:
            raise ValueError(
                f"Class {label!r} has {class_indices.size} samples; "
                "at least 3 are required to place samples in train, dev, and test."
            )

        rng.shuffle(class_indices)
        ideal_sizes = np.asarray([0.8, 0.1, 0.1]) * class_indices.size
        sizes = np.floor(ideal_sizes).astype(int)
        sizes = np.maximum(sizes, 1)

        while sizes.sum() > class_indices.size:
            reducible = np.flatnonzero(sizes > 1)
            excess = sizes[reducible] - ideal_sizes[reducible]
            sizes[reducible[np.argmax(excess)]] -= 1

        while sizes.sum() < class_indices.size:
            deficit = ideal_sizes - sizes
            sizes[np.argmax(deficit)] += 1

        train_end = sizes[0]
        dev_end = train_end + sizes[1]
        split_indices[0].extend(class_indices[:train_end])
        split_indices[1].extend(class_indices[train_end:dev_end])
        split_indices[2].extend(class_indices[dev_end:])

    output = []
    for indices in split_indices:
        indices = np.asarray(indices)
        rng.shuffle(indices)
        data = np.column_stack((X[indices], y[indices])).astype(np.float32)
        output.append(pd.DataFrame(data))

    return tuple(output)


def trans2DataLoader(data, batch_size=32, shuffle=True, drop_last=False):
    # drop_last：当batch_size较大时，可能会drop掉某类别的全部标签导致分类出错
    X = data.iloc[:, :-1].to_numpy()
    y = data.iloc[:, -1:].to_numpy()
    expected_features = N_EEG_CHANNELS * N_EPOCH_SAMPLES
    if X.shape[1] != expected_features:
        raise ValueError(
            f"Expected {N_EPOCH_SAMPLES} samples x {N_EEG_CHANNELS} EEG channels "
            f"({expected_features} features per trial); got {X.shape[1]}."
        )
    dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(X.reshape(X.shape[0], -1, N_EEG_CHANNELS)),
        torch.from_numpy(y),
    )
    # pin_memory=True speeds up host-to-device transfers (RAM to GPU)
    # num_workers=0 is usually faster for small TensorDatasets on Windows
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, pin_memory=True, num_workers=0)
    return loader


# use for leave-one-subject-experiment
def build_cross_datasets(config, subject_name):
    # subject_name : sxx
    print("build datasets...")
    batch_size = config.batch_size
    train_x = np.load('data/cross/train_'+subject_name+'_x.npy')
    train_y = np.load('data/cross/train_'+subject_name+'_y.npy')

    dev_x = np.load('data/cross/dev_'+subject_name+'_x.npy')
    dev_y = np.load('data/cross/dev_'+subject_name+'_y.npy')

    test_x = np.load('data/cross/test_'+subject_name+'_x.npy')
    test_y = np.load('data/cross/test_'+subject_name+'_y.npy')

    train_y = train_y.reshape(-1)
    dev_y = dev_y.reshape(-1)
    test_y = test_y.reshape(-1)
    validate_epoch_arrays(train_x, train_y)
    validate_epoch_arrays(dev_x, dev_y)
    validate_epoch_arrays(test_x, test_y)

    if train_y.size:
        train_indices = oversample_indices(train_y.tolist())
        train_x = train_x[train_indices]
        train_y = train_y[train_indices]
    train_dataset = torch.utils.data.TensorDataset(torch.from_numpy(train_x.astype(np.float32)), torch.from_numpy(train_y.astype(np.int64)))
    dev_dataset = torch.utils.data.TensorDataset(torch.from_numpy(dev_x.astype(np.float32)), torch.from_numpy(dev_y.astype(np.int64)))
    test_dataset = torch.utils.data.TensorDataset(torch.from_numpy(test_x.astype(np.float32)), torch.from_numpy(test_y.astype(np.int64)))

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=True, num_workers=0)
    dev_loader = torch.utils.data.DataLoader(dev_dataset, batch_size=batch_size, shuffle=False, drop_last=False, pin_memory=True, num_workers=0)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False, pin_memory=True, num_workers=0)

    return train_loader, dev_loader, test_loader


def build_datasets(config, subject_name, mode=False, oversample=True, drop_last=False):
    global X, y
    print("build datasets...")
    if not mode:
        X = np.load(config.data_path + '/' + subject_name + '_x.npy')
        y = np.load(config.data_path + '/' + subject_name + '_y.npy')
        y = y.reshape(-1)
        validate_epoch_arrays(X, y)
        print(X.shape)
        print(y.shape)
    train_data, dev_data, test_data = split_data(X, y)
    if oversample:
        train_indices = oversample_indices(train_data.iloc[:, -1].tolist())
        train_data = train_data.iloc[train_indices].reset_index(drop=True)
    train_loader = trans2DataLoader(train_data, config.batch_size, drop_last=drop_last)
    dev_loader = trans2DataLoader(dev_data, config.batch_size, shuffle=False, drop_last=False)
    test_loader = trans2DataLoader(test_data, config.batch_size, shuffle=False, drop_last=False)
    return train_loader, dev_loader, test_loader


def sync_offline2WB(path):
    data_list = os.listdir(path)
    for name in data_list:
        command = 'wandb sync wandb/c/' + name
        os.system(command)
    print("finish offline upload!")


if __name__ == '__main__':
    sync_offline2WB('wandb/c')
