import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from run import get_all_subjects
from dataset_contract import (
    EPOCH_TMAX,
    EPOCH_TMIN,
    N_EEG_CHANNELS,
    N_EPOCH_SAMPLES,
    SAMPLE_RATE,
    classify_response_time,
    resolve_eeg_channel_names,
    stable_seed_for_name,
    validate_epoch_arrays,
)


# 裁剪event发生前后的数据（tmin, tmax），事件发生的时间结点相对定位为0
# 只裁剪event发生前2s的数据
tmin = EPOCH_TMIN
tmax = EPOCH_TMAX
event_id = {'deviation/left': 1, 'deviation/right': 2, 'response/onset': 3,
            'response/offset': 4}
file_path = 'data/raw'

# sample windows size
start = 90

# sampel rate
freq = SAMPLE_RATE


# convert .set to Epochs
def convert_raw2epoch(file_name, tmin, tmax, event_id):
    raw = mne.io.read_raw_eeglab(file_name, preload=True)
    raw.resample(freq)
    events_from_annot, event_dict = mne.events_from_annotations(raw)
    epochs = mne.Epochs(raw, events_from_annot, event_id=event_id, tmin=tmin, tmax=tmax, preload=True)
    return epochs


# find deviation events
def find_events(epochs):
    Y = []
    for index in range(epochs.events.shape[0] - 1):
        temp1 = epochs.events[index]
        next_index = index + 1
        temp2 = epochs.events[next_index]
        if temp1[2] == 1 or temp1[2] == 2:
            if temp2[2] == 3:
                # index表示epoch的匹配，后一个表示时间差
                Y.append([index, int(temp2[0]) - int(temp1[0])])
    return Y


# find start point of first event
def find_start_pos(efa):
    index = 0
    for epoch in efa:
        if epoch[0] > start * freq:
            break
        index += 1
    return index


# 同样需要排除RT<50的误操作
# exclude the wrong operation that RT < 50
def find_events_diff(efa):
    Y = []
    for index in range(efa.shape[0] - 1):
        temp1 = efa[index]
        next_index = index + 1
        temp2 = efa[next_index]
        if temp1[2] == 1 or temp1[2] == 2:
            if temp2[2] == 3:
                if int(temp2[0]) - int(temp1[0]) > 50:
                    # index表示epoch的匹配，第二个表示response相应的绝对时间，后一个表示时间差
                    Y.append([next_index, temp2[0], int(temp2[0]) - int(temp1[0])])
    return Y


# t表示事件开始的点
def find_pos(efa, t):
    index = 0
    for epoch in efa:
        if epoch[1] > t:
            break
        index += 1
    return index


# Y: from find_events_diff()
# format: [index, response_time_point, time_diff]
def cal_global_RTs(Y):
    incremental = start * freq
    start_time = 0
    # 找到开始的epoch_index
    start_index = find_pos(Y, incremental)
    # deviation发生的时间
    end_time = Y[start_index][1]
    global_RTs = []
    # 这里从90s后的events开始计算global_RTs
    for index in range(start_index, len(Y)):
        cal_rt = 0
        c = 0
        ii = 0
        # 累计该deviation发生前90s内的平均偏移量
        for ind, i in enumerate(Y):
            if start_time <= i[1] <= end_time:
                cal_rt += i[2]
                c += 1
                ii = ind
            # 当时间超出规定后，更新窗口
            elif i[1] > end_time:
                end_time = i[1]
                start_time = end_time - incremental
                break
        # 剪掉自身
        cal_rt -= Y[ii][2]
        c -= 1
        if c != 0:
            global_RTs.append([ii, cal_rt // c])
        # 如果该时间段没有deviation发生，则global为0
        else:
            global_RTs.append([ii, 0])
    return global_RTs


def clf_labels(epochs, percentile=0.05):
    labels = []
    efa = epochs.events
    local_RTs = find_events_diff(efa)  # local_RT
    global_RTs = cal_global_RTs(local_RTs)  # global_RTs
    values = []
    # 得到local_RTs的time_diff
    for i in local_RTs:
        values.append(i[2])
    values = sorted(values)
    index = int(len(values) * percentile)
    # alert_RT取百分之五分位数
    alert_RT = values[index]
    incremental = start * freq
    start_index = find_pos(local_RTs, incremental)
    for index, RT in enumerate(global_RTs):
        global_RT = RT[1]
        local_RT = local_RTs[index + start_index][2]
        labels.append([index, classify_response_time(global_RT, local_RT, alert_RT)])
    return labels


# 将同一个被试的所有数据整合到Ys中(仅包含events)
def analysis_subjects(file_path, tmin, tmax, event_id):
    Ys = []
    for file in sorted(os.listdir(file_path)):
        print(file)
        if file[-4:] == ".set":
            epochs = convert_raw2epoch(file_path + '/' + file, tmin, tmax, event_id)
            Y = find_events(epochs)
            for y in Y:
                Ys.append(y)
    return Ys


# 排除RT<50的，然后求5%分位数得到alert_RT
# 得到被试所有数据的alert_RT
def cal_alert_RT(file_path, tmin, tmax, event_id, percentile=0.05):
    Ys = analysis_subjects(file_path, tmin, tmax, event_id)
    total = 0
    count = 0
    clear = []
    for y in Ys:
        if y[1] > 50:
            total += y[1]
            count += 1
            clear.append(y[1])
    avg_total = total // count
    clear = sorted(clear)
    pos = len(clear) * percentile  # 百分之五分位数作为alert_RT
    return avg_total, clear[int(pos)]


def cal_local_RTs(epochs):
    efa = epochs.events
    return find_events_diff(efa)


# 将存储下来的.npz文件进行读取，过采样/欠采样并制作训练集和测试集
def load_datasets(subject, underfit_sample=False):
    X = np.load(subject + '_x.npy')
    y = np.load(subject + '_y.npy')
    return X, y


# 为所有的被试准备数据集
def clf_labels_for_all_subjects(file_path, tmin, tmax, event_id, incremental, percentile=0.05):
    subject_list = sorted(os.listdir(file_path))
    for subject in subject_list:
        subject_path = file_path + '/' + subject
        print(subject_path)
        X, y = clf_labels_for_subjects(subject_path, tmin, tmax, event_id, incremental)
        print(X.shape)
        print(y.shape)
        np.save(subject_path + '/' + subject + '_x', X)
        np.save(subject_path + '/' + subject + '_y', y)
        print("save " + str(subject) + " finished.")


# 得到X， y
def clf_labels_for_subjects(file_path, tmin, tmax, event_id, incremental, percentile=0.05):
    avg_total, alert_RT = cal_alert_RT(file_path, tmin, tmax, event_id, percentile)
    file_list = sorted(os.listdir(file_path))
    X = []
    y = []
    for file_name in file_list:
        print(file_name)
        if file_name[-4:] == '.set':
            epochs = convert_raw2epoch(file_path + '/' + file_name, tmin, tmax, event_id)
            epochs.pick(resolve_eeg_channel_names(epochs.ch_names))
            df = epochs.to_data_frame(index=['condition', 'epoch', 'time'])
            local_RTs = cal_local_RTs(epochs)
            global_RTs = cal_global_RTs(local_RTs)
            start_index = find_pos(local_RTs, incremental)
            # 一个数据集.set
            labels = []
            for index, RT in enumerate(global_RTs):
                epoch = RT[0]
                global_RT = RT[1]
                local_RT = local_RTs[index + start_index][2]
                labels.append(
                    [epoch, classify_response_time(global_RT, local_RT, alert_RT)]
                )
            for epoch, label in labels:
                X.append(df.xs(epoch, level='epoch').to_numpy())
                y.append(label)
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)
    validate_epoch_arrays(X, y)
    return X, y


def make_cross_datasets():
    data_path = 'data'
    if not os.path.exists('data/cross'):
        os.makedirs('data/cross')
    all_subjects_path = get_all_subjects(data_path)
    # 每个被试都需要作为训练集一次
    for subject_name in all_subjects_path:
        train_zeros, train_ones, train_twos, dev_zeros, dev_ones, dev_twos = get_cross_subject(subject_name[-3:],
                                                                                               all_subjects_path)
        train_x = np.concatenate((train_zeros, train_ones))
        train_x = np.concatenate((train_x, train_twos))
        train_y = np.concatenate((np.zeros(train_zeros.shape[0]), np.ones(train_ones.shape[0])))
        train_y = np.concatenate((train_y, np.full(train_twos.shape[0], 2)))

        dev_x = np.concatenate((dev_zeros, dev_ones))
        dev_x = np.concatenate((dev_x, dev_twos))
        dev_y = np.concatenate((np.zeros(dev_zeros.shape[0]), np.ones(dev_ones.shape[0])))
        dev_y = np.concatenate((dev_y, np.full(dev_twos.shape[0], 2)))

        test_x = np.load(subject_name + '/' + subject_name[-3:] + '_x.npy')
        test_y = np.load(subject_name + '/' + subject_name[-3:] + '_y.npy')

        # save
        np.save('data/cross/train_' + subject_name[-3:] + '_x', train_x)
        np.save('data/cross/train_' + subject_name[-3:] + '_y', train_y)
        np.save('data/cross/dev_' + subject_name[-3:] + '_x', dev_x)
        np.save('data/cross/dev_' + subject_name[-3:] + '_y', dev_y)
        np.save('data/cross/test_' + subject_name[-3:] + '_x', test_x)
        np.save('data/cross/test_' + subject_name[-3:] + '_y', test_y)
        print("save " + str(subject_name[-3:]) + " finished.")


def get_cross_subject(subject_name, all_subjects):
    # 这里的subject_name表示用做测试集的被试数据
    dev_zeros = []
    dev_ones = []
    dev_twos = []

    train_zeros = []
    train_ones = []
    train_twos = []

    for name in all_subjects:
        if name[-3:] == subject_name:
            print("{} is used for test".format(name[-3:]))
        else:
            # Force float32 to save memory
            all_zero, all_one, all_two = get_single_subject(name)

            train_0_data, dev_0_data = split_data(all_zero)
            train_1_data, dev_1_data = split_data(all_one)
            train_2_data, dev_2_data = split_data(all_two)

            # Extend lists instead of custom flatten function
            train_zeros.extend(train_0_data)
            train_ones.extend(train_1_data)
            train_twos.extend(train_2_data)

            dev_zeros.extend(dev_0_data)
            dev_ones.extend(dev_1_data)
            dev_twos.extend(dev_2_data)

    return tuple(
        _trial_array(samples)
        for samples in (
            train_zeros,
            train_ones,
            train_twos,
            dev_zeros,
            dev_ones,
            dev_twos,
        )
    )


def _trial_array(samples):
    if not samples:
        return np.empty((0, N_EPOCH_SAMPLES, N_EEG_CHANNELS), dtype=np.float32)
    return np.asarray(samples, dtype=np.float32)


def flatten(data, data_list):
    for i in data:
        data_list.append(i)
    return data_list


# 按照9：1划分数据得到训练集和验证集
def split_data(data):
    num = int(data.shape[0] * 0.9)
    train_data = data[:num]
    dev_data = data[num:]
    return train_data, dev_data


def get_single_subject(subject_name):
    X = np.load(subject_name + '/' + subject_name[-3:] + '_x.npy')
    y = np.load(subject_name + '/' + subject_name[-3:] + '_y.npy').reshape(-1)
    validate_epoch_arrays(X, y)
    rng = np.random.RandomState(
        stable_seed_for_name(os.path.basename(subject_name))
    )
    return tuple(
        rng.permutation(X[y == label]).astype(np.float32)
        for label in range(3)
    )


if __name__ == '__main__':
    # incremental = start * freq
    # clf_labels_for_all_subjects(file_path, tmin, tmax, event_id, incremental, percentile=0.05)
    make_cross_datasets()