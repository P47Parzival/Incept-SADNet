# coding: UTF-8
import time
import argparse
import torch
from importlib import import_module

from train_test import train_MAG
from utils import get_time_dif, build_datasets, set_seed, build_cross_datasets
from train_test import init_network
import os
import warnings
import wandb

parser = argparse.ArgumentParser(description='bci project')
parser.add_argument('--model', type=str, default='CompactCNN')
# two mode：inner, outer
# for one subject to split the train and test dataset
# leave-one-subject-out
parser.add_argument('--mode', type=str, default='False')
parser.add_argument('--compile', action='store_true', help='Enable torch.compile for potential speedup (PyTorch 2.0+)')
parser.add_argument('--resume', action='store_true', help='Resume training from last checkpoint')
args = parser.parse_args()

# Enable cuDNN benchmark for faster performance on fixed input sizes
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    # Enable TF32 for Ampere+ GPUs (RTX 30xx/40xx) - ~2x matmul speedup with no code changes
    torch.set_float32_matmul_precision('high')


# get all subjects path
def get_all_subjects(dataset):
    # Filter to only use the 16 subjects from the SADNet paper
    valid_subjects = ['s01', 's05', 's09', 's13', 's22', 's31', 's35', 's40', 
                      's41', 's42', 's43', 's44', 's45', 's49', 's50', 's53']
    data_list = os.listdir(dataset + '/raw')
    all_subjects = []
    for subject_name in data_list:
        if subject_name.lower() in valid_subjects:
            data_path = dataset + '/raw/' + subject_name  # data/raw/sxx
            all_subjects.append(data_path)
    return sorted(all_subjects)

# inner subject
def inner_subject_train():
    dataset = 'data'
    # model_name = ['CompactCNN', 'DeepConvNet', 'EEGInception', 'EEGNet', 'EEGResNet', 'InterpretableCNN',
    #               'ShallowConvNet']
    model_name = ['InceptSADNet']
    all_subjects = get_all_subjects(dataset)
    for index in range(len(model_name)):
        x = import_module('models.' + model_name[index])
        config = x.Config(dataset)
        start_time = time.time()

        # loading data
        # for 27 subjects
        # data_list = os.listdir(dataset + '/raw')
        data_list = all_subjects
        # print(data_list)
        for subject_name in data_list:
            subject_id = subject_name[-3:]
            
            # Check if this subject is already fully trained (skip if resume and completed)
            if args.resume:
                save_dir = os.path.join(config.f1_save_path, config.model_name)
                checkpoint_path = os.path.join(save_dir, f'checkpoint_{subject_id}_inter.pt')
                has_checkpoint = os.path.exists(checkpoint_path)
                # If no checkpoint exists, check if subject was already completed
                if not has_checkpoint:
                    f1_path = os.path.join(save_dir, f'f1_{subject_id}_inter.ckpt')
                    auc_path = os.path.join(save_dir, f'auc_{subject_id}_inter.ckpt')
                    if os.path.exists(f1_path) and os.path.exists(auc_path):
                        print(f"[RESUME] Subject {subject_id} already completed, skipping...")
                        continue
            
            drop_last = True
            print('loading data...')
            config.data_path = subject_name
            print(config.data_path)
            train_iter, test_iter, dev_iter = build_datasets(config, subject_id, mode=False, oversample=True,
                                                             drop_last=drop_last)
            time_dif = get_time_dif(start_time)
            print("loading data usage:", time_dif)

            # train and test
            model = x.Model(config).to(config.device)
            
            if args.compile and hasattr(torch, 'compile'):
                print("Compiling model...")
                try:
                    model = torch.compile(model)
                except Exception as e:
                    print(f"torch.compile failed, falling back to eager mode: {e}")
            
            print(model.parameters)
            train_MAG(config, model, train_iter, dev_iter, test_iter, subject_id, "inter", resume=args.resume)


# cross subject
def leave_one_subject_out():
    dataset = 'data'
    model_name = ['InceptSADNet']
    # 首先得到所有被试的文件名
    all_subjects = get_all_subjects(dataset)
    # 选model
    for index in range(len(model_name)):
        data_list = all_subjects

        x = import_module('models.' + model_name[index])
        config = x.Config(dataset)
        start_time = time.time()

        # 选一个数据集作为测试集，其余的混合后作为训练集和验证集
        for subject_name in data_list:
            subject_id = subject_name[-3:]
            
            # Check if this subject is already fully trained (skip if resume and completed)
            if args.resume:
                save_dir = os.path.join(config.f1_save_path, config.model_name)
                checkpoint_path = os.path.join(save_dir, f'checkpoint_{subject_id}_cross.pt')
                has_checkpoint = os.path.exists(checkpoint_path)
                if not has_checkpoint:
                    f1_path = os.path.join(save_dir, f'f1_{subject_id}_cross.ckpt')
                    auc_path = os.path.join(save_dir, f'auc_{subject_id}_cross.ckpt')
                    if os.path.exists(f1_path) and os.path.exists(auc_path):
                        print(f"[RESUME] Subject {subject_id} already completed, skipping...")
                        continue
            
            print('loading data...')
            config.data_path = subject_name
            print(config.data_path)
            train_iter, test_iter, dev_iter = build_cross_datasets(config, subject_id)
            time_dif = get_time_dif(start_time)
            print("loading data usage:", time_dif)

            # train and test
            model = x.Model(config).to(config.device)
            
            if args.compile and hasattr(torch, 'compile'):
                print("Compiling model...")
                try:
                    model = torch.compile(model)
                except Exception as e:
                    print(f"torch.compile failed, falling back to eager mode: {e}")
            
            print(model.parameters)
            train_MAG(config, model, train_iter, dev_iter, test_iter, subject_id, "cross", resume=args.resume)


if __name__ == '__main__':
    set_seed()
    # ignore the warning
    warnings.filterwarnings('ignore')
    # offline
    # API Key handling
    api_key = os.environ.get("WANDB_API_KEY")
    if not api_key:
        print("WandB API Key not found in environment variables.")
        api_key = input("Please enter your WandB API Key (or press Enter to skip/use offline mode): ").strip()
        if api_key:
            os.environ["WANDB_API_KEY"] = api_key
    
    if not os.environ.get("WANDB_API_KEY"):
        print("No API Key provided. Running in offline mode.")
        os.environ['WANDB_MODE'] = 'offline'
    else:
        os.environ['WANDB_MODE'] = 'online'
    if args.mode == 'True':
        leave_one_subject_out()
    else:
        inner_subject_train()
