# coding: UTF-8
import time
import argparse
import torch
from importlib import import_module

from train_test import train_MAG, has_completed_subject
from utils import get_time_dif, build_datasets, set_seed, build_cross_datasets
from dataset_contract import resolve_subject_names
from train_test import init_network
import os
import warnings
import wandb

parser = argparse.ArgumentParser(description='bci project')
parser.add_argument('--model', type=str, default='InceptSADNet',
                    help='Model module name, optionally prefixed with its package (e.g. models.ablations.InceptSADNet_no_se)')
# two mode：inner, outer
# for one subject to split the train and test dataset
# leave-one-subject-out
parser.add_argument('--mode', type=str, default='False')
parser.add_argument('--compile', action='store_true', help='Enable torch.compile for potential speedup (PyTorch 2.0+)')
parser.add_argument('--resume', action='store_true', help='Resume training from last checkpoint')
parser.add_argument('--seed', type=int, default=27, help='Random seed for model initialization and training')
args = parser.parse_args()


def get_model_module(model_name):
    module_name = model_name if '.' in model_name else 'models.' + model_name
    return import_module(module_name)


# Enable cuDNN benchmark for faster performance on fixed input sizes
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    # Enable TF32 for Ampere+ GPUs (RTX 30xx/40xx) - ~2x matmul speedup with no code changes
    torch.set_float32_matmul_precision('high')


# get all subjects path
def get_all_subjects(dataset):
    """Return all configured subject folders, failing rather than using a partial cohort."""
    raw_dir = os.path.join(dataset, "raw")
    if not os.path.isdir(raw_dir):
        raise FileNotFoundError(f"Raw dataset directory does not exist: {raw_dir}")

    subject_names = [
        name for name in os.listdir(raw_dir)
        if os.path.isdir(os.path.join(raw_dir, name))
    ]
    ordered_names = resolve_subject_names(subject_names)
    return [os.path.join(raw_dir, name) for name in ordered_names]

# inner subject
def inner_subject_train():
    dataset = 'data'
    model_name = [args.model]
    all_subjects = get_all_subjects(dataset)
    for index in range(len(model_name)):
        x = get_model_module(model_name[index])
        config = x.Config(dataset)
        start_time = time.time()

        # loading data
        data_list = all_subjects
        for subject_name in data_list:
            subject_id = subject_name[-3:]
            set_seed(args.seed)
            
            # Check if this subject is already fully trained (skip if resume and completed)
            if args.resume:
                if has_completed_subject(config, subject_id, "inter", args.seed):
                    print(f"[RESUME] Subject {subject_id} already completed, skipping...")
                    continue
            
            drop_last = True
            print('loading data...')
            config.data_path = subject_name
            print(config.data_path)
            train_iter, dev_iter, test_iter = build_datasets(config, subject_id, mode=False, oversample=True,
                                                             drop_last=drop_last)
            time_dif = get_time_dif(start_time)
            print("loading data usage:", time_dif)

            # train and test
            model = x.Model(config).to(config.device)
            init_network(model, seed=args.seed)
            
            if args.compile and hasattr(torch, 'compile'):
                print("Compiling model...")
                try:
                    model = torch.compile(model)
                except Exception as e:
                    print(f"torch.compile failed, falling back to eager mode: {e}")
            
            print(model.parameters)
            train_MAG(config, model, train_iter, dev_iter, test_iter, subject_id, "inter",
                      resume=args.resume, seed=args.seed)


# cross subject
def leave_one_subject_out():
    dataset = 'data'
    model_name = [args.model]
    # 首先得到所有被试的文件名
    all_subjects = get_all_subjects(dataset)
    for index in range(len(model_name)):
        data_list = all_subjects

        x = get_model_module(model_name[index])
        config = x.Config(dataset)
        start_time = time.time()

        # 选一个数据集作为测试集，其余的混合后作为训练集和验证集
        for subject_name in data_list:
            subject_id = subject_name[-3:]
            set_seed(args.seed)
            
            # Check if this subject is already fully trained (skip if resume and completed)
            if args.resume:
                if has_completed_subject(config, subject_id, "cross", args.seed):
                    print(f"[RESUME] Subject {subject_id} already completed, skipping...")
                    continue
            
            print('loading data...')
            config.data_path = subject_name
            print(config.data_path)
            train_iter, dev_iter, test_iter = build_cross_datasets(config, subject_id)
            time_dif = get_time_dif(start_time)
            print("loading data usage:", time_dif)

            # train and test
            model = x.Model(config).to(config.device)
            init_network(model, seed=args.seed)
            
            if args.compile and hasattr(torch, 'compile'):
                print("Compiling model...")
                try:
                    model = torch.compile(model)
                except Exception as e:
                    print(f"torch.compile failed, falling back to eager mode: {e}")
            
            print(model.parameters)
            train_MAG(config, model, train_iter, dev_iter, test_iter, subject_id, "cross",
                      resume=args.resume, seed=args.seed)


if __name__ == '__main__':
    set_seed(args.seed)
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
