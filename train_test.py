# coding: UTF-8
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
import time
from utils import get_time_dif, build_datasets
import wandb
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve, auc
from torch.utils.tensorboard import SummaryWriter


# 使用xavier初始化网络权重
def init_network(model, method='xavier', exclude='embedding', seed=27):
    for name, w in model.named_parameters():
        if exclude not in name:
            if 'weight' in name:
                if method == 'xavier':
                    nn.init.xavier_normal_(w)
                elif method == 'kaiming':
                    nn.init.kaiming_normal_(w)
                else:
                    nn.init.normal_(w)
            elif 'bias' in name:
                nn.init.constant_(w, 0)
            else:
                pass


def train_epoch(config, model, train_iter, optimizer):
    model.train()
    tr_loss = 0
    total_steps = 0
    for step, (X, y) in enumerate(tqdm(train_iter, desc='Iteration')):
        X = X.to(config.device, non_blocking=True)
        y = y.to(config.device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        y_hat = model(X)
        loss = F.cross_entropy(y_hat, y.squeeze().long())
        loss.backward()
        optimizer.step()
        tr_loss += loss.item()
        total_steps += 1
    return tr_loss / total_steps


def eval_epoch(config, model, dev_iter):
    model.eval()
    dev_loss = 0
    total_steps = 0
    with torch.no_grad():
        for step, (X, y) in enumerate(tqdm(dev_iter, desc='Iteration')):
            X = X.to(config.device, non_blocking=True)
            y = y.to(config.device, non_blocking=True)
            y_hat = model(X)
            loss = F.cross_entropy(y_hat, y.squeeze().long())
            dev_loss += loss.item()
            total_steps += 1
    return dev_loss / total_steps


def test_epoch(config, model, test_iter):
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    y_hats_all = []
    with torch.no_grad():
        for X, y in test_iter:
            X = X.to(config.device, non_blocking=True)
            y = y.to(config.device, non_blocking=True)
            y_hat = model(X)
            loss = F.cross_entropy(y_hat, y.squeeze().long())
            loss_total += loss
            labels = y.data.cpu().numpy()
            # predict = torch.max(y_hat.data, 1)[1].cpu().numpy()
            predict = y_hat.cpu().numpy().argmax(axis=-1)
            # print(y_hat.data)
            # print(predict)
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predict)
            y_hats_all.append(y_hat.cpu().numpy())
    return predict_all, labels_all, np.array(y_hats_all).reshape(-1, 3)


def test_model(config, model, test_iter, use_zero=False):
    preds, y_test, y_hat = test_epoch(config, model, test_iter)
    # print(preds.shape)
    # print(y_test.shape)
    # print(y_hat.shape)
    # report = metrics.classification_report(y_test, preds, target_names=config.class_list, digits=4)
    # confusion = metrics.confusion_matrix(y_test, preds)
    # print("Precision, Recall and F1-Score...")
    # print(report)
    # print("Confusion Matrix...")
    # print(confusion)
    # non_zeros = np.array(
    #     [i for i, e in enumerate(y_test) if e != 0 or use_zero])
    #
    # preds = preds[non_zeros]
    # y_test = y_test[non_zeros]

    mae = np.mean(np.absolute(preds - y_test))
    corr = np.corrcoef(preds, y_test)[0][1]
    std = np.std(preds, ddof=1)

    # preds = preds >= 0
    # y_test = y_test >= 0
    # print(preds)
    # print(y_test)

    f_score = f1_score(y_test, preds, average="weighted")
    acc = accuracy_score(y_test, preds)
    # multi-class auc-roc calculation
    # 这里y_hat经过softmax的概率化后使得其加和为1
    try:
        auc = roc_auc_score(y_test, F.softmax(torch.Tensor(y_hat), dim=1).numpy(), multi_class='ovr', average='weighted', labels=list(range(config.num_classes)))
    except ValueError:
        auc = 0.5  # Default to 0.5 if AUC cannot be calculated (e.g. only one class in y_test)

    # multi-label的AUC-ROC曲线需要二值化
    # auc = roc_auc_score(y_test, preds)
    # preds = label_binarize(preds, classes=[0, 1, 2])
    # n_classes = preds.shape[1]
    #
    # # 计算每一类ROC
    # fpr = dict()
    # tpr = dict()
    # roc_auc = dict()
    # for i in range(n_classes):
    #     fpr[i], tpr[i], _ = roc_curve(y_test[:, i].reshape(1, -1).unsqueeze(), preds[:, i].reshape(1, -1).unsqueeze())
    #     roc_auc[i] = auc(fpr[i], tpr[i])
    #
    # # method 2
    # fpr['micro'], tpr['micro'], _ = roc_curve(y_test.ravel(), preds.ravel())
    # roc_auc['micro'] = auc(fpr['micro'], tpr['micro'])
    #
    # # method 1
    # all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    # mean_tpr = np.zeros_like(all_fpr)
    # for i in range(n_classes):
    #     mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    # mean_tpr /= n_classes
    # fpr['macro'] = all_fpr
    # tpr['macro'] = mean_tpr
    # roc_auc['macro'] = auc(fpr['macro'], tpr['macro'])

    return acc, mae, corr, f_score, auc


# reference MAG
def train_MAG(config, model, train_iter, dev_iter, test_iter, subject_name, mode="inter", resume=False):
    run = wandb.init(project="DSADNet", entity=None, config=config.__dict__, reinit=True)
    wandb.config.update(config.__dict__, allow_val_change=True)  # 记录config中设置的参数
    valid_losses = []
    test_f1s = []
    test_aucs = []
    best_f1 = 0
    best_auc = 0
    start_epoch = 0
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    # to get single model for certain subject
    import os
    # Use local variables, do not modify config in place
    save_dir = os.path.join(config.f1_save_path, config.model_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    current_f1_save_path = os.path.join(save_dir, f'f1_{subject_name}_{mode}.ckpt')
    current_auc_save_path = os.path.join(save_dir, f'auc_{subject_name}_{mode}.ckpt')
    checkpoint_path = os.path.join(save_dir, f'checkpoint_{subject_name}_{mode}.pt')
    
    # --- Resume from checkpoint ---
    if resume and os.path.exists(checkpoint_path):
        print(f"[RESUME] Loading checkpoint for {subject_name} ({mode})...")
        checkpoint = torch.load(checkpoint_path, map_location=config.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1  # resume from the NEXT epoch
        best_f1 = checkpoint['best_f1']
        best_auc = checkpoint['best_auc']
        valid_losses = checkpoint['valid_losses']
        test_f1s = checkpoint['test_f1s']
        test_aucs = checkpoint['test_aucs']
        print(f"[RESUME] Resuming from epoch {start_epoch}/{config.num_epoch} "
              f"(best_f1={best_f1:.4f}, best_auc={best_auc:.4f})")
    
    for i in range(start_epoch, int(config.num_epoch)):
        train_loss = train_epoch(config, model, train_iter, optimizer)
        valid_loss = eval_epoch(config, model, dev_iter)
        test_acc, test_mae, test_corr, test_f1, test_auc = test_model(config, model, test_iter)
        print(
            "epoch:{}, train_loss:{}, valid_loss:{}, test_f1:{}, test_auc:{}".format(
                i, train_loss, valid_loss, test_f1, test_auc
            )
        )
        valid_losses.append(valid_loss)
        test_f1s.append(test_f1)
        test_aucs.append(test_auc)
        if test_f1 > best_f1:
            best_f1 = test_f1
            torch.save(model.state_dict(), current_f1_save_path)
        if test_auc > best_auc:
            best_auc = test_auc
            torch.save(model.state_dict(), current_auc_save_path)
        
        # --- Save checkpoint every epoch for crash recovery ---
        torch.save({
            'epoch': i,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_f1': best_f1,
            'best_auc': best_auc,
            'valid_losses': valid_losses,
            'test_f1s': test_f1s,
            'test_aucs': test_aucs,
        }, checkpoint_path)
        
        wandb.log(
            (
                {
                    "train_loss": train_loss,
                    "valid_loss": valid_loss,
                    "test_acc": test_acc,
                    "test_mae": test_mae,
                    "test_corr": test_corr,
                    "test_f_score": test_f1,
                    "best_valid_loss": min(valid_losses),
                    "best_test_f1": max(test_f1s),
                    "best_test_auc": max(test_aucs),
                }
            )
        )
    
    # Training complete for this subject - clean up checkpoint
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        print(f"[CHECKPOINT] Removed completed checkpoint for {subject_name}")
    
    wandb.log(({"avg_test_f1": sum(test_f1s) / len(test_f1s)}))
    wandb.log(({"avg_test_auc": sum(test_aucs) / len(test_aucs)}))
    run.finish()
    f = open(config.model_name+'_result.txt', 'a+')
    f.write('best_test_f1:{}, best_test_auc:{}\n'.format(max(test_f1s), max(test_aucs)))
    f.close()
    return max(test_f1s)
