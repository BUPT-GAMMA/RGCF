import logging
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from scipy.stats import spearmanr, kendalltau, pearsonr
import copy


def log_prob(dist, groundtruth):
    log_p = dist.log_prob(groundtruth)
    return -log_p.mean()

def ranking_loss(yq_hat_src, yq_src):
    shuffle_id = torch.randperm(len(yq_hat_src))
    yq_hat_dst = yq_hat_src[shuffle_id]
    yq_dst = yq_src[shuffle_id]
    label = torch.where(yq_src > yq_dst, 1.0, -1.0).cuda()
    return F.margin_ranking_loss(yq_hat_src, yq_hat_dst, label,margin=0.6)


def ranking_loss_bpr(yq_hat_src, yq_src):
    shuffle_id = torch.randperm(len(yq_hat_src))
    yq_hat_dst = yq_hat_src[shuffle_id]
    yq_dst = yq_src[shuffle_id]
    label = torch.where(yq_src > yq_dst, 1.0, 0.0).cuda()

    return F.binary_cross_entropy_with_logits(yq_hat_src-yq_hat_dst, label)

def ranking_loss_bpr_mix(yq_hat_src, yq_src_i, yq_src_j, lam):
    shuffle_id = torch.randperm(len(yq_hat_src))
    yq_hat_dst = yq_hat_src[shuffle_id]
    yq_dst_i = yq_src_i[shuffle_id]
    yq_dst_j = yq_src_j[shuffle_id]
    label_i = torch.where(yq_src_i > yq_dst_i, 1.0, 0.0).cuda()
    label_j = torch.where(yq_src_j > yq_dst_j, 1.0, 0.0).cuda()
    label = mix_up(label_i, label_j, lam)
    return F.binary_cross_entropy_with_logits(yq_hat_src - yq_hat_dst, label)

def listMLE(y_pred, y_true, eps=1e-10):
    loss = 0
    for i in range(len(y_pred)):
        preds_sorted_by_true = y_pred[i].unsqueeze(dim=0) #取出预测值中真实值top index对应的预测值
        max_pred_values, _ = preds_sorted_by_true.max(dim=1, keepdim=True)
        preds_sorted_by_true_minus_max = preds_sorted_by_true - max_pred_values
        cumsums = torch.cumsum(preds_sorted_by_true_minus_max.exp().flip(dims=[1]), dim=1).flip(dims=[1])
        observation_loss = torch.log(cumsums + eps) - preds_sorted_by_true_minus_max
        loss += torch.sum(observation_loss, dim=1)/len(y_pred)
    return loss

# def lambda_rank(y_pred, y_true):


loss_fn = {
    'mse': lambda yq_hat, yq: F.mse_loss(yq_hat, yq).float(),
    'logprob': lambda yq_hat, yq: log_prob(yq_hat, yq),
    'rank_loss': lambda yq_hat, yq: ranking_loss(yq_hat, yq),
    'rank_bpr': lambda yq_hat, yq: ranking_loss_bpr(yq_hat, yq),
    'rank_bpr_mix': lambda yq_hat_src, yq_src_i, yq_src_j, lam: ranking_loss_bpr_mix(yq_hat_src, yq_src_i, yq_src_j, lam),
    'listMLE': lambda y_pred, y_true: listMLE(y_pred, y_true)
}


def flat(v):
    if torch.is_tensor(v):
        return v.detach().cpu().numpy().reshape(-1)
    else:
        return v.reshape(-1)




metrics_fn = {
    'spearman': lambda yq_hat, yq: spearmanr(flat(yq_hat), flat(yq)),
    'pearsonr': lambda yq_hat, yq: pearsonr(flat(yq_hat), flat(yq)),
    'kendalltau': lambda yq_hat, yq: kendalltau(flat(yq_hat), flat(yq)),
}


feature_map = {
    'full': [0, 1, 2, 3, 4, 5, 6, 7, 8],
    'graph': [0, 1, 2, 3, 4, 7, 8],
    # 'no_bipartite': [0, 1,2, 3, 4, 5, 6, 20, 21],
    #'bipartite': [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
    'distribution': [5, 6]
}

class Log():
    def __init__(self, save_path, summary_steps, metrics, datasets, split, writer=None):
        self.save_path = save_path
        self.metrics = metrics
        self.datasets = datasets
        self.summary_steps = summary_steps
        self.split = split
        self.writer = writer

        self.epi = []
        self.elems = {}
        for metric in metrics:
            self.elems[metric] = {dataset: [] for dataset in datasets}
        self.elems['mse_loss'] = {dataset: [] for dataset in datasets}


    def update_epi(self, i_epi):
        self.epi.append(i_epi)

    def update(self, i_epi, metric, dataset, val):
        self.elems[metric][dataset].append(val)
        if self.writer is not None:
            self.writer.add_scalar(f'{self.split}_{metric}/{dataset}', val, i_epi)

    def avg(self, i_epi, metric, is_print=True):
        v = 0.0
        cnt = 0
        for dataset in self.datasets:
            v += self.get(metric, dataset, i_epi)
            cnt += 1
        if self.writer is not None and is_print:
            self.writer.add_scalar(f'mean/{self.split}_{metric}', v / cnt, i_epi)
        return v / cnt

    def get(self, metric, dataset, i_epi):
        idx = self.epi.index(i_epi)
        return self.elems[metric][dataset][idx]

    def save(self):
        torch.save({
            'summary_steps': self.summary_steps,
            'episode': self.epi,
            'elems': self.elems
        },
        os.path.join(self.save_path, f'{self.split}_log_data.pt'))


class EarlyStopping(object):
    def __init__(self, patience=20, save_path=None):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.save_path = save_path

    def step(self, score, save_dict):
        if isinstance(score, tuple):
            score = score[0]
        save_path = os.path.join(self.save_path, 'checkpoint', f'max_corr.pt')
        if self.best_score is None:
            self.best_score = score
            if not os.path.exists(save_path):
                os.makedirs(os.path.join(self.save_path, 'checkpoint'))
            torch.save(save_dict, save_path)
            print(f'\n==> save {save_path}')
        elif score <= self.best_score:
            self.counter += 1
            if score == self.best_score:
                torch.save(save_dict, save_path)
            # print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        elif score > self.best_score:
                torch.save(save_dict, save_path)
                print(f'\n==> save {save_path}')
                self.best_score = score
                self.counter = 0
        print(f'patience ======================================={self.counter}')
        return self.early_stop



def set_logger(log_path):
    """Set the logger to log info in terminal and file `log_path`.
    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.
    Example:
    ```
    logging.info("Starting training...")
    ```
    Args:
        log_path: (string) where to log
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path, 'w')
        file_handler.setFormatter(
            logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)


def mix_up(data_i,  data_j, lam):
    return lam * data_i + (1 - lam) * data_j

def denorm(lat, maxv, minv):
    return lat * (maxv - minv) + minv


def normalization(performance, index=None):
    if index != None:
        min_val = min(performance[index])
        max_val = max(performance[index])
    else:
        min_val = min(performance)
        max_val = max(performance)
    performance = (performance - min_val) / (max_val - min_val)
    return performance

