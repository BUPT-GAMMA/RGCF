import torch.nn as nn
import torch
from graphgym.register import register, register_loss
from graphgym.config import cfg
import torch.nn.functional as F

def loss_example(pred, true):
    if cfg.model.loss_fun == 'smoothl1':
        l1_loss = nn.SmoothL1Loss()
        loss = l1_loss(pred, true)
        return loss, pred

def loss_mae(pred, true):
    if cfg.model.loss_fun == 'mae':
        l1_loss = nn.L1Loss(reduction=cfg.model.size_average)
        loss = l1_loss(pred, true)
        return loss, pred

def loss_bpr(pred, true, gamma=0):
    if cfg.model.loss_fun == 'bpr':
        pos_score = pred[:, 0].unsqueeze(-1)
        neg_score = pred[:, 1:]
        loss = -torch.log(gamma + torch.sigmoid(pos_score - neg_score)).mean()
        return loss

def loss_hgl(pred, true, margin = 1):
    if cfg.model.loss_fun == 'hgl':
        pos_score = pred[:, 0].unsqueeze(-1)
        neg_score = pred[:, 1:]
        loss = torch.relu(margin + neg_score-pos_score ).mean()
        return loss

def loss_softmax_cross_entropy(pred, true):
    if cfg.model.loss_fun == 'sce':
        probs = F.softmax(pred, dim=1)
        hit_probs = probs[:, 0]
        loss = -torch.log(hit_probs).mean()
        return loss

register_loss('smoothl1', loss_example)
register_loss('mae', loss_mae)
