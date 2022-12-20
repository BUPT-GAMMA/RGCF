import torch
import torch.nn as nn
import torch.nn.functional as F

from graphgym.contrib.loss import *
import graphgym.register as register
from graphgym.config import cfg

def loss_bpr(pred, true, gamma=1e-10):
    pos_score = pred[:, 0].unsqueeze(-1)
    neg_score = pred[:, 1:]
    loss = -torch.log(gamma + torch.sigmoid(pos_score - neg_score)).mean()
    return loss

def loss_hgl(pred, true, margin = 1):
    pos_score = pred[:, 0].unsqueeze(-1)
    neg_score = pred[:, 1:]
    loss = torch.relu(margin + neg_score-pos_score ).mean()
    return loss

def loss_softmax_cross_entropy(pred, true):
    probs = F.softmax(pred, dim=1)
    hit_probs = probs[:, 0]
    loss = -torch.log(hit_probs).mean()
    return loss

def loss_bce(pred, true):
    """
    :param y_true: Labels
    :param y_pred: Predicted result
    """
    labels = torch.zeros(size=pred.shape, device=pred.device)
    labels[:, 0] = 1
    logits = pred.flatten()
    labels = labels.flatten()
    loss = F.binary_cross_entropy_with_logits(logits, labels, reduction="mean")
    return loss

def loss_sce(y_pred, y_true):
    """
    :param y_true: Labels
    :param y_pred: Predicted result.
    """
    probs = F.softmax(y_pred, dim=1)
    hit_probs = probs[:, 0]
    loss = -torch.log(hit_probs).mean()
    return loss

def loss_mse_rank(y_pred, y_true):
    """
    :param y_pred: prdicted values of shape (batch_size, 1 + num_negs)
    :param y_true: true labels of shape (batch_size, 1 + num_negs)
    """
    pos_logits = y_pred[:, 0]
    pos_loss = torch.pow(pos_logits - 1, 2) / 2
    neg_logits = y_pred[:, 1:]
    neg_loss = torch.pow(neg_logits, 2).sum(dim=-1) / 2
    loss = pos_loss + neg_loss
    return loss.mean()

def loss_ccl(y_pred, y_true, margin=0, negative_weight=None):
    """
    :param y_pred: prdicted values of shape (batch_size, 1 + num_negs)
    :param y_true: true labels of shape (batch_size, 1 + num_negs)
    """
    pos_logits = y_pred[:, 0]
    pos_loss = torch.relu(1 - pos_logits)
    neg_logits = y_pred[:, 1:]
    neg_loss = torch.relu(neg_logits - margin)
    if negative_weight:
        loss = pos_loss + neg_loss.mean(dim=-1) * negative_weight
    else:
        loss = pos_loss + neg_loss.sum(dim=-1)
    return loss.mean()

def compute_loss(pred, true):
    '''
    :param pred: unnormalized prediction
    :param true: label
    :return: loss, normalized prediction score
    '''
    bce_loss = nn.BCEWithLogitsLoss(reduction=cfg.model.size_average)
    mse_loss = nn.MSELoss(reduction=cfg.model.size_average)

    # default manipulation for pred and true
    # can be skipped if special loss computation is needed
    # if multi task binary classification, treat as flatten binary
    if true.ndim > 1 and cfg.model.loss_fun == 'cross_entropy':
        pred, true = torch.flatten(pred), torch.flatten(true)
    pred = pred.squeeze(-1) if pred.ndim > 1 else pred
    true = true.squeeze(-1) if true.ndim > 1 else true

    pred = pred.float()
    true = true.float()

    # Try to load customized loss
    for func in register.loss_dict.values():
        value = func(pred, true)
        if value is not None:
            return value

    if cfg.model.loss_fun == 'cross_entropy':
        # multiclass
        if pred.ndim > 1:
            pred = F.log_softmax(pred, dim=-1)
            return F.nll_loss(pred, true), pred
        # binary
        else:
            true = true.float()
            return bce_loss(pred, true), torch.sigmoid(pred)
    elif cfg.model.loss_fun == 'mse':
        true = true.float()
        return mse_loss(pred, true), pred
    elif cfg.model.loss_fun == 'bpr':
        true = true.float()
        return loss_bpr(pred, true), torch.sigmoid(pred[:,0])
    elif cfg.model.loss_fun == 'hgl':
        true = true.float()
        return loss_hgl(pred, true), torch.sigmoid(pred[:, 0])
    elif cfg.model.loss_fun == 'bce':
        return loss_bce(pred, true), torch.sigmoid(pred[:, 0])
    elif cfg.model.loss_fun == 'sce':
        return loss_bce(pred, true), torch.sigmoid(pred[:, 0])
    elif cfg.model.loss_fun == 'mse_rank':
        return loss_mse_rank(pred, true), torch.sigmoid(pred[:, 0])
    elif cfg.model.loss_fun == 'ccl':
        return loss_ccl(pred, true), torch.sigmoid(pred[:, 0])
    else:
        raise ValueError('Loss func {} not supported'.
                         format(cfg.model.loss_fun))

