from collections import defaultdict
import torch
import time
import logging
import gc
import numpy as np
import math
import numpy as np
from graphgym.config import cfg
from graphgym.loss import compute_loss
from graphgym.utils.epoch import is_eval_epoch, is_ckpt_epoch
from graphgym.checkpoint import load_ckpt, save_ckpt, clean_ckpt
from graphgym.sampler import create_negative_sample_bank, init_srns_mu
from tqdm import tqdm
import random


def evaluator_top_k(eval_matrix, edge_index, item_start_index, topk=10, remove_items_dict=None):
    mask = torch.zeros(eval_matrix.shape, dtype=torch.uint8, device=eval_matrix.device)

    for user, items in remove_items_dict.items():
        if len(items)>0:
            eval_matrix[user, torch.tensor(items) - item_start_index] = -np.inf
    top_k_index = torch.topk(eval_matrix, topk, dim=1).indices
    users_index = edge_index[0]
    items_index = edge_index[1] - item_start_index
    mask[users_index, items_index] = 1
    pos_items_sum = torch.sum(mask, dim=1)
    print(f'mask sum{torch.sum(pos_items_sum)}')
    hit_matrix = torch.gather(mask, 1, top_k_index)
    recall_of_each_user = torch.sum(hit_matrix, dim=1) / pos_items_sum
    recall_of_each_user[recall_of_each_user != recall_of_each_user] = 0.0
    recall_sum = torch.sum(recall_of_each_user)

    # top_k_index_all = top_k_index[users_index]
    #
    # recall = (top_k_index_all == items_index.reshape(-1, 1)).any(1).float().mean()
    return float(recall_sum.numpy())


def train_epoch(logger, loader, model, optimizer, scheduler, simple_info_dict=None, variance_all=None,
                score_one_epoch=None, cur_epoch=None):
    model.train()
    time_start = time.time()
    for batch in loader:
        edge_index = batch.edge_label_index[:, 0:batch.edge_label_index.shape[1] // 2].numpy()
        if cfg.train.negative_sample_method == 'dynamic' or cfg.train.negative_sample_method == 'similar' or cfg.train.negative_sample_method == 'neighbor':
            batch.to(torch.device(cfg.device))
            node_emb = model(batch, only_emb=True)
            node_emb = node_emb.cpu()
            sample_bank = create_negative_sample_bank(edge_index, simple_info_dict, node_emb=node_emb)
        elif cfg.train.negative_sample_method == 'srns':
            pass
        else:
            sample_bank = torch.tensor(create_negative_sample_bank(edge_index, simple_info_dict, cur_epoch=cur_epoch), dtype=int).to(
                torch.device(cfg.device))

    for batch in loader:
        sup_pos_edge_num = int((batch.edge_label_index.shape[1]) // 2)
        batch_size = cfg.train.batch_size
        batch_num = int(math.ceil(sup_pos_edge_num / batch_size))
        for i in tqdm(range(batch_num), desc='Evaluating'):
            optimizer.zero_grad()
            batch.to(torch.device(cfg.device))
            if i == batch_num - 1:
                edge_index = torch.arange(i * batch_size, sup_pos_edge_num)  # batch_idx * batch_size:sup_pos_edge_num
            else:
                edge_index = torch.arange(i * batch_size, (i + 1) * batch_size)
            if cfg.train.negative_sample_method == 'srns':
                edges = batch.edge_label_index[:, edge_index]
                Mu = simple_info_dict['mu']
                users = edges[0].detach().cpu().numpy()
                Mu_batch = Mu[users]
                Mu_temp = Mu[users] - simple_info_dict['node_type']['user']
                variance_used = variance_all[users]
                variance_used = torch.tensor(variance_used)
                Mu_temp = torch.tensor(Mu_temp)
                variance_used = variance_used.gather(1, Mu_temp)
                score_used = score_one_epoch[users]
                score_used = torch.tensor(score_used)
                score_used = score_used.gather(1, Mu_temp)
                candidate_score = score_used + min(1, cur_epoch / cfg.optim.max_epoch) * 1.0 * variance_used
                topk_index = torch.topk(candidate_score, cfg.train.negative_sample_k, dim=1).indices
                sample_bank = torch.tensor(Mu_batch).gather(1, topk_index)
                S2 = 100
                # update Mu
                for user in users:
                    Mu_user = Mu[user]
                    Mu_user_s2 = np.random.choice(simple_info_dict['node_type']['item'], size=S2) + \
                                 simple_info_dict['node_type']['user']
                    for s2_index in range(Mu_user_s2.shape[0]):
                        while Mu_user_s2[s2_index] in simple_info_dict['user2items'][user]:
                            random_item = random.randint(0, simple_info_dict['node_type']['item'] - 1) + \
                                          simple_info_dict['node_type']['user']
                            Mu_user_s2[s2_index] = random_item
                    candidate_item = np.concatenate([Mu_user, Mu_user_s2])
                    score_used = score_one_epoch[user][candidate_item - simple_info_dict['node_type']['user']]
                    score_used = score_used
                    temperature = 1.0
                    score_used = np.array(score_used) / temperature
                    score_used = np.exp(score_used) / np.sum(np.exp(score_used))
                    Mu_items_arg = np.random.choice(candidate_item, 100,
                                                    p=score_used, replace=False)
                    Mu[user] = Mu_items_arg
                simple_info_dict['mu'] = Mu
            pred, true = model(batch, edge_index=edge_index, sample_bank=sample_bank)

            loss, pred_score = compute_loss(pred, true)
            loss.backward()
            optimizer.step()

        logger.update_stats(true=true.detach().cpu(),
                            pred=pred_score.detach().cpu(),
                            loss=loss.item(),
                            lr=scheduler.get_last_lr()[0],
                            time_used=time.time() - time_start,
                            params=cfg.params)
        time_start = time.time()
    scheduler.step()
    if cfg.train.negative_sample_method == 'srns':
        return simple_info_dict

def train_epoch_rp(logger, loader, model, optimizer, scheduler):
    model.train()
    time_start = time.time()
    for batch in loader:
        optimizer.zero_grad()
        batch.to(torch.device(cfg.device))
        pred, true = model(batch)
        loss, pred_score = compute_loss(pred, true)
        loss.backward()
        optimizer.step()
        logger.update_stats(true=true.detach().cpu(),
                            pred=pred_score.detach().cpu(),
                            loss=loss.item(),
                            lr=scheduler.get_last_lr()[0],
                            time_used=time.time() - time_start,
                            params=cfg.params)
        time_start = time.time()
    scheduler.step()

def eval_epoch(logger, loader, model):
    model.eval()
    time_start = time.time()
    for batch in loader:
        batch.to(torch.device(cfg.device))
        print(cfg.model.eval_type)
        if cfg.model.eval_type == 'non-ranking':
            pred, true = model(batch)
            loss, pred_score = compute_loss(pred, true)
            logger.update_stats(true=true.detach().cpu(),
                                pred=pred_score.detach().cpu(),
                                loss=loss.item(),
                                lr=0,
                                time_used=time.time() - time_start,
                                params=cfg.params)


def eval_epoch_topK(loggers, loaders, model, item_start_index, cur_epoch, simple_info_dict):
    model.eval()
    for batch in loaders[0]:
        batch.to(torch.device(cfg.device))
        rating_matrix = model(batch)
    num_splits = len(loggers)
    for i in range(1, num_splits):
        for batch in loaders[i]:
            edge_index = batch.edge_label_index[:, 0:batch.edge_label_index.shape[1] // 2]
            if i == 1:  # eval valid
                metric_top_k = evaluator_top_k(rating_matrix, edge_index, item_start_index, cfg.topk,
                                               remove_items_dict=simple_info_dict['user2items'])
                metric_top_k = metric_top_k / simple_info_dict['valid_user_num']
            elif i == 2:  # eval test
                metric_top_k = evaluator_top_k(rating_matrix, edge_index, item_start_index, cfg.topk,
                                               remove_items_dict=simple_info_dict['user2items_in_test'])
                metric_top_k = metric_top_k / simple_info_dict['test_user_num']
            loggers[i].update_stats_ranking_eval(
                lr=0,
                params=cfg.params,
                eval_metric=metric_top_k)
        loggers[i].write_epoch_ranking(cur_epoch)
    if cfg.train.negative_sample_method == 'srns':
        return rating_matrix


def train(loggers, loaders, model, optimizer, scheduler, exp_num=0, simple_info_dict=None):
    start_epoch = 0
    if cfg.train.auto_resume:
        start_epoch = load_ckpt(model, optimizer, scheduler)
    if start_epoch == cfg.optim.max_epoch:
        logging.info('Checkpoint found, Task already done')
    else:
        logging.info('Start from epoch {}'.format(start_epoch))
    if cfg.train.negative_sample_method == 'srns':
        score_all = []
        model.eval()
        for batch in loaders[0]:
            batch.to(torch.device(cfg.device))
            rating_matrix = model(batch)
            rating_matrix = torch.sigmoid(rating_matrix)
        score_all.append(rating_matrix.numpy())
    num_splits = len(loggers)
    
    for cur_epoch in range(start_epoch, cfg.optim.max_epoch):
        t1 = time.time()
        if cfg.train.negative_sample_method == 'srns':
            num_items = simple_info_dict['node_type']['item']
            num_user = simple_info_dict['node_type']['user']
            if len(score_all) < 5:
                variance_all = np.zeros((num_user, num_items))
            else:
                score_np = np.array(score_all)
                mean_all = np.mean(score_np, axis=0)
                variance_all = np.zeros(mean_all.shape)
                for i in range(5):
                    variance_all += (score_np[i, :, :] - mean_all) ** 2
                variance_all = np.sqrt(variance_all / 5)
            simple_info_dict = train_epoch(loggers[0], loaders[0], model, optimizer, scheduler, simple_info_dict,
                                           variance_all=variance_all, score_one_epoch=score_all[-1],
                                           cur_epoch=cur_epoch)
        elif cfg.model.eval_type == 'ranking' :
            train_epoch(loggers[0], loaders[0], model, optimizer, scheduler, simple_info_dict, cur_epoch=cur_epoch)
        elif cfg.model.eval_type == 'non-ranking':
            train_epoch_rp(loggers[0], loaders[0], model, optimizer, scheduler)
        logging.info(f'[Exp{exp_num}]')
        if cfg.model.eval_type == 'non-ranking':
            loggers[0].write_epoch(cur_epoch)
        else:
            loggers[0].write_epoch_ranking(cur_epoch)

        if is_eval_epoch(cur_epoch) and cfg.model.eval_type == 'non-ranking':
            for i in range(1, num_splits):
                eval_epoch(loggers[i], loaders[i], model, )
                loggers[i].write_epoch(cur_epoch)
        elif is_eval_epoch(cur_epoch) and cfg.model.eval_type == 'ranking':
            if cfg.train.negative_sample_method == 'srns':
                rate_matrix = eval_epoch_topK(loggers, loaders, model, simple_info_dict['node_type']['user'], cur_epoch,
                                              simple_info_dict)
                score_all.append(torch.sigmoid(rate_matrix).numpy())
                if len(score_all) > 5:
                    score_all.pop(0)
            else:
                eval_epoch_topK(loggers, loaders, model, simple_info_dict['node_type']['user'], cur_epoch,
                                simple_info_dict)
        print(f'test time one epoch{time.time()-t1}')
        if is_ckpt_epoch(cur_epoch):
            save_ckpt(model, optimizer, scheduler, cur_epoch)

    for logger in loggers:
        logger.close()
    if cfg.train.ckpt_clean:
        clean_ckpt()

    logging.info('Task done, results saved in {}'.format(cfg.out_dir))

    # Collect unused memory
    del model
    for loader in loaders:
        for batch in loader:
            del batch
    #gc.collect()
    #torch.cuda.empty_cache()




