
import numpy as np

from collections import defaultdict,Counter
import os
import torch
import pickle
from graphgym.config import cfg
import pandas as pd
from time import time
import random
def get_user2items_dict(edge_index):
    user2items_dict = defaultdict(list)
    count = 0
    for user_index, item_index in zip(edge_index[0].numpy(), edge_index[1].numpy()):
        user2items_dict[user_index].append(item_index)
        count += 1
    print(f'count:{count}')
    return user2items_dict


def uniform_negative_sample(users_index, user2items_dict, user_nums, item_nums, item_start_index, cur_epoch=None):
    sampled_items = np.zeros(shape=(users_index.shape[0], cfg.train.negative_sample_k),dtype=np.int)
    dataset_dir = cfg.dataset.dir
    name = cfg.dataset.name
    save_path = f'{dataset_dir}/{name}/seed{cfg.seed}_{cur_epoch}_{cfg.train.negative_sample_method}_{cfg.train.negative_sample_k}'
    save_fn = f'{save_path}/negative_node.pkl'

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    if os.path.isfile(save_fn):
        with open(save_fn, 'rb') as file:
            sampled_items = pickle.load(file)
            print('load negative dict from pre')
    else:
        for user_index, pos_items_index in user2items_dict.items():
            pos_items = user2items_dict[user_index]
            probs = np.ones(item_nums)
            probs[np.array(pos_items) - item_start_index] = 0
            probs = probs / np.sum(probs)  # renomalize to sum 1
            sample_index = np.random.choice(item_nums, size=cfg.train.negative_sample_k * len(pos_items), replace=True,
                                            p=probs) + item_start_index
            sampled_items[users_index == user_index] = sample_index.reshape(-1, cfg.train.negative_sample_k)
        with open(save_fn, 'wb') as file:
            pickle.dump(sampled_items, file)
        print(f'create negative seed{cfg.seed}_{cur_epoch}_{cfg.train.negative_sample_method}_{cfg.train.negative_sample_k} file first time')

    return sampled_items


def neighbor_negative_sample(edges_index, user2items_dict, user_nums, item_nums, item_start_index, neighbor_index,node_emb, candidate_size = None, alpha=0.5):
    if candidate_size:
        pass
    else:
        candidate_size = 500
    users_index = edges_index[0]
    candidate_items = np.zeros(shape=(edges_index.shape[1], candidate_size),dtype=np.int)

    for user_index, pos_items_index in user2items_dict.items():
        pos_items = user2items_dict[user_index]
        intermediate_items = neighbor_index[user_index]
        if(len(intermediate_items)!=0): 
            sample_index = np.random.choice(intermediate_items, size=candidate_size * len(pos_items), replace=True)
        else:
            probs = np.ones(item_nums)
            probs[np.array(pos_items) - item_start_index] = 0
            probs = probs / np.sum(probs)  # renomalize to sum 1
            sample_index = np.random.choice(item_nums, size=candidate_size * len(pos_items), replace=True,
                                            p=probs) + item_start_index
        candidate_items[users_index == user_index] = sample_index.reshape(-1, candidate_size)
    user_emb = node_emb[0:user_nums]
    item_emb = node_emb[user_nums:]
    user_to_item_similar = torch.mm(user_emb,item_emb.T).detach()
    item_to_item_similar = torch.mm(item_emb, item_emb.T).detach()

    sampled_items = np.zeros(shape=(edges_index.shape[1], cfg.train.negative_sample_k), dtype=np.int)
    for i in range(edges_index.shape[1]):
        user = edges_index[0][i]
        item = edges_index[1][i]
        user_to_item = user_to_item_similar[user, candidate_items[i]-item_start_index]
        item_to_item = item_to_item_similar[item-item_start_index, candidate_items[i]-item_start_index]
        score = user_to_item*alpha+item_to_item*(1-alpha)
        score = torch.sigmoid(score)
        score = score/score.sum()
        sampled_items[i] = np.random.choice(candidate_items[i], size=cfg.train.negative_sample_k,
                                              p=score.cpu().numpy(),
                                              replace=True)
    return sampled_items


def popular_negative_sample(users_index, user2items_dict, user_nums, item_nums, item_start_index, popular_dict,cur_epoch=None):
    sampled_items = np.zeros(shape=(users_index.shape[0], cfg.train.negative_sample_k),dtype=np.int)
    dataset_dir = cfg.dataset.dir
    name = cfg.dataset.name
    save_path = f'{dataset_dir}/{name}/seed{cfg.seed}_{cur_epoch}_{cfg.train.negative_sample_method}_{cfg.train.negative_sample_k}'
    save_fn = f'{save_path}/negative_node.pkl'

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    if os.path.isfile(save_fn):
        with open(save_fn, 'rb') as file:
            sampled_items = pickle.load(file)
            print('load negative dict from pre')
    else:
        for user_index, pos_items_index in user2items_dict.items():
            pos_items = user2items_dict[user_index]
            probs = np.zeros(item_nums)
            probs[np.array(list(popular_dict.keys()))-item_start_index] = np.array(list(popular_dict.values()))
            probs[np.array(pos_items)-item_start_index] = 0
            probs = probs / np.sum(probs)  # renomalize to sum 1
            sample_index = np.random.choice(item_nums, size=cfg.train.negative_sample_k * len(pos_items), replace=True,
                                            p=probs) + item_start_index
            sampled_items[users_index == user_index] = sample_index.reshape(-1, cfg.train.negative_sample_k)
        with open(save_fn, 'wb') as file:
            pickle.dump(sampled_items, file)
        print(
            f'create negative seed{cfg.seed}_{cur_epoch}_{cfg.train.negative_sample_method}_{cfg.train.negative_sample_k} file first time')
    return sampled_items

def dynamic_negative_sample(users_index, user2items_dict, user_nums, item_nums, item_start_index, node_emb, beta=0.5):
    sampled_items = np.zeros(shape=(users_index.shape[0], cfg.train.negative_sample_k*2), dtype=np.int)
    for user_index, pos_items_index in user2items_dict.items():
        pos_items = user2items_dict[user_index]
        probs = np.ones(item_nums)
        probs[np.array(pos_items) - item_start_index] = 0
        probs = probs / np.sum(probs)  # renomalize to sum 1
        # sampled_items[user_index] = np.random.choice(item_nums, size=cfg.train.negative_sample_k*2, replace=True,
        #                                              p=probs) + item_start_index
        sample_index = np.random.choice(item_nums, size=cfg.train.negative_sample_k*2*len(pos_items), replace=True,
                                        p=probs) + item_start_index
        sampled_items[users_index == user_index] = sample_index.reshape(-1, cfg.train.negative_sample_k*2)
    rejection_index = np.random.choice(2, size=cfg.train.negative_sample_k*users_index.shape[0], replace=True,
                                                     p=np.array([1.0/(1.0+beta), beta/(1.0+beta)]))
    rejection_index = torch.tensor(rejection_index)
    sampled_items = torch.tensor(sampled_items)
    user_emb = node_emb[users_index]
    user_emb = user_emb.repeat_interleave(cfg.train.negative_sample_k*2, dim=0)
    items_emb = node_emb[sampled_items]
    items_emb = items_emb.view(-1, items_emb.shape[-1])
    score = (user_emb*items_emb).sum(1)
    score = score.view(-1, 2)
    select_index = torch.argmax(score, 1)
    select_index[rejection_index==1] = 1-select_index[rejection_index==1]
    sampled_items = sampled_items.view(-1, 2)
    sampled_items = sampled_items.gather(1, select_index.reshape(-1,1))
    sampled_items = sampled_items.view(-1, cfg.train.negative_sample_k)
    return sampled_items

def similar_negative_sample(edges_index,user2items_dict ,user_nums, item_nums, item_start_index, popular_dict, node_emb, candidate_size = None , beta=0.5):
    # candidate_items = np.zeros(shape=(user_nums, cfg.train.negative_sample_k), dtype=np.int)
    if candidate_size:
        pass
    else:
        candidate_size = 500
    probs = np.zeros(item_nums)
    probs[np.array(list(popular_dict.keys())) - item_start_index] = np.array(list(popular_dict.values()))
    probs = probs / np.sum(probs)  # renomalize to sum 1
    probs = probs**beta
    probs = probs / np.sum(probs)
    sample_index = np.random.choice(item_nums, size=candidate_size, replace=False,
                                    p=probs) + item_start_index

    pos_emb = node_emb[user_nums:]
    neg_emb = node_emb[sample_index]
    item_spreadout_distances = torch.mm(pos_emb,neg_emb.T)
    weights_matrix = np.zeros(shape=(item_nums, candidate_size))
    for i in range(item_spreadout_distances.shape[0]):
        weights_matrix[i] = calculate_weights(item_spreadout_distances[i].detach().cpu().numpy(),node_emb.shape[1])


    sampled_items = np.zeros(shape=(edges_index.shape[1],cfg.train.negative_sample_k), dtype=np.int)
    for i in range(edges_index.shape[1]):
        user = edges_index[0, i]
        item = edges_index[1, i]
        weights = weights_matrix[item-item_start_index]

        weights_sum = np.sum(weights)
        if weights_sum > 0:
            weights = weights / weights_sum

        if weights_sum > 0:
            sampled_items[i] = np.random.choice(sample_index,
                                              size=cfg.train.negative_sample_k,
                                              p=weights,
                                              replace=True)
        else:
            sampled_items[i] = np.random.choice(sample_index,
                                                   size=cfg.train.negative_sample_k,
                                                   replace=True)
        for j, neg in enumerate(sampled_items[i]):
            current_try = 0

            while neg in user2items_dict[user] and current_try < 3:

                if weights_sum > 0:
                    sampled_items[i, j] = neg = np.random.choice(sample_index,
                                                               p=weights)
                else:
                    sampled_items[i, j] = neg = np.random.choice(sample_index)
                current_try += 1

    return sampled_items

def calculate_weights(spreadout_distance, emb_dim):
    BETA_VOLUMN_CONST = 0.2228655673209082014292
    mask = spreadout_distance > 0
    log_weights = (1.0 - (float(emb_dim) - 1) / 2) * np.log(
        1.0 - np.square(spreadout_distance) + 1e-8) + np.log(
        BETA_VOLUMN_CONST)
    weights = np.exp(log_weights)
    weights[np.isnan(weights)] = 0.
    weights[~mask] = 0.
    weights_sum = np.sum(weights)

    if weights_sum > 0:
        weights = weights / weights_sum
    return weights

def init_srns_mu(user2items_dict, user_nums, item_nums, item_start_index, S1=100):
    Mu = []
    for i in range(user_nums):
        Mu.append([])
        while len(Mu[i])!= S1:
            random_item=random.randint(0, item_nums-1)+item_start_index
            while random_item in user2items_dict[i]:
                random_item=random.randint(0, item_nums-1)+item_start_index
            Mu[i].append(random_item)
    return Mu

# def init_score_all():
#
#
# def srns_negative_sample():



def create_negative_sample_bank(edge_index, info_dict, node_emb = None, cur_epoch=None):
    print('You will get negative sample here')
    t1 = time()
    if cfg.train.negative_sample_method == 'uniform':
        user2items_dict = info_dict['user2items']
        node_type_dict = info_dict['node_type']
        sample_bank = uniform_negative_sample(edge_index[0], user2items_dict, node_type_dict['user'], node_type_dict['item'], node_type_dict['user'],cur_epoch=cur_epoch)

    elif cfg.train.negative_sample_method == 'popular':
        user2items_dict = info_dict['user2items']
        node_type_dict = info_dict['node_type']
        sample_bank = popular_negative_sample(edge_index[0], user2items_dict, node_type_dict['user'], node_type_dict['item'],                                           node_type_dict['user'],info_dict['items_popular']
                                              ,cur_epoch=cur_epoch)

    elif cfg.train.negative_sample_method == 'dynamic':
        user2items_dict = info_dict['user2items']
        node_type_dict = info_dict['node_type']
        sample_bank = dynamic_negative_sample(edge_index[0], user2items_dict, node_type_dict['user'], node_type_dict['item'], node_type_dict['user'], node_emb)

    elif cfg.train.negative_sample_method == 'similar':
        user2items_dict = info_dict['user2items']
        node_type_dict = info_dict['node_type']
        sample_bank = similar_negative_sample(edge_index, user2items_dict, node_type_dict['user'], node_type_dict['item'], node_type_dict['user'], info_dict['items_popular'], node_emb)
        print(f'one epoch sample time{time() - t1}')
        return torch.tensor(sample_bank, dtype=int).to(torch.device(cfg.device))
    elif cfg.train.negative_sample_method == 'neighbor':
        user2items_dict = info_dict['user2items']
        node_type_dict = info_dict['node_type']
        neighbor_dict = info_dict['neighbor']
        sample_bank = neighbor_negative_sample(edge_index,user2items_dict, node_type_dict['user'], node_type_dict['item'], node_type_dict['user'], neighbor_dict,node_emb )
        print(f'one epoch sample time{time() - t1}')
        return torch.tensor(sample_bank, dtype=int).to(torch.device(cfg.device))
    elif cfg.train.negative_sample_method == 'srns':
        return []
    print(f'one epoch sample time{time()-t1}')
    return sample_bank

def get_info_dict(datasets): # two dict: user item infer, node type count
    g = datasets[0].graphs[0]
    edge_index = g.edge_label_index[:, 0: (g.edge_label_index.shape[1]) // 2]
    user2items_dict = get_user2items_dict(edge_index)
    node_type_dict = Counter(g.node_type)
    popular_dict = Counter(edge_index[1].numpy())
    info_dict = {}
    info_dict['user2items'] = user2items_dict
    info_dict['node_type'] = node_type_dict
    info_dict['items_popular'] = popular_dict
    if cfg.train.negative_sample_method == 'neighbor': # construct a dict u2i2u2i
        t1 = time()
        dataset_dir = cfg.dataset.dir
        name = cfg.dataset.name
        save_path = f'{dataset_dir}/{name}/neighbor'
        save_fn = f'{save_path}/neighbor.pkl'

        if not os.path.exists(save_path):
            os.mkdir(save_path)

        if os.path.isfile(save_fn):
            with open(save_fn, 'rb') as file:
                neighbor_dict = pickle.load(file)
                print('load neighbor dict from pre')
        else:
            neighbor_dict = get_neighbor_dict(edge_index,user2items_dict)
            with open(save_fn, 'wb') as file:
                pickle.dump(neighbor_dict, file)
            print('create neighbor dict first time')
        info_dict['neighbor'] = neighbor_dict
        print(f'time for create neighbor set{time()-t1}')
    elif cfg.train.negative_sample_method == 'srns':
        info_dict['mu'] = np.array(init_srns_mu(user2items_dict, node_type_dict['user'], node_type_dict['item'], node_type_dict['user']))

    g = datasets[1].graphs[0]
    info_dict['valid_user_num'] = torch.unique(g.edge_label_index[:, 0: (g.edge_label_index.shape[1]) // 2][0]).shape[0]
    edge_index = torch.cat([edge_index, g.edge_label_index[:, 0: (g.edge_label_index.shape[1]) // 2]],dim=1)
    user2items_dict_train_and_valid = get_user2items_dict(edge_index)
    info_dict['user2items_in_test'] = user2items_dict_train_and_valid
    g = datasets[2].graphs[0]
    info_dict['test_user_num'] = torch.unique(g.edge_label_index[:, 0: (g.edge_label_index.shape[1]) // 2][0]).shape[0]

    return info_dict

def get_neighbor_dict(edge_index, user2items_dict):
    neighbor_dict = dict()
    edge_df = pd.DataFrame({'user': edge_index[0].numpy(), 'item': edge_index[1].numpy()})
    items_to_items = pd.merge(edge_df, edge_df, on=['user'])
    items_to_items = items_to_items.drop(columns=['user'])
    items_to_items.columns = ['item_src', 'item']
    items_to_items = items_to_items.drop_duplicates()
    items_to_items = items_to_items[items_to_items['item_src'] != items_to_items['item']]
    items_to_items_agg = items_to_items.groupby('item_src').agg(list)
    items_neighbor = dict(zip(list(items_to_items_agg.index), list(items_to_items_agg['item'])))
    for user_index, pos_items_index in user2items_dict.items():
        temp_set = set()
        for pos_item in pos_items_index:
            if pos_item in items_neighbor.keys():
                temp_set.update(items_neighbor[pos_item])

        temp_list = list(temp_set)
        neighbor_dict[user_index] = [i for i in temp_list if i not in pos_items_index]
    return neighbor_dict