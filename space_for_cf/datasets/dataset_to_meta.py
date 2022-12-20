import numpy as np
import pandas as pd
import h5py
import networkx as nx
import scipy.sparse as sp
from cluster import clustering, robins_alexander_clustering
from networkx import bipartite
bipartite.clustering
from collections import Counter
from itertools import combinations
from multiprocessing import Process, Pool
from time import time
import os
from networkx import NetworkXError
from time import time
import pickle
dtypes = {
        'user': np.int64, 'item': np.int64,
        'rating': np.float32, 'timestamp': float}

def map_node_label(user, item):
    """
    mark the node label starting from 0 and filter user/item label without interaction
    """
    unique_user = np.unique(user)
    unique_item = np.unique(item)
    user_label_map = {}
    for l, u in enumerate(unique_user):
        user_label_map[u] = l
    item_label_map = {}
    for l, i in enumerate(unique_item):
        item_label_map[i] = l

    mapped_user = [user_label_map[u] for u in user]
    mapped_item = [item_label_map[i] for i in item]
    mapped_user = np.asarray(mapped_user, dtype=user.dtype)
    mapped_item = np.asarray(mapped_item, dtype=item.dtype)

    mapped_item += mapped_user.max() + 1
    return mapped_user, mapped_item


def load_dataset_ml_100k(name, dataset_dir):
    dataset_dir = f'{dataset_dir}/{name}'
    if 'ml-100k' in name:

        dataset_raw = pd.read_csv(f'{dataset_dir}/u.data', sep='\t', header=None,
                                  names=['user', 'item', 'rating', 'timestamp'])

        train_data = pd.read_csv(f'{dataset_dir}/ua.base', sep='\t', header=None,
                                 names=['user', 'item', 'rating', 'timestamp'])
        test_data = pd.read_csv(f'{dataset_dir}/ua.test', sep='\t', header=None,
                                names=['user', 'item', 'rating', 'timestamp'])

        train_array = train_data.values.tolist()
        train_array = np.array(train_array)
        test_array = test_data.values.tolist()
        test_array = np.array(test_array)

        data_array = np.concatenate([train_array, test_array], axis=0)

        users = data_array[:, 0].astype(dtypes['user'])
        items = data_array[:, 1].astype(dtypes['item'])
        ratings = data_array[:, 2].astype(dtypes['rating'])

        # train_mean = np.mean(train_array[:, 2])
        # train_std = np.std(train_array[:, 2])
        # ratings = (ratings - train_mean) / train_std

        users, items = map_node_label(users, items)
        num_users = int(users.max() + 1)
        num_items = int(items.max() - users.max())
        # pairs = [(u, i, float(r)) for u, i, r in zip(users, items, ratings)]
        # graph = gen_graph(num_users, num_items, pairs)
        return users, items, num_users, num_items


def load_dataset_ml_1m(name, dataset_dir):
    dataset_dir = f'{dataset_dir}/{name}'

    dataset_raw = pd.read_csv(f'{dataset_dir}/ratings.dat', sep='::', header=None, engine='python',
                              names=['user', 'item', 'rating', 'timestamp'])

    users = np.array(dataset_raw['user'])
    items = np.array(dataset_raw['item'])

    users, items = map_node_label(users, items)
    num_users = int(users.max() + 1)
    num_items = int(items.max() - users.max())

    # pairs = [(u, i) for u, i in zip(users, items)]
    # graph = gen_graph(num_users, num_items, pairs)

    return users, items, num_users, num_items


def load_dataset_monti(name, dataset_dir):
    if name in ['douban', 'flixster', 'yahoo-music']:
        dataset_dir = f'{dataset_dir}/{name}/training_test_dataset.mat'

        M = load_matlab_file(dataset_dir, 'M')
        Otraining = load_matlab_file(dataset_dir, 'Otraining')
        Otest = load_matlab_file(dataset_dir, 'Otest')

        train_user, train_item = np.where(Otraining)[0].astype(dtypes['user']), np.where(Otraining)[1].astype(
            dtypes['item'])
        test_user, test_item = np.where(Otest)[0].astype(dtypes['user']), np.where(Otest)[1].astype(dtypes['item'])

        users = np.concatenate([train_user, test_user], axis=0)
        items = np.concatenate([train_item, test_item], axis=0)

        # Convert the rating scale of flixster to [1, 10]
        users, items = map_node_label(users, items)
        num_users = int(users.max() + 1)
        num_items = int(items.max() - users.max())
        # pairs = [(u, i) for u, i in zip(users, items)]
        # graph = gen_graph(num_users, num_items, pairs)

        return users, items, num_users, num_items


def load_dataset_amazon(name, dataset_dir):
    if 'amazon' in name or 'yelp2020' in name:
        dataset_file = f'{dataset_dir}/{name}/filtered_rating.csv'

        dataset_raw = pd.read_csv(dataset_file, sep='\t', names=['user', 'item', 'rating', 'timestamp'],engine='python')

        users = np.array(dataset_raw['user'])
        items = np.array(dataset_raw['item'])
        users, items = map_node_label(users, items)
        num_users = int(users.max() + 1)
        num_items = int(items.max() - users.max())

        # pairs = [(u, i) for u, i in zip(users, items)]
        #
        # graph = gen_graph(num_users, num_items, pairs)

        return users, items, num_users, num_items


def load_dataset_epinions(name, dataset_dir):
    if name in ['epinions']:
        dataset_file = f'{dataset_dir}/{name}/ratings_data.txt'
        dataset_raw = pd.read_csv(dataset_file, sep=' ', names=['user', 'item', 'rating'], engine='python')
        users = np.array(dataset_raw['user'])
        items = np.array(dataset_raw['item'])

        users, items = map_node_label(users, items)
        num_users = int(users.max() + 1)
        num_items = int(items.max() - users.max())

        # pairs = [(u, i) for u, i in zip(users, items)]
        # graph = gen_graph(num_users, num_items, pairs)

        return users, items, num_users, num_items

def load_dataset_meta_test(name, dataset_dir):
    dataset_file = f'{dataset_dir}/{name}/split/split_pointwise_ranking_True.pkl'
    with open(dataset_file, 'rb') as file:
        datasets = pickle.load(file)
        train_len = datasets[0].graphs[0].edge_label_index.shape[1] // 2
        val_len = datasets[1].graphs[0].edge_label_index.shape[1] // 2
        u_i_train = datasets[0].graphs[0].edge_label_index[:, :train_len]
        u_train, i_train = u_i_train[0], u_i_train[1]
        u_i_val = datasets[1].graphs[0].edge_label_index[:, :val_len]
        u_val, i_val = u_i_val[0], u_i_val[1]
        u_train = u_train.numpy().tolist()
        i_train = i_train.numpy().tolist()
        u_val = u_val.numpy().tolist()
        i_val = i_val.numpy().tolist()
        print(len(u_train+u_val), len(i_train+i_val))
        users, items = map_node_label( np.array(u_train+u_val), np.array(i_train+i_val))
        num_users = int(users.max() + 1)
        num_items = int(items.max() - users.max())
        return users, items, num_users, num_items

def load_matlab_file(path_file, name_field):
    """
    load '.mat' files
    inputs:
        path_file, string containing the file path
        name_field, string containig the field name (default='shape')
    warning:
        '.mat' files should be saved in the '-v7.3' format
    """
    raw_file = h5py.File(path_file, 'r')
    raw_data = raw_file[name_field]
    try:
        if 'ir' in raw_data.keys():
            data = np.asarray(raw_data['data'])
            ir = np.asarray(raw_data['ir'])
            jc = np.asarray(raw_data['jc'])
            output = sp.csc_matrix((data, ir, jc)).astype(np.float32)
    except AttributeError:
        # Transpose in case is a dense matrix because of the row- vs column- major ordering between python and matlab
        output = np.asarray(raw_data).astype(np.float32).T

    raw_file.close()

    return output

def get_rec_feature(users, items, num_users, num_items):
    space = np.log10((num_users*num_items)/1000)
    shape = np.log10(num_users/num_items)
    density = np.log10(len(users)/(num_users*num_items))
    user_degree = np.log10(len(users)/(num_users))
    item_degree = np.log10(len(users)/(num_items))
    users_gini = get_gini(users, num_users, len(users))
    items_gini = get_gini(items, num_items, len(items))
    print([space, shape, density, users_gini, items_gini])
    return [space, shape, density, user_degree, item_degree,  users_gini, items_gini]

def get_gini(obj, obj_num, length):
    obi_list = Counter(obj)
    weight = np.array(sorted(obi_list.values())) / length
    gini = (obj_num+1-np.arange(1, obj_num + 1))/(obj_num+1)
    gini = 1-np.sum(weight*gini)
    return gini

def gen_graph(num_users, num_items, pairs):

    G = nx.Graph()
    G.add_nodes_from(range(num_users), node_type='user')
    G.add_nodes_from(range(num_users, num_users + num_items), node_type='item')

    G.add_edges_from(pairs)

    return G

def node_redundancy(G, nodes=None):

    if nodes is None:
        nodes = G
    nodes_dict = {}
    for v in nodes:
        if len(G[v]) < 2:
            nodes_dict[v] = 0
        else:
            nodes_dict[v] = _node_redundancy(G, v)
    return nodes_dict



def _node_redundancy(G, v):
    n = len(G[v])
    # TODO On Python 3, we could just use `G[u].keys() & G[w].keys()` instead
    # of instantiating the entire sets.
    overlap = sum(
        1 for (u, w) in combinations(G[v], 2) if (set(G[u]) & set(G[w])) - {v}
    )
    return (2 * overlap) / (n * (n - 1))

def get_bipartite_feature(num_users, num_items, graph):
    feature_list = []
    dot_cluster, max_cluster, min_cluster = clustering(graph)
    feature_list.append(sum(dot_cluster.values()) / len(graph))
    feature_list.append(sum([dot_cluster[n] for n in range(num_users)]) / num_users)
    feature_list.append(sum([dot_cluster[n] for n in range(num_users, num_users + num_items)]) / num_items)
    feature_list.append(sum(min_cluster.values()) / len(graph))
    feature_list.append(sum([min_cluster[n] for n in range(num_users)]) / num_users)
    feature_list.append(sum([min_cluster[n] for n in range(num_users, num_users + num_items)]) / num_items)
    feature_list.append(sum(max_cluster.values()) / len(graph))
    feature_list.append(sum([max_cluster[n] for n in range(num_users)]) / num_users)
    feature_list.append(sum([max_cluster[n] for n in range(num_users, num_users + num_items)]) / num_items)
    coefficient = robins_alexander_clustering(graph)
    feature_list.append(coefficient)
    rc = node_redundancy(graph)
    rc_sum = sum(rc.values()) / len(graph)
    feature_list.append(rc_sum)
    rc_user = sum([rc[n] for n in range(num_users)]) / num_users
    feature_list.append(rc_user)
    rc_item = sum([rc[n] for n in range(num_users, num_users+num_items)]) / num_items
    feature_list.append(rc_item)

    return feature_list

def get_graph_feature(num_users, num_items, graph):
    cc = nx.number_connected_components(graph)
    pearson = nx.degree_pearson_correlation_coefficient(graph)
    print([cc, pearson])
    return [cc, pearson]

def get_meta_feature(name, dataset_dir='.'):
    if 'ml-100k' in name:
        users, items, num_users, num_items = load_dataset_ml_100k(name,dataset_dir)
    elif name in ['amazon-sports', 'epinions', 'beeradvocate']:
        users, items, num_users, num_items = load_dataset_meta_test(name,dataset_dir)
    elif 'ml-1m' in name:
        users, items, num_users, num_items = load_dataset_ml_1m(name,dataset_dir)
    elif name in ['douban', 'flixster', 'yahoo-music']:
        users, items, num_users, num_items = load_dataset_monti(name,dataset_dir)
    elif 'amazon' in name or 'yelp2020' in name:
        users, items, num_users, num_items = load_dataset_amazon(name,dataset_dir)
    pairs = [(u, i) for u, i in zip(users, items)]
    bi_graph = gen_graph(num_users, num_items, pairs)
    feature_list = get_rec_feature(users, items, num_users, num_items)
    # feature_list = get_bipartite_feature(num_users, num_items, bi_graph)
    feature_list += get_graph_feature(num_users, num_items, bi_graph)
    print(feature_list)
    return feature_list

def single_process(name):
    if name in ['flixster', 'ml-100k', 'ml-1m', 'douban', 'yahoo-music', 'yelp2020','amazon-cd', 'amazon-movies', 'amazon-beauty']:
        meta_feature = get_meta_feature(name, '.')
    elif name in ['epinions', 'amazon-sports', 'beeradvocate']:
        meta_feature = get_meta_feature(name, '/data/dataset/dataset/datasets/')
    if name == 'yahoo-music':
        name = 'yahoo'
    meta_feature = [name] + meta_feature
    with open('result.txt', 'a') as f:
        for i in meta_feature[0:-1]:
            f.write(str(i) + ',')
        f.write(str(meta_feature[-1]))
        f.write('\n')
    return meta_feature

if __name__ == "__main__":
    dataset_list = ['flixster', 'ml-100k', 'ml-1m', 'douban', 'yahoo-music', 'yelp2020','epinions','amazon-cd', 'amazon-movies', 'amazon-sports','amazon-beauty','beeradvocate']
    temp_list = ['douban', 'yahoo-music', 'yelp2020','epinions','amazon-cd', 'amazon-movies', 'amazon-sports','amazon-beauty']
    # temp_list = ['flixster']
    feature_list = []
    # single_process('epinions')
    for name in dataset_list:
        meta_feature = single_process(name)
        feature_list.append(meta_feature)
    meta_feature_df = pd.DataFrame(data=feature_list)
    meta_feature_df.to_csv('dataset_feature.csv',mode='a')
    # with Pool(11) as p:
    #      feature_list = p.map(single_process, dataset_list)
    #meta_feature_df = pd.DataFrame(data=feature_list)
    #meta_feature_df.to_csv('dataset_feature.csv', mode='a')
    # single_process('amazon-sports')
