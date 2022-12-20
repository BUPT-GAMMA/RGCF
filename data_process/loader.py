import os
import random
import numpy as np
import torch
import pandas as pd
import math
from util.util import normalization, feature_map
from itertools import product
import pickle
class Data:
    def __init__(self,
                 mode,
                 data_path,
                 dataset_path,
                 meta_train_datasets,
                 meta_valid_datasets,
                 meta_test_datasets,
                 num_inner_tasks,
                 meta_train_ratio,
                 num_sample,
                 space_dims,
                 performance_name,
                 feature_type,
                 num_query=200,
                 ):
        self.mode = mode
        self.space_dims = space_dims
        self.performance_name = performance_name
        self.space_dim_dict = {}  # 3 if one space_dim contains 3 type
        self.space_arch_dict = {}
        self.space_arch_list = []
        self.space_arch_name_list = []
        self.data_path = data_path
        self.data = self.load_data()
        self.dataset_path = dataset_path
        self.feature_type = feature_type
        self.dataset_emb, self.dataset_input_dim = self.get_dataset_emb()
        self.meta_train_datasets = meta_train_datasets
        self.meta_valid_datasets = meta_valid_datasets
        self.meta_test_datasets = meta_test_datasets
        self.num_inner_tasks = num_inner_tasks
        self.meta_train_ratio = meta_train_ratio
        self.num_sample = num_sample
        self.num_query = math.ceil(num_query * meta_train_ratio)
        self.train_idx = {}
        self.valid_idx = {}
        self.performance = {}
        self.archs = {}
        # self.norm_performance = {}
        for dataset in meta_train_datasets + meta_valid_datasets +meta_test_datasets:
            # meta_sample = self.get_arch_and_performance(dataset)
            self.archs[dataset], self.performance[dataset] = self.get_arch_and_performance(dataset)
            # self.arch[dataset] = meta_sample[self.space_dims]
            nts = math.ceil(self.archs[dataset].shape[0] * self.meta_train_ratio * 0.8)
            train_idx = torch.arange(len(self.archs[dataset]))[:nts]
            valid_idx = torch.arange(len(self.archs[dataset]))[nts: math.ceil(self.archs[dataset].shape[0] * self.meta_train_ratio)-1]
            self.train_idx[dataset] = train_idx
            self.valid_idx[dataset] = valid_idx
            # self.norm_performance[dataset] = normalization(
            #     latency=self.performance[dataset],
            #     index=self.train_idx[dataset]
            # )
        # load index set of reference architectures

        print('==> load data ...')

    def get_arch_and_performance(self, dataset):
        task_data = self.data[self.data['dataset'] == dataset]
        task_data = task_data.sample(frac=1)
        meta_sample_x = torch.tensor(task_data[self.space_dims].values)
        meta_sample_y = torch.tensor(task_data[self.performance_name].values,dtype=torch.float32)

        return meta_sample_x, meta_sample_y

    def load_data(self):
        # include dataset name, design space dim, performance
        data = pd.read_csv(self.data_path)
        sum_dim = 0
        for dim in self.space_dims:
            dim_len = len(data[dim].unique())
            self.space_dim_dict[dim] = dim_len
            map_dict = dict(zip(data[dim].unique(), range(sum_dim, sum_dim + len(data[dim].unique()))))
            self.space_arch_dict[dim] = map_dict
            self.space_arch_list.append(map_dict.values())
            self.space_arch_name_list.append(map_dict.keys())
            sum_dim += dim_len
            data[dim] = data[dim].map(map_dict)
        self.space_dim_dict['sum_dim'] = sum_dim
        return data

    def generate_episode(self):
        # metabatch
        episode = []

        # meta-batch
        rand_datasets_idx = torch.randperm(
            len(self.meta_train_datasets))[:self.num_inner_tasks]
        for t in rand_datasets_idx:
            # sample datasets
            dataset = self.meta_train_datasets[t]
            # print(dataset)
            # print(self.dataset_emb)
            # dataset embedding
            dataset_emb = self.dataset_emb[dataset]
            # samples for finetuning & test (query)
            rand_idx = self.train_idx[dataset][torch.randperm(len(self.train_idx[dataset]))]
            finetune_idx = rand_idx[:self.num_sample]
            qry_idx = rand_idx[self.num_sample:self.num_sample + self.num_query]
            x_finetune = torch.stack([self.archs[dataset][_] for _ in finetune_idx])
            x_qry = torch.stack([self.archs[dataset][_] for _ in qry_idx])

            y_finetune = self.performance[dataset][finetune_idx].view(-1, 1)
            y_qry = self.performance[dataset][qry_idx].view(-1, 1)

            episode.append((dataset_emb, x_finetune, y_finetune, x_qry, y_qry, dataset))
        return episode

    def get_dataset_emb(self):
        dataset_feature = pd.read_csv(self.dataset_path,header=None)
        # print(dataset_feature[0])
        for column in range(1,dataset_feature.shape[1]):
            dataset_feature[column] = (dataset_feature[column] - dataset_feature[column].min())/(dataset_feature[column].max() - dataset_feature[column].min())
        dataset_feature.rename(columns={0:'name'},inplace=True)

        dataset_emb = {}
        for dataset in dataset_feature['name'].unique():
            temp_feature = dataset_feature[dataset_feature['name'] == dataset].drop(columns='name')
            temp_feature = temp_feature.values[:, feature_map[self.feature_type]]

            dataset_emb[dataset] = torch.tensor(temp_feature, dtype=torch.float32)
        dataset_input_dim = temp_feature.shape[1]
        return dataset_emb, dataset_input_dim

    def generate_test_tasks(self, split=None):
        if split == 'meta_train':
            dataset_list = self.meta_train_datasets
        elif split == 'meta_valid':
            dataset_list = self.meta_valid_datasets
        elif split == 'meta_test':
            dataset_list = self.meta_test_datasets
        else:
            NotImplementedError

        tasks = []
        if split == 'meta_train' or split == 'meta_valid':
            for dataset in dataset_list:
                tasks.append(self.get_task(dataset))
        elif split == 'meta_test':
            for dataset in dataset_list:
                tasks.append(self.get_test_task(dataset))
        else:
            NotImplementedError
        return tasks

    def get_test_task(self, dataset=None, num_sample=None):
        if num_sample == None:
            num_sample = self.num_sample
        rand_idx = torch.randperm(len(self.archs[dataset]))
        finetune_idx = rand_idx[:num_sample]
        qry_idx = rand_idx

        x_finetune = torch.stack([self.archs[dataset][_] for _ in finetune_idx])
        x_qry = torch.stack([self.archs[dataset][_] for _ in qry_idx])

        y_finetune = self.performance[dataset][finetune_idx].view(-1, 1)
        y_qry = self.performance[dataset][qry_idx].view(-1, 1)
        dataset_emb = self.dataset_emb[dataset]
        return dataset_emb, x_finetune, y_finetune, x_qry, y_qry, dataset

    def get_task(self, dataset=None, num_sample=None):
        if num_sample == None:
            num_sample = self.num_sample

        dataset_emb = self.dataset_emb[dataset]

        rand_idx = self.valid_idx[dataset][torch.randperm(len(self.valid_idx[dataset]))]
        finetune_idx = rand_idx[:num_sample]
        # qry_idx = rand_idx[num_sample:]
        qry_idx = rand_idx
        x_finetune = torch.stack([self.archs[dataset][_] for _ in finetune_idx])
        x_qry = torch.stack([self.archs[dataset][_] for _ in qry_idx])

        y_finetune = self.performance[dataset][finetune_idx].view(-1, 1)
        y_qry = self.performance[dataset][qry_idx].view(-1, 1)

        return dataset_emb, x_finetune, y_finetune, x_qry, y_qry, dataset

    def generate_arch_full(self, load_path, load_pretrain):
        if os.path.exists(load_path):
            if load_pretrain:
                x_qry = torch.load(os.path.join(load_path, 'arch_dim.pt'))
                with open("arch_name", "rb") as f:  # Unpickling
                    x_qry_arch_name = pickle.load(f)
            else:
                x_qry = torch.tensor(list(product(*self.space_arch_list)))
                torch.save(x_qry, os.path.join(load_path, 'arch_dim.pt'))
                x_qry_arch_name = list(product(*self.space_arch_name_list))
                with open("arch_name", "wb") as f:
                    pickle.dump(x_qry_arch_name, f)
        else:
            x_qry = torch.tensor(list(product(*self.space_arch_list)))
            os.mkdir(load_path)
            torch.save(x_qry, os.path.join(load_path, 'arch_dim.pt'))
            x_qry_arch_name = list(product(*self.space_arch_name_list))
            with open("arch_name", "wb") as f:
                pickle.dump(x_qry_arch_name, f)
        return x_qry, x_qry_arch_name

    def get_test_dataset_emb(self):
        emb = []
        for dataset in self.meta_test_datasets:
            emb.append((self.dataset_emb[dataset], dataset))
        return emb

    def _load_arch_str2idx(self):
        self.arch_str2idx = torch.load(os.path.join(self.data_path, 'str_arch2idx.pt'))

