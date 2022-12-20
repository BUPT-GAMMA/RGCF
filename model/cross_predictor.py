import torch.nn as nn
import torch.nn.functional as F
import torch
import math
from model.arch_encoder import ArchitectureEncoder
from model.pnn import PNN
from model.dataset_encoder import DatasetEncoder
from util.util import mix_up
import random

class Predictor(nn.Module):
    def __init__(self, arch_num, arch_embed_dim, hidden_dim, n_heads, n_layers, cross_hid_dim, dataset_on=True, dataset_input_dim=None, dataset_hid_dim=None, dataset_out_dim=None):
        super(Predictor, self).__init__()
        # self.arch_encoder = ArchitectureEncoder(arch_num, arch_embed_dim, hidden_dim, n_heads, n_layers)
        self.arch_encoder = PNN(arch_num, arch_embed_dim, hidden_dim)
        self.dataset_on = dataset_on
        if dataset_on:
            self.dataset_encoder = DatasetEncoder(dataset_input_dim, dataset_hid_dim, dataset_out_dim)
            # assert dataset_hid_dim == arch_embed_dim
            self.add_module('fc_cross_1', nn.Linear(arch_embed_dim+dataset_out_dim, cross_hid_dim))
        else:
            self.add_module('fc_cross_1', nn.Linear(arch_embed_dim, cross_hid_dim))
        self.add_module('fc_cross_2', nn.Linear(cross_hid_dim, cross_hid_dim))
        self.add_module('fc_cross_3', nn.Linear(cross_hid_dim, 1))
        self.reconstruct_loss = nn.MSELoss()

    def forward(self, arch_input, dataset_feature=None, params=None, inner_loop=False):
        re_loss = None
        if params is None:
            out = self.arch_encoder(arch_input)
            if self.dataset_on:
                dataset_emb, re_loss = self.dataset_encoder(dataset_feature)
                out = torch.cat([out,dataset_emb],dim=-1)
            out = F.relu(self.fc_cross_1(out))
            out = F.relu(self.fc_cross_2(out))
            out = self.fc_cross_3(out)
        else:
            out = self.arch_encoder(arch_input, params)
            if self.dataset_on:
                dataset_emb, re_loss= self.dataset_encoder(dataset_feature, inner_loop=inner_loop)
                out = torch.cat([out, dataset_emb], dim=-1)
            out = F.linear(out, params['meta_learner.fc_cross_1.weight'], params['meta_learner.fc_cross_1.bias'])
            out = F.relu(out)
            out = F.linear(out, params['meta_learner.fc_cross_2.weight'], params['meta_learner.fc_cross_2.bias'])
            out = F.relu(out)
            out = F.linear(out, params['meta_learner.fc_cross_3.weight'], params['meta_learner.fc_cross_3.bias'])
        return out, re_loss

    def mix_forward(self, X_i, X_j, dataset_feature_i, dataset_feature_j, lam, inter_layer, params=None, inner_loop=False):
        if params is None:
            out_i = self.arch_encoder(X_i)
            out_j = self.arch_encoder(X_j)
            dataset_emb_i, re_loss_i = self.dataset_encoder(dataset_feature_i)
            dataset_emb_j, re_loss_j = self.dataset_encoder(dataset_feature_j)

            out_i = torch.cat([out_i, dataset_emb_i],dim=-1)
            out_j = torch.cat([out_j, dataset_emb_j], dim=-1)
            if inter_layer == 0:
                out = mix_up(out_i, out_j, lam)
                out = F.relu(self.fc_cross_1(out))
                out = F.relu(self.fc_cross_2(out))
            elif inter_layer == 1:
                out_i = F.relu(self.fc_cross_1(out_i))
                out_j = F.relu(self.fc_cross_1(out_i))
                out = mix_up(out_i, out_j, lam)
                out = F.relu(self.fc_cross_2(out))
            elif inter_layer == 2:
                out_i = F.relu(self.fc_cross_1(out_i))
                out_j = F.relu(self.fc_cross_1(out_j))
                out_i = F.relu(self.fc_cross_2(out_i))
                out_j = F.relu(self.fc_cross_2(out_j))
                out = mix_up(out_i, out_j, lam)
            out = self.fc_cross_3(out)
        else:
            out_i = self.arch_encoder(X_i, params)
            out_j = self.arch_encoder(X_j, params)
            dataset_emb_i, re_loss_i = self.dataset_encoder(dataset_feature_i, inner_loop=inner_loop)
            dataset_emb_j, re_loss_j = self.dataset_encoder(dataset_feature_j, inner_loop=inner_loop)

            out_i = torch.cat([out_i, dataset_emb_i], dim=-1)
            out_j = torch.cat([out_j, dataset_emb_j], dim=-1)

            if inter_layer == 0:
                out = mix_up(out_i, out_j, lam)
                out = F.linear(out, params['meta_learner.fc_cross_1.weight'], params['meta_learner.fc_cross_1.bias'])
                out = F.relu(out)
                out = F.linear(out, params['meta_learner.fc_cross_2.weight'], params['meta_learner.fc_cross_2.bias'])
                out = F.relu(out)
            elif inter_layer == 1:
                out_i = F.linear(out_i, params['meta_learner.fc_cross_1.weight'], params['meta_learner.fc_cross_1.bias'])
                out_i = F.relu(out_i)
                out_j = F.linear(out_j, params['meta_learner.fc_cross_1.weight'], params['meta_learner.fc_cross_1.bias'])
                out_j = F.relu(out_j)
                out = mix_up(out_i, out_j, lam)
                out = F.linear(out, params['meta_learner.fc_cross_2.weight'], params['meta_learner.fc_cross_2.bias'])
                out = F.relu(out)
            elif inter_layer == 2:
                out_i = F.linear(out_i, params['meta_learner.fc_cross_1.weight'],
                                 params['meta_learner.fc_cross_1.bias'])
                out_i = F.relu(out_i)
                out_j = F.linear(out_j, params['meta_learner.fc_cross_1.weight'],
                                 params['meta_learner.fc_cross_1.bias'])
                out_j = F.relu(out_j)
                out_i = F.linear(out_i, params['meta_learner.fc_cross_2.weight'], params['meta_learner.fc_cross_2.bias'])
                out_i = F.relu(out_i)
                out_j = F.linear(out_j, params['meta_learner.fc_cross_2.weight'],
                                 params['meta_learner.fc_cross_2.bias'])
                out_j = F.relu(out_j)
                out = mix_up(out_i, out_j, lam)

            out = F.linear(out, params['meta_learner.fc_cross_3.weight'], params['meta_learner.fc_cross_3.bias'])
        if re_loss_i and re_loss_j:
            re_loss = mix_up(re_loss_i, re_loss_j, lam)
        else:
            re_loss = None
        return out, re_loss

    def mix_no_feature_forward(self, X_i, X_j, lam, inter_layer, params=None):
        if params is None:
            out_i = self.arch_encoder(X_i)
            out_j = self.arch_encoder(X_j)

            if inter_layer == 0:
                out = mix_up(out_i, out_j, lam)
                out = F.relu(self.fc_cross_1(out))
                out = F.relu(self.fc_cross_2(out))
            elif inter_layer == 1:
                out_i = F.relu(self.fc_cross_1(out_i))
                out_j = F.relu(self.fc_cross_1(out_i))
                out = mix_up(out_i, out_j, lam)
                out = F.relu(self.fc_cross_2(out))
            elif inter_layer == 2:
                out_i = F.relu(self.fc_cross_1(out_i))
                out_j = F.relu(self.fc_cross_1(out_j))
                out_i = F.relu(self.fc_cross_2(out_i))
                out_j = F.relu(self.fc_cross_2(out_j))
                out = mix_up(out_i, out_j, lam)
            out = self.fc_cross_3(out)
        else:
            out_i = self.arch_encoder(X_i, params)
            out_j = self.arch_encoder(X_j, params)
            if inter_layer == 0:
                out = mix_up(out_i, out_j, lam)
                out = F.linear(out, params['meta_learner.fc_cross_1.weight'], params['meta_learner.fc_cross_1.bias'])
                out = F.relu(out)
                out = F.linear(out, params['meta_learner.fc_cross_2.weight'], params['meta_learner.fc_cross_2.bias'])
                out = F.relu(out)
            elif inter_layer == 1:
                out_i = F.linear(out_i, params['meta_learner.fc_cross_1.weight'], params['meta_learner.fc_cross_1.bias'])
                out_i = F.relu(out_i)
                out_j = F.linear(out_j, params['meta_learner.fc_cross_1.weight'], params['meta_learner.fc_cross_1.bias'])
                out_j = F.relu(out_j)
                out = mix_up(out_i, out_j, lam)
                out = F.linear(out, params['meta_learner.fc_cross_2.weight'], params['meta_learner.fc_cross_2.bias'])
                out = F.relu(out)
            elif inter_layer == 2:
                out_i = F.linear(out_i, params['meta_learner.fc_cross_1.weight'],
                                 params['meta_learner.fc_cross_1.bias'])
                out_i = F.relu(out_i)
                out_j = F.linear(out_j, params['meta_learner.fc_cross_1.weight'],
                                 params['meta_learner.fc_cross_1.bias'])
                out_j = F.relu(out_j)
                out_i = F.linear(out_i, params['meta_learner.fc_cross_2.weight'], params['meta_learner.fc_cross_2.bias'])
                out_i = F.relu(out_i)
                out_j = F.linear(out_j, params['meta_learner.fc_cross_2.weight'],
                                 params['meta_learner.fc_cross_2.bias'])
                out_j = F.relu(out_j)
                out = mix_up(out_i, out_j, lam)

            out = F.linear(out, params['meta_learner.fc_cross_3.weight'], params['meta_learner.fc_cross_3.bias'])
        return out


class DotPredictor(nn.Module):
    def __init__(self, arch_num, arch_embed_dim, hidden_dim, n_heads, n_layers, cross_hid_dim, dataset_on=True,
                 dataset_input_dim=None, dataset_hid_dim=None, dataset_out_dim=None):
        super(DotPredictor, self).__init__()
        self.arch_encoder = ArchitectureEncoder(arch_num, arch_embed_dim, hidden_dim, n_heads, n_layers)
        self.dataset_encoder = DatasetEncoder(dataset_input_dim, dataset_hid_dim, dataset_out_dim)


    def forward(self, arch_input, dataset_feature=None, params=None, inner_loop=False):
        if params is None:
            out_arch = self.arch_encoder(arch_input)
            out_dataset = self.dataset_encoder(dataset_feature)
        else:
            out_arch = self.arch_encoder(arch_input, params)
            out_dataset = self.dataset_encoder(dataset_feature, inner_loop=inner_loop)
        out = torch.mul(out_arch, out_dataset)
        out = torch.sum(out, dim=1, keepdim=True)
        return out