import torch.nn as nn
import torch.nn.functional as F
import torch
import math



class ArchitectureEncoder(nn.Module):
    def __init__(self, arch_num, arch_embed_dim, hidden_dim, n_heads, n_layers):
        super(ArchitectureEncoder, self).__init__()
        self.arch_num = arch_num
        self.add_module('arch_emb', nn.Embedding(arch_num, arch_embed_dim))
        self.n_layers = n_layers
        self.layers = nn.ModuleList()
        self.layers.append(MultiHeadAttention(arch_embed_dim, arch_embed_dim, hidden_dim, n_heads))
        for layer in range(n_layers-1):
            self.layers.append(MultiHeadAttention(hidden_dim, hidden_dim, hidden_dim, n_heads))

    def forward(self, x, params=None):
        if params == None:
            out = self.arch_emb(x)
            for layer in self.layers:
                out = layer(out,out,out)
            out = out.transpose(1, 2)
            # out = F.max_pool1d(out, out.shape[-1]).squeeze(-1)
            out = F.avg_pool1d(out, out.shape[-1]).squeeze(-1)
        else:
            out = F.embedding(x, params['meta_learner.arch_encoder.arch_emb.weight'])
            for i in range(self.n_layers):
                layer = self.layers[i]
                out = layer(out,out,out, params, i)
            out = out.transpose(1, 2)
            # out = F.max_pool1d(out, out.shape[-1]).squeeze(-1)
            out = F.avg_pool1d(out, out.shape[-1]).squeeze(-1)
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self,  q_dim, k_dim, v_dim, n_heads):
        super(MultiHeadAttention, self).__init__()
        self.n_heads = n_heads
        self.v_dim = v_dim
        assert self.v_dim % self.n_heads == 0
        self.add_module('fc_q', nn.Linear(q_dim, v_dim))
        self.add_module('fc_k', nn.Linear(k_dim, v_dim))
        self.add_module('fc_v', nn.Linear(k_dim, v_dim))
        self.add_module('ln1', nn.LayerNorm(normalized_shape=v_dim))
        self.add_module('ln2', nn.LayerNorm(normalized_shape=v_dim))
        self.add_module('fc_o', nn.Linear(v_dim, v_dim))
        self.drop = nn.Dropout()

    def forward(self, Q, K, V, params=None, layer=None):
        if params == None:
            Q = self.fc_q(Q)
            K = self.fc_k(K)
            V = self.fc_v(V)
            dim_split = self.v_dim // self.n_heads
            Q_ = torch.cat(Q.split(dim_split, 2), 0)
            K_ = torch.cat(K.split(dim_split, 2), 0)
            V_ = torch.cat(V.split(dim_split, 2), 0)

            A = torch.softmax(Q_.bmm(K_.transpose(1,2))/math.sqrt(self.v_dim), 2)
            O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)
            O = self.ln1(O)
            O = O + F.relu(self.fc_o(O))
            O = self.ln2(O)

        elif layer is not None:
            Q = F.linear(Q, params[f'meta_learner.arch_encoder.layers.{layer}.fc_q.weight'],params[f'meta_learner.arch_encoder.layers.{layer}.fc_q.bias'])
            K = F.linear(K, params[f'meta_learner.arch_encoder.layers.{layer}.fc_k.weight'],params[f'meta_learner.arch_encoder.layers.{layer}.fc_k.bias'])
            V = F.linear(V, params[f'meta_learner.arch_encoder.layers.{layer}.fc_v.weight'],params[f'meta_learner.arch_encoder.layers.{layer}.fc_v.bias'])
            dim_split = self.v_dim // self.n_heads
            Q_ = torch.cat(Q.split(dim_split, 2), 0)
            K_ = torch.cat(K.split(dim_split, 2), 0)
            V_ = torch.cat(V.split(dim_split, 2), 0)
            A = torch.softmax(Q_.bmm(K_.transpose(1, 2)) / math.sqrt(self.v_dim), 2)
            O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)

            O = F.layer_norm(O, (self.v_dim,), params[f'meta_learner.arch_encoder.layers.{layer}.ln1.weight'], params[f'meta_learner.arch_encoder.layers.{layer}.ln1.bias'])
            O = O + F.relu(F.linear(O, params[f'meta_learner.arch_encoder.layers.{layer}.fc_o.weight'], params[f'meta_learner.arch_encoder.layers.{layer}.fc_o.bias']))
            O = F.layer_norm(O, (self.v_dim,), params[f'meta_learner.arch_encoder.layers.{layer}.ln2.weight'], params[f'meta_learner.arch_encoder.layers.{layer}.ln2.bias'])

        return O
