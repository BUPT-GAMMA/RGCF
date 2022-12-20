import torch
import torch.nn as nn
from torch.nn.init import xavier_normal_, constant_
import torch.nn as nn
import torch.nn.functional as F

class PNN(nn.Module):
    def __init__(self, arch_num, arch_embed_dim, hidden_dim,  num_feature_field=9):
        super(PNN, self).__init__()

        # load parameters info
        self.num_feature_field = num_feature_field
        self.arch_num = arch_num
        self.embedding_size = arch_embed_dim
        self.add_module('arch_emb_fm', nn.Embedding(arch_num, arch_embed_dim))
        # self.add_module('arch_emb', nn.Embedding(arch_num, arch_embed_dim))
        self.num_pair = int(self.num_feature_field * (self.num_feature_field - 1) / 2)

        product_out_dim = self.num_feature_field * self.embedding_size
        product_out_dim += self.num_pair
        self.inner_product = InnerProductLayer(self.num_feature_field)
        self.add_module('fc_1', nn.Linear(product_out_dim, hidden_dim*2))
        self.add_module('fc_2', nn.Linear(hidden_dim*2, hidden_dim))
        # parameters initialization
        self.apply(self._init_weights)

    def reg_loss(self):
        reg_loss = 0
        for name, parm in self.mlp_layers.named_parameters():
            if name.endswith('weight'):
                reg_loss = reg_loss + self.reg_weight * parm.norm(2)
        return reg_loss


    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            xavier_normal_(module.weight.data)
        elif isinstance(module, nn.Linear):
            xavier_normal_(module.weight.data)
            if module.bias is not None:
                constant_(module.bias.data, 0)

    def forward(self, x, params=None, fix_embedding=True):
        if params == None:
            pnn_all_embeddings = self.arch_emb_fm(x)
            # single_emb = self.arch_emb(x)
            batch_size = pnn_all_embeddings.shape[0]
            # linear_part = single_emb.view(batch_size, -1)
            linear_part = pnn_all_embeddings.view(batch_size, -1)  # [batch_size,num_field*embed_dim]
            output = [linear_part]
            inner_product = self.inner_product(pnn_all_embeddings).view(batch_size, -1)  # [batch_size,num_pairs]
            output.append(inner_product)
            output = torch.cat(output, dim=1)   # [batch_size,d]
            output = F.relu(self.fc_1(output))
            output = self.fc_2(output)
            output = F.relu(output)
        else:
            pnn_all_embeddings = F.embedding(x, params['meta_learner.arch_encoder.arch_emb_fm.weight'])
            # single_emb = F.embedding(x, params['meta_learner.arch_encoder.arch_emb.weight'])
            batch_size = pnn_all_embeddings.shape[0]
            # linear_part = single_emb.view(batch_size, -1)
            linear_part = pnn_all_embeddings.view(batch_size, -1)  # [batch_size,num_field*embed_dim]
            output = [linear_part]
            inner_product = self.inner_product(pnn_all_embeddings).view(batch_size, -1)  # [batch_size,num_pairs]
            output.append(inner_product)
            output = torch.cat(output, dim=1)  # [batch_size,d]
            output = F.linear(output, params['meta_learner.arch_encoder.fc_1.weight'], params['meta_learner.arch_encoder.fc_1.bias'])
            output = F.relu(output)
            output = F.linear(output, params['meta_learner.arch_encoder.fc_2.weight'], params['meta_learner.arch_encoder.fc_2.bias'])
            output = F.relu(output)
          # [batch_size,1]

        return output


class InnerProductLayer(nn.Module):
    def __init__(self, num_feature_field):
        super(InnerProductLayer, self).__init__()
        self.num_feature_field = num_feature_field

    def forward(self, feat_emb):
        row = []
        col = []
        for i in range(self.num_feature_field - 1):
            for j in range(i + 1, self.num_feature_field):
                row.append(i)
                col.append(j)
        p = feat_emb[:, row]  # [batch_size, num_pairs, emb_dim]
        q = feat_emb[:, col]  # [batch_size, num_pairs, emb_dim]

        inner_product = p * q

        return inner_product.sum(dim=-1)  # [batch_size, num_pairs]


