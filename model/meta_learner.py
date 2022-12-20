import torch.nn as nn
from collections import OrderedDict
from model.cross_predictor import Predictor,DotPredictor

class MetaLearner(nn.Module):
    def __init__(self, arch_num,
                 arch_embed_dim,
                 hidden_dim,
                 n_heads,
                 n_layers,
                 cross_hid_dim,
                 dataset_on=True,
                 dataset_input_dim=None,
                 dataset_hid_dim=None,
                 dataset_out_dim=None,
                 dot_on = False):
        super(MetaLearner, self).__init__()
        if dot_on:
            self.meta_learner = DotPredictor(arch_num, arch_embed_dim, hidden_dim, n_heads, n_layers, cross_hid_dim,
                                          dataset_on=dataset_on, dataset_input_dim=dataset_input_dim,
                                          dataset_hid_dim=dataset_hid_dim, dataset_out_dim=dataset_out_dim)
        else:
            self.meta_learner = Predictor(arch_num, arch_embed_dim, hidden_dim, n_heads, n_layers, cross_hid_dim,
                                      dataset_on=dataset_on, dataset_input_dim=dataset_input_dim,
                                      dataset_hid_dim=dataset_hid_dim, dataset_out_dim=dataset_out_dim)
        self.init_parameters()


    def forward(self, X, dataset_feature=None, adapted_params=None, inner_loop=False):
        if dataset_feature is not None:
            dataset_feature = dataset_feature.repeat(len(X), 1)
        if adapted_params == None:
            out, re_loss = self.meta_learner(X, dataset_feature=dataset_feature, inner_loop=inner_loop)
        else:
            out, re_loss = self.meta_learner(X, dataset_feature=dataset_feature, params=adapted_params, inner_loop=inner_loop)
        return out, re_loss

    def mix_forward(self, X_i, X_j, dataset_feature_i, dataset_feature_j, lam, inter_layer, adapted_params=None, inner_loop=False ):
        dataset_feature_i = dataset_feature_i.repeat(len(X_i), 1)
        dataset_feature_j = dataset_feature_j.repeat(len(X_j), 1)
        if adapted_params == None:
            out, re_loss = self.meta_learner.mix_forward(X_i, X_j, dataset_feature_i, dataset_feature_j, lam, inter_layer, inner_loop=inner_loop)
        else:
            out, re_loss = self.meta_learner.mix_forward(X_i, X_j, dataset_feature_i, dataset_feature_j, lam, inter_layer, params=adapted_params, inner_loop=inner_loop)
        return out, re_loss

    def mix_no_feature_forward(self, X_i, X_j, lam, inter_layer, adapted_params=None):
        re_loss = None
        if adapted_params == None:
            out = self.meta_learner.mix_forward(X_i, X_j, lam, inter_layer)
        else:
            out = self.meta_learner.mix_no_feature_forward(X_i, X_j, lam, inter_layer, params=adapted_params)
        return out, re_loss

    def cloned_params(self):
        params = OrderedDict()
        for (key, val) in self.named_parameters():
            params[key] = val.clone()
        return params

    def init_parameters(self):
        for layer in self.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()