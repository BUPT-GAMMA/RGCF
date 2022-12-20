import torch.nn as nn
import torch.nn.functional as F
import torch
class DatasetEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(DatasetEncoder, self).__init__()
        self.layer_one = nn.Linear(input_dim, hidden_dim)
        self.layer_two = nn.Linear(hidden_dim, output_dim)
        self.reconstruct = MetaFeatureReconstruct(input_dim, hidden_dim, output_dim)
        self.loss = nn.MSELoss()

    def forward(self, x, inner_loop=False):

        if inner_loop:
            with torch.no_grad():
                x = self.layer_one(x)
                x = F.relu(x)
                x = self.layer_two(x)
            re_loss = None
        else:
            raw = x
            x = self.layer_one(x)
            x = F.relu(x)
            x = self.layer_two(x)
            y = self.reconstruct(x)
            re_loss = self.loss(raw, y)
        return x, re_loss

class MetaFeatureReconstruct(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MetaFeatureReconstruct, self).__init__()
        self.layer_reconstruct_one = nn.Linear(output_dim, hidden_dim)
        self.layer_reconstruct_two = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        y = F.relu(self.layer_reconstruct_one(x))
        y = self.layer_reconstruct_two(y)
        return y