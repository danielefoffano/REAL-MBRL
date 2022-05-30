import torch
import torch.nn as nn
from torch.nn.functional import normalize
import numpy as np

class Learned_model(nn.Module):

    def __init__(self, observation_space_size, action_dim, hidden_size, device):

        super(Learned_model, self).__init__()
        self.dev = device

        self.net = nn.Sequential(
            nn.Linear(in_features = observation_space_size + action_dim, out_features = hidden_size, bias = True),
            nn.ReLU(),
            nn.Linear(in_features = hidden_size, out_features = hidden_size, bias = True),
            nn.ReLU(),
            nn.Linear(in_features = hidden_size, out_features = hidden_size, bias = True),
            nn.ReLU(),
            nn.Linear(in_features = hidden_size, out_features = observation_space_size, bias = True)
        )

    def forward(self, x):
    
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float)

        if len(x.shape) > 1:
            x = normalize(x, dim=1) #check source
        else:
            x = normalize(x, dim=0)

        x = self.net(x.to(self.dev))
        return x