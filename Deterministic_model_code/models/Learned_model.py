import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import normalize

class Learned_model(nn.Module):

    def __init__(self, observation_space_size, action_dim, hidden_size):

        super(Learned_model, self).__init__()

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
            x = normalize(x, dim=1)
        else:
            x = normalize(x, dim=0)

        x = self.net(x)
        return x
