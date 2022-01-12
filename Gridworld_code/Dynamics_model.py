import torch
from torch import nn
import numpy as np 

class Dynamics_model(nn.Module):
    def __init__ (self, state_size, action_dim, hidden_size):
        super(Dynamics_model, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(in_features = state_size + action_dim, out_features = hidden_size, bias = True),
            nn.ReLU(),
            nn.Linear(in_features = hidden_size, out_features = state_size, bias = True)
        )

    def forward(self, x):
    
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.double)

        x = self.net(x)
        return x