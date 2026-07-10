import torch
import torch.nn as nn
import qiskit_script

class CriticNet(nn.Module):
    def __init__(self, state_dim:int, n_units:int, hidden:int=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + n_units, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1)
        )
    def forward(self, s:torch.Tensor, a:torch.Tensor):
        x = torch.cat([s, a], dim=-1)
        return self.net(x).squeeze(-1)