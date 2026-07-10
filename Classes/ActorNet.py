import torch
import torch.nn as nn
import qiskit_script

class ActorNet(nn.Module):
    def __init__(self, state_dim:int, n_units:int, hidden:int=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_units)
        )
    def forward(self, s:torch.Tensor):
        return self.net(s)  # raw logits per unit