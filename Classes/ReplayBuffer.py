import torch
import numpy as np
from typing import List
import qiskit_script

class ReplayBuffer:
    def __init__(self, capacity:int, state_dim:int, n_units:int, device):
        self.capacity = capacity
        self.device = device
        self.ptr = 0
        self.full = False
        self.states = torch.zeros((capacity, state_dim), dtype=torch.float32, device=device)
        self.actions = torch.zeros((capacity, n_units), dtype=torch.float32, device=device)
        self.rewards = torch.zeros((capacity,1), dtype=torch.float32, device=device)
        self.next_states = torch.zeros((capacity, state_dim), dtype=torch.float32, device=device)
        self.dones = torch.zeros((capacity,1), dtype=torch.float32, device=device)
    def add(self, s,a,r,s2,d):
        i = self.ptr
        self.states[i].copy_(torch.tensor(s, device=self.device))
        self.actions[i].copy_(torch.tensor(a, device=self.device))
        self.rewards[i,0] = float(r)
        self.next_states[i].copy_(torch.tensor(s2 if s2 is not None else np.zeros_like(s), device=self.device))
        self.dones[i,0] = float(d)
        self.ptr = (i+1) % self.capacity
        if self.ptr == 0:
            self.full = True
    def sample(self, batch:int):
        n = self.capacity if self.full else self.ptr
        if n == 0:
            raise RuntimeError("Replay buffer is empty")
        idx = np.random.randint(0, n, size=batch)
        return (self.states[idx], self.actions[idx], self.rewards[idx], self.next_states[idx], self.dones[idx])