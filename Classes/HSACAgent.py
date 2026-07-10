import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from Classes.ActorNet import ActorNet
from Classes.CriticNet import CriticNet
from Classes.ReplayBuffer import ReplayBuffer
import qiskit_script

class HSACAgent:
    def __init__(self, state_dim:int, n_units:int, device:str='cpu', use_quantum_features:bool=False):
        self.device = torch.device(device)
        self.n_units = n_units
        self.state_dim = state_dim
        self.actor = ActorNet(state_dim, n_units).to(self.device)
        self.q1 = CriticNet(state_dim, n_units).to(self.device)
        self.q2 = CriticNet(state_dim, n_units).to(self.device)
        self.q1t = CriticNet(state_dim, n_units).to(self.device)
        self.q2t = CriticNet(state_dim, n_units).to(self.device)
        self.q1t.load_state_dict(self.q1.state_dict()); self.q2t.load_state_dict(self.q2.state_dict())
        self.opt_actor = optim.Adam(self.actor.parameters(), lr=3e-4)
        self.opt_q1 = optim.Adam(self.q1.parameters(), lr=3e-4)
        self.opt_q2 = optim.Adam(self.q2.parameters(), lr=3e-4)
        self.replay = ReplayBuffer(200_000, state_dim, n_units, self.device)
        self.gamma = 0.99
        self.tau = 5e-3
        self.batch = 256
        self.alpha = 0.05
        self.use_quantum_features = use_quantum_features

    def select_action(self, s:np.ndarray, deterministic:bool=False):
        self.actor.eval()
        with torch.no_grad():
            st = torch.tensor(s, dtype=torch.float32, device=self.device).unsqueeze(0)
            logits = self.actor(st)
            probs = torch.sigmoid(logits)
            if deterministic:
                a = (probs > 0.5).float()
            else:
                a = torch.bernoulli(probs)
            return a.cpu().numpy().squeeze(0), probs.cpu().numpy().squeeze(0)

    def update(self, updates:int=1):
        if (not self.replay.full) and (self.replay.ptr < 1000):
            return
        for _ in range(updates):
            try:
                s,a,r,s2,d = self.replay.sample(self.batch)
            except RuntimeError:
                return
            with torch.no_grad():
                logits_next = self.actor(s2)
                probs_next = torch.sigmoid(logits_next)
                a2 = probs_next
                q1t = self.q1t(s2, a2)
                q2t = self.q2t(s2, a2)
                ent = - (probs_next*probs_next.clamp(1e-9).log()).sum(dim=-1, keepdim=True)
                q_target = torch.min(q1t, q2t) - self.alpha * ent
                y = r.squeeze(-1) + self.gamma * (1.0 - d.squeeze(-1)) * q_target
            q1_pred = self.q1(s, a)
            q2_pred = self.q2(s, a)
            loss_q1 = F.mse_loss(q1_pred, y)
            loss_q2 = F.mse_loss(q2_pred, y)
            self.opt_q1.zero_grad(); loss_q1.backward(); self.opt_q1.step()
            self.opt_q2.zero_grad(); loss_q2.backward(); self.opt_q2.step()
            logits = self.actor(s)
            probs = torch.sigmoid(logits)
            a_sample = torch.bernoulli(probs)
            q1_a = self.q1(s, a_sample); q2_a = self.q2(s, a_sample)
            q_a = torch.min(q1_a, q2_a)
            entropy = - (probs*probs.clamp(1e-9).log() + (1-probs)*(1-probs).clamp(1e-9).log()).sum(dim=-1, keepdim=True)
            actor_loss = - (q_a + self.alpha * entropy).mean()
            self.opt_actor.zero_grad(); actor_loss.backward(); self.opt_actor.step()
            with torch.no_grad():
                for p, tp in zip(self.q1.parameters(), self.q1t.parameters()):
                    tp.data.mul_(1 - self.tau).add_(self.tau * p.data)
                for p, tp in zip(self.q2.parameters(), self.q2t.parameters()):
                    tp.data.mul_(1 - self.tau).add_(self.tau * p.data)

    def store_transition(self, s, a, r, s2, d):
        self.replay.add(s,a,r,s2,d)