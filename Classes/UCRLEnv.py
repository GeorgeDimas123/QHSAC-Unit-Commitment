import math
from typing import List, Tuple
import numpy as np
import json
from Classes.reward_shaper import RewardShaper
from Classes.QuantumFeatureSampler import QuantumFeatureSampler
import qiskit_script
import os

# C:\Users\Georg\.julia\juliaup\julia-1.11.8+0.x64.w64.mingw32\bin

from juliacall import Main as jl
jl.include("Classes/julia_script.jl")

class UCRLEnv:
    """
    Receding-horizon environment using PyUC helpers.
    Action: binary vector of length num_units for the current hour.
    """
    def __init__(self, instance_name=qiskit_script.POWER_SYSTEM, use_quantum_features:bool=False):
        gens, T = jl.PyUC.load_instance(instance_name)
        self.MILP_cost = jl.PyUC.get_MILP_cost(instance_name)
        self.gen_names: List[str] = list(gens)
        self.T: int = int(T)
        self.num_units = len(self.gen_names)
        self.t = 1
        self.fixed: List[Tuple[str,int,int]] = []
        self.prev_obj = None
        # baseline MILP (may take time)
        obj_star, _ = jl.PyUC.solve_with_fixes([])  # baseline solve
        self.reward_shaper = RewardShaper(J_star_total=obj_star, T=self.T)
        # quantum features
        self.use_quantum = use_quantum_features
        if self.use_quantum:
            self.qsampler = QuantumFeatureSampler(n_qubits=self.num_units, shots=128, train_mode="simulator", eval_mode="qpu", ibm_backend="ibm_oslo")
        else:
            self.qsampler = None
    
    def get_milp_cost(self):
        return self.MILP_cost

    def reset(self):
        self.t = 1
        self.fixed.clear()
        self.prev_obj = None
        return self._state([0]*self.num_units)

    def _state(self, prev_bits: List[int]):
        angle = 2 * math.pi * ((self.t - 1) % 24) / 24.0
        base = np.array([math.sin(angle), math.cos(angle)] + list(prev_bits), dtype=np.float32)
        if self.use_quantum:
            qfeat = self.qsampler.run(base.reshape(1,-1), training=True)[0]  # training=True -> Aer
            return np.concatenate([base, qfeat.astype(np.float32)])
        else:
            return base

    def step(self, action_bits: List[int]):
        assert len(action_bits) == self.num_units
        # early infeasibility check
        total_cap = sum(action_bits) * 1e3  # fallback
        if sum(action_bits) == 0 or total_cap < 0.1:
            reward = -5e3
            return self._state(action_bits), reward, True, {"status":"trivial_infeasible"}
        # apply fixes
        for g, bit in zip(self.gen_names, action_bits):
            self.fixed.append((g, self.t, int(bit)))
        obj, sol_json = jl.PyUC.solve_with_fixes(self.fixed)
        if obj is None or sol_json is None:
            reward = -self.reward_shaper.w.hard_penalty * 0.1
            return self._state(action_bits), reward, True, {"status":"solver_failed"}
        # parse solution robustly
        try:
            sol = json.loads(sol_json)
        except Exception:
            sol = {}
        costs_block = sol.get("cost") or sol.get("costs") or sol.get("objective") or {}
        gen_cost = None
        if isinstance(costs_block, dict):
            for key in ("generation","gen","gen_cost","energy","fuel","total_cost"):
                if key in costs_block:
                    try:
                        gen_cost = float(costs_block[key])
                        break
                    except Exception:
                        pass
        if gen_cost is None:
            try:
                gen_cost = float(obj)
            except Exception:
                gen_cost = 0.0
        # extract other cost components safely
        no_load  = float(costs_block.get("no_load", 0.0)) if isinstance(costs_block, dict) else 0.0
        startup  = float(costs_block.get("startup", 0.0)) if isinstance(costs_block, dict) else 0.0
        shutdown = float(costs_block.get("shutdown", 0.0)) if isinstance(costs_block, dict) else 0.0

        uc_dict = {
            "feasible": True,
            "costs": {"gen": gen_cost, "no_load": no_load, "startup": startup, "shutdown": shutdown},
            "p_t": [],
            "u_t": list(action_bits),
            "solve_sec": sol.get("solve_time_sec", 0.0)
        }
        forecasts = {"next_net_load_MW": sol.get("next_net_load_MW", sol.get("Pd_sum_MW", 0.0))}
        reward, done_flag = self.reward_shaper.step_reward(t=self.t, uc=uc_dict, state={}, forecasts=forecasts, gen_var_costs=[0.0]*self.num_units, pmax=[0.0]*self.num_units)
        self.prev_obj = gen_cost
        done = (self.t >= self.T) or bool(done_flag)
        self.t += 1
        next_state = None if done else self._state(action_bits)
        info = {"objective": gen_cost, "solution_json": sol_json if done else None}
        return next_state, reward, done, info