import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import json
import math
import datetime
import pickle
from typing import List

def make_results_dir(base="results"):
    now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    d = os.path.join(base, f"hsac_run_{now}")
    os.makedirs(d, exist_ok=True)
    return d

def plot_learning_curve(episode_costs, outdir, milp_cost=None, window=20, save_png="learning_curve.png"):
    eps = np.arange(1, len(episode_costs)+1)
    plt.figure(figsize=(10,5))
    plt.plot(eps, episode_costs, marker='o', linestyle='-', alpha=0.6, label='Episode final cost')
    if len(episode_costs) >= window:
        ma = np.convolve(episode_costs, np.ones(window)/window, mode='valid')
        plt.plot(eps[window-1:], ma, linewidth=2.5, label=f'{window}-ep MA')
    if milp_cost is not None:
        plt.axhline(y=milp_cost, color='r', linestyle='--', label='MILP optimal')
    plt.xlabel('Episode'); plt.ylabel('Final episode cost ($)')
    plt.title('HSAC RL learning curve'); plt.legend(); plt.grid(True)
    plt.tight_layout()
    fn = os.path.join(outdir, save_png)
    plt.savefig(fn)
    plt.close()
    print("Saved", fn)

def plot_histogram(costs, outdir, bins=30, save_png="cost_hist.png"):
    plt.figure(figsize=(6,4))
    plt.hist(costs, bins=bins, alpha=0.8)
    plt.xlabel('Episode final cost'); plt.ylabel('Count'); plt.title('Distribution of final costs')
    plt.tight_layout()
    fn = os.path.join(outdir, save_png)
    plt.savefig(fn)
    plt.close()
    print("Saved", fn)

def save_episode_data(episode_costs, history_actions, outdir, fname_prefix="hsac_run"):
    with open(os.path.join(outdir, f"{fname_prefix}_episode_costs.pkl"), "wb") as f:
        pickle.dump(episode_costs, f)
    with open(os.path.join(outdir, f"{fname_prefix}_actions.pkl"), "wb") as f:
        pickle.dump(history_actions, f)
    print("Saved episode data to", outdir)

def summarize_actions(history_actions, topk=10):
    from collections import Counter
    ints = []
    for a in history_actions:
        if a is None: continue
        if isinstance(a, (list, tuple, np.ndarray)):
            val = sum(int(b) << i for i,b in enumerate(a))
            ints.append(val)
        else:
            ints.append(int(a))
    c = Counter(ints)
    most = c.most_common(topk)
    print("Top actions (int, count):", most)
    return most