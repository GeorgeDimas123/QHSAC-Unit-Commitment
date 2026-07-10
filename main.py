import os
from juliacall import Main as jl
import time
import json
import csv
import numpy as np
import matplotlib.pyplot as plt
import torch
from Classes.UCRLEnv import UCRLEnv
from qiskit_script import QUICK_TEST, TRAIN_EPISODES_QUICK, TRAIN_EPISODES_FULL, QPU_EVAL_EPISODES, QPU_EVAL_SHOTS, QPU_BACKEND
from Classes.reward_shaper import RewardShaper
from Classes.HSACAgent import HSACAgent
from plotting_functions import make_results_dir, save_episode_data, plot_learning_curve, plot_histogram, summarize_actions
import qiskit_script

# MAKING CHANGE ON NEW BRANCH

def train_hsac(episodes:int=TRAIN_EPISODES_FULL, instance_name:str=qiskit_script.POWER_SYSTEM,
               use_quantum_features:bool=False, device:str='cpu', outdir:str=None):
    env = UCRLEnv(instance_name, use_quantum_features=use_quantum_features)
    state0 = env.reset()
    state_dim = len(state0)
    agent = HSACAgent(state_dim, env.num_units, device=device, use_quantum_features=use_quantum_features)
    total_start = time.time()
    episode_costs = []
    history_actions = []
    episode_shots = []
    episode_times = []
    for ep in range(1, episodes+1):
        start_time = time.perf_counter()
        s = env.reset()
        done = False
        ep_reward = 0.0
        steps = 0
        ep_shots = 0
        while not done:
            a, aprob = agent.select_action(s, deterministic=False)
            a_bits = [int(x) for x in a.tolist()]
            # record action
            history_actions.append(list(a_bits))
            s2, r, done, info = env.step(a_bits)
            agent.store_transition(s, a, r, s2 if s2 is not None else np.zeros_like(s), done)
            agent.update(updates=2)
            s = s2 if s2 is not None else s
            ep_reward += r
            steps += 1

            if env.qsampler is not None:
                ep_shots += env.qsampler.shots

        end_time = time.perf_counter()
        episode_times.append(end_time - start_time)

        final_cost = env.prev_obj if env.prev_obj is not None else 1e6
        episode_costs.append(final_cost)
        episode_shots.append(ep_shots)
        if ep % 10 == 0 or ep == 1:
            elapsed = time.time() - total_start
            ma = np.mean(episode_costs[-20:]) if len(episode_costs) >= 1 else final_cost
            print(f"[EP {ep:04d}] final_cost={final_cost:.2f}   recent_MA20={ma:.2f}  elapsed={elapsed:.1f}s")
    print("Training done. Total time:", time.time()-total_start)
    # Save and plot if outdir provided
    if outdir is not None:
        try:
            save_episode_data(episode_costs, history_actions, outdir, fname_prefix="hsac_run")
            plot_learning_curve(episode_costs, outdir, milp_cost=None, window=20, save_png="hsac_learning_curve.png")
            plot_histogram(episode_costs, outdir, bins=30, save_png="hsac_cost_hist.png")
            summarize_actions(history_actions, topk=20)
            plot_shots_per_episode(episode_shots, outdir)
            plot_times_per_episode(episode_times, outdir)
            create_table(episode_costs, episode_shots, episode_times, outdir)
        except Exception as e:
            print("Post-processing failed:", e)
    return agent, episode_costs, history_actions, episode_shots, episode_times

def plot_shots_per_episode(episode_shots, outdir, save_png="shots_per_episode.png"):
    eps = np.arange(1, len(episode_shots)+1)

    plt.figure(figsize=(8,5))
    plt.plot(eps, episode_shots, marker='o')
    plt.xlabel("Episode")
    plt.ylabel("Total Shots Used")
    plt.title("Quantum Shots Used Per Episode")
    plt.grid(True)

    fn = os.path.join(outdir, save_png)
    plt.savefig(fn)
    plt.close()
    print("Saved", fn)

def plot_times_per_episode(episode_times, outdir, save_png="times_per_episode.png"):
    eps = np.arange(1, len(episode_times)+1)

    plt.figure(figsize=(8,5))
    plt.plot(eps, episode_times, marker='o')
    plt.xlabel("Episode")
    plt.ylabel("Time (seconds)")
    plt.title("Time Taken Per Episode")
    plt.grid(True)

    fn = os.path.join(outdir, save_png)
    plt.savefig(fn)
    plt.close()
    print("Saved", fn)

def create_table(episode_costs, episode_shots, episode_times, outdir, fname="summary_table.csv"):
    with open(os.path.join(outdir, fname), mode='w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["Episode", "Final Cost", "Shots Used", "Time (s)"])
        writer.writeheader()

        for i, (cost, shots, t) in enumerate(zip(episode_costs, episode_shots, episode_times), start=1):
            writer.writerow({
                "Episode": i,
                "Final Cost": cost,
                "Shots Used": shots,
                "Time (s)": t
            })

# -------------------------
# CLI entry point
# -------------------------
if __name__ == "__main__":
    print("HSAC + Quantum UC single-file runner")
    outdir = make_results_dir()
    # select training size
    episodes = TRAIN_EPISODES_QUICK if QUICK_TEST else TRAIN_EPISODES_FULL
    print(f"QUICK_TEST={QUICK_TEST}; running {episodes} episodes. Results will be saved to: {outdir}")

    milp_cost = UCRLEnv(instance_name=qiskit_script.POWER_SYSTEM).get_milp_cost()
    print("MILP baseline cost (case1354pegase):", milp_cost)

    # traing RL on simulator first
    agent_sim, costs_sim, history_actions_sim, shots_sim, times_sim = train_hsac(episodes=episodes, instance_name=qiskit_script.POWER_SYSTEM,
                                                  use_quantum_features=True, device='cpu', outdir=outdir)
        
    print(f"running {QPU_EVAL_EPISODES} episodes. Results will be saved to: {outdir}")

    # Save trained actor weights from simulation traing (torch)
    try:
        torch.save(agent_sim.actor.state_dict(), os.path.join(outdir, "actor_state_dict.pt"))
        torch.save(agent_sim.q1.state_dict(), os.path.join(outdir, "q1_state_dict.pt"))
        torch.save(agent_sim.q2.state_dict(), os.path.join(outdir, "q2_state_dict.pt"))
        print("Saved model state dicts to", outdir)
    except Exception as e:
        print("Failed to save model state dicts:", e)

    #print("Training finished. Sample final costs (last 10):", costs[-10:])
    print("All done. Results folder:", outdir)