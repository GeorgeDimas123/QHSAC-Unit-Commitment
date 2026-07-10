import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import json
import math
from typing import List
import qiskit
from qiskit import QuantumCircuit, transpile

QUICK_TEST = False    # True -> very small train for debugging
EVAL_ON_QPU = True    # True -> attempt real IBM QPU evaluation (ibm_kingston)
QPU_BACKEND = "ibm_kingston"
RESULTS_ROOT = "results"
# training sizes
TRAIN_EPISODES_QUICK = 1 # <-- CHANGE THIS FOR QPU TRAINING SESSIONS
TRAIN_EPISODES_FULL = 10000 # <-- CHANGE THIS FOR SIMULATION TRAINING SESSIONS
POWER_SYSTEM = "matpower/case14/2017-01-01"  # <-- CHANGE THIS TO YOUR PREFERRED TEST INSTANCE
# matpower/case1354pegase/2017-01-01	

# Quantum evaluation parameters (keeps jobs short)
QPU_EVAL_EPISODES = 1      # small number of episodes to evaluate
QPU_EVAL_SHOTS = 512       # shots per circuit submission for marginals
# ----------------------------

api_token = "l0gEJmwMxHis48jyx3WR-PvPjiSdL3ySkfs9mub3XW83"
current_instance = "crn:v1:bluemix:public:quantum-computing:us-east:a/3fd1683bdb2f4bd7b5a9c16264e9531f:6b8b978e-35aa-45f6-8661-1f4f85c2c7a5::"
from qiskit_ibm_runtime import QiskitRuntimeService
QiskitRuntimeService.save_account(
    channel="ibm_cloud",
    token=api_token,
    instance=current_instance,
    set_as_default=True,
    overwrite=True
)