# ENTER YOUR PREFERRED TRAINING PARAMETERS HERE 

# training sizes
TRAIN_EPISODES_FULL = 5000 # <-- CHANGE THIS FOR SIMULATION TRAINING SESSIONS
POWER_SYSTEM = "matpower/case300/2017-01-01"  # <-- CHANGE THIS TO YOUR PREFERRED TEST INSTANCE	

# Quantum evaluation parameters 
QPU_EVAL_EPISODES = 1      # small number of episodes to evaluate
QPU_EVAL_SHOTS = 512       # shots per circuit submission for marginals