# QHSAC Unit Commitment
The Qauntum Hybrid Soft Actor-Critic agent presented here is intended to demonstrate the potiential implementation of quantum computing techniques for the well-known HSAC reinforcement learning framework. This repository contains the code and data of the corresponding IEEE SmartGridComm 2026 paper, "Feasibility-Aware Security-Constrained Unit Commitment via Hybrid Soft Actor-Critic with Quantum-Sampled Features".

IEEE SmartGridComm 2026 publication: https://arxiv.org/abs/2606.26345

# Installation & Setup
Run the following command in python:
```python
pip install numpy torch matplotlib qiskit qiskit-aer qiskit-ibm-runtime juliacall
```
Run the following command in julia as well:
```julia
using Pkg
Pkg.add(["UnitCommitment", "JuMP", "Cbc", "JSON", "Gurobi", "MathOptInterface"])
```

# How to use
After installing the necessary packages, modify the training parameters. Customize the number of training episodes for the agent, as well as the power system instance that the agent should train on. Benchmark instances can be found at the UnitCommitment.jl documentation website [here](https://anl-ceeesa.github.io/UnitCommitment.jl/0.3/instances/).
```python
TRAIN_EPISODES_FULL = 1000 # <-- CHANGE THIS FOR SIMULATION TRAINING SESSIONS
POWER_SYSTEM = "matpower/case57/2017-01-01"  # <-- CHANGE THIS TO YOUR PREFERRED TEST INSTANCE	
```

Once the training parameters are inputted, run the following command in the terminal:
```windows
python main.py
```

# Authors
George Dimas (University of Missouri, Columbia)

Amin Masoumi (University of Missouri, Columbia)

Mert Korkali (University of Missouri, Columbia)

# Citing 
If you use this work in your research, we kindly ask that you cite this package as follows: 
Dimas, George, Amin Masoumi, and Mert Korkali. "Feasibility-Aware Security-Constrained Unit Commitment via Hybrid Soft Actor-Critic with Quantum-Sampled Features." arXiv preprint arXiv:2606.26345 (2026).

# References
[1] Y. Chen et al., “Security-constrained unit commitment for electricity
market: Modeling, solution methods, and future challenges,” IEEE Trans.
Power Syst., vol. 38, no. 5, pp. 4668–4681, Sep. 2023.

[2] Y. Yang and L. Wu, “Machine learning approaches to the unit commitment
problem: Current trends, emerging challenges, and new strategies,” Electr.
J., vol. 34, no. 1, Jan.-Feb. 2021, Art. no. 106889.

[3] Á. S. Xavier, F. Qiu, and S. Ahmed, “Learning to solve large-scale
security-constrained unit commitment problems,” INFORMS J. Comput.,
vol. 33, no. 2, pp. 739–756, 2021.

[4] S. Pineda and J. M. Morales, “Is learning for the unit commitment
problem a low-hanging fruit?” Electr. Power Syst. Res., vol. 207, 2022,
Art. no. 107851.

[5] Y. Dai et al., “Deep reinforcement learning explanation-assisted integer
variable reduction method for security-constrained unit commitment,”
Eng. Appl. Artif. Intell., vol. 144, Mar. 2025, Art. no. 110139.

[6] B. Venkatesh, M. I. A. Shekeew, and J. Ma, “Feasibility-guaranteed
machine learning unit commitment: Fuzzy optimization approaches,”
Appl. Energy, vol. 379, Feb. 2025, Art. no. 124923.

[7] G. Wang et al., “Structure-aware commitment reduction for network-
constrained unit commitment with solver-preserving guarantees,” arXiv
preprint arXiv:2604.02788, 2026.

[8] J. Xiong et al., “Successive fixing for large-scale security-constrained unit
commitment using first-order methods,” arXiv preprint arXiv:2510.10891,
2025.

[9] P. de Mars and A. O’Sullivan, “Applying reinforcement learning and
tree search to the unit commitment problem,” Appl. Energy, vol. 302,
Nov. 2021, Art. no. 117519.

[10] P. de Mars and A. O’Sullivan, “Reinforcement learning and A* search
for the unit commitment problem,” Energy AI, vol. 9, Aug. 2022, Art.
no. 100179.

[11] J. Qin et al., “An optimization method-assisted ensemble deep reinforce-
ment learning algorithm to solve unit commitment problems,” IEEE
Access, vol. 11, pp. 100 125–100 136, 2023.

[12] A. R. Sayed et al., “Deep reinforcement learning-assisted convex
programming for AC unit commitment and its variants,” IEEE Trans.
Power Syst., vol. 39, no. 4, pp. 5561–5574, Jul. 2024.

[13] G. Xu et al., “Deep reinforcement learning based model-free optimization
for unit commitment against wind power uncertainty,” Int. J. Electr. Power
Energy Syst., vol. 155, Jan. 2024, Art. no. 109526.

[14] J. Yan et al., “Look-ahead unit commitment with adaptive horizon based
on deep reinforcement learning,” IEEE Trans. Power Syst., vol. 39, no. 2,
pp. 3673–3684, Mar. 2024.

[15] H. Liang, C. Lin, and A. Pang, “Expert knowledge data-driven based
actor-critic reinforcement learning framework to solve computationally
expensive unit commitment problems with uncertain wind energy,” Int.
J. Electr. Power Energy Syst., vol. 159, Aug. 2024, Art. no. 110033.

[16] W. Lu et al., “Graph reinforcement learning with auxiliary temporal-
graph convolutional neural network for unit commitment,” Int. J. Electr.
Power Energy Syst., vol. 176, Mar. 2026, Art. no. 111708.

[17] S. Koretsky et al., “Adapting quantum approximation optimization
algorithm (QAOA) for unit commitment,” in Proc. IEEE Int. Conf.
Quantum Comput. Eng. (QCE), 2021, pp. 181–187.

[18] W. Hong, W. Xu, and F. Teng, “Qubit-efficient quantum annealing for
stochastic unit commitment,” arXiv preprint arXiv:2502.15917v2, 2026.

[19] F. Feng et al., “Novel resolution of unit commitment problems through
quantum surrogate Lagrangian relaxation,” IEEE Trans. Power Syst.,
vol. 38, no. 3, pp. 2460–2471, May 2023.

[20] X. Zheng, J. Wang, and M. Yue, “A fast quantum algorithm for searching
the quasi-optimal solutions of unit commitment,” IEEE Trans. Power
Syst., vol. 39, no. 2, pp. 4755–4758, Mar. 2024.

[21] J. Liu et al., “Exact quantum algorithm for unit commitment optimization
based on partially connected quantum neural networks,” Chin. Phys. B,
vol. 34, no. 10, 2025, Art. no. 100303.

[22] X. Wei et al., “Quantum reinforcement learning based two-stage unit
commitment with integration of virtual power plants and renewable
energy,” J. Mod. Power Syst. Clean Energy, pp. 1–12, 2026, early access.

[23] W. Aboumrad et al., “A new hybrid quantum-classical algorithm for
solving the unit commitment problem,” in Proc. IEEE Int. Conf. Quantum
Comput. Eng. (QCE), 2025, pp. 1905–1915.

[24] M. Hasanzadeh and A. Kargarian, “D2-UC: A distributed-distributed
quantum-classical framework for unit commitment,” arXiv preprint
arXiv:2511.03104, 2025.

[25] R. Barrass, H. Nagarajan, and C. Coffrin, “Leveraging quantum comput-
ing for accelerated classical algorithms in power systems optimization,”
in Integration of Constraint Programming, Artificial Intelligence, and
Operations Research (CPAIOR), G. Tack, Ed. Cham: Springer Nature
Switzerland, 2025, pp. 52–67.

[26] M. Hasanzadeh and A. Kargarian, “A survey on applications of quantum
computing for unit commitment,” arXiv preprint arXiv:2601.01777, 2026.

[27] A. S. Xavier et al., “UnitCommitment.jl: A Julia/JuMP optimization
package for security-constrained unit commitment,” Zenodo, 2024.

[28] A. Vaswani et al., “Attention is all you need,” in Proc. Adv. Neural Inf.
Process. Syst. (NeurIPS), 2017, pp. 5998–6008.

[29] V. Havlíˇcek et al., “Supervised learning with quantum-enhanced feature
spaces,” Nature, vol. 567, no. 7747, pp. 209–212, 2019.

[30] A. Rahimi and B. Recht, “Random features for large-scale kernel
machines,” in Proc. Adv. Neural Inf. Process. Syst. (NeurIPS), 2007, pp.
1177–1184.

[31] T. Haarnoja et al., “Soft actor-critic: Off-policy maximum entropy deep
reinforcement learning with a stochastic actor,” in Proc. 35th Int. Conf.
Mach. Learn. (ICML), 2018, pp. 1861–1870.

[32] M. Lubin et al., “JuMP 1.0: Recent improvements to a modeling language
for mathematical optimization,” Math. Program. Comput., vol. 15, pp.
581–589, 2023.
