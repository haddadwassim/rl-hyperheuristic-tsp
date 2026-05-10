# Adaptive Reinforcement Learning Hyper-Heuristic for the Traveling Salesman Problem

This repository implements a reinforcement learning-based hyper-heuristic for the Traveling Salesman Problem (TSP).

The goal is not to replace highly optimized TSP solvers such as LKH or Concorde. Instead, the project studies whether a lightweight learning-based heuristic selector can adaptively choose among several low-level heuristics and improve robustness compared with fixed or random heuristic strategies.

## Idea

The method starts from an initial TSP tour and repeatedly selects one low-level heuristic to apply.

The low-level heuristics currently include:

- first-improvement 2-opt
- best-improvement 2-opt
- random swap
- random insertion
- perturbation

A tabular Q-learning agent observes a compact state describing the search process and learns which heuristic action to apply.

## Repository Structure

```text
rl-hyperheuristic-tsp/
│
├── src/tsp_hh/
│   ├── instances.py       # TSP instance generation and distance matrix
│   ├── tour.py            # Tour validation and tour length
│   ├── heuristics.py      # Low-level TSP heuristics
│   ├── hyper_env.py       # Hyper-heuristic environment
│   └── q_learning.py      # Tabular Q-learning agent
│
├── experiments/
│   ├── run_baselines.py
│   ├── run_random_hh.py
│   ├── run_comparison.py
│   ├── train_q_learning.py
│   ├── evaluate_q_learning.py
│   └── plot_comparison.py
│
├── tests/
├── data/
├── results/
├── notebooks/
├── requirements.txt
├── pyproject.toml
└── README.md