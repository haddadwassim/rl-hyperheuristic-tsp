# Adaptive Reinforcement Learning Hyper-Heuristic for the Traveling Salesman Problem

This project implements and evaluates a reinforcement learning-based hyper-heuristic for the Traveling Salesman Problem (TSP).

The goal is not to replace highly optimized solvers such as LKH, but to study whether an adaptive learning-based heuristic selector can improve robustness over fixed local-search strategies.

## Main Idea

Instead of directly constructing TSP tours, the reinforcement learning agent selects among several low-level heuristics such as:

- 2-opt
- swap
- insertion
- random perturbation
- local search intensification

At each step, the agent observes the current search state and chooses which heuristic to apply next.

## Planned Structure

1. Implement TSP instance generation and tour evaluation.
2. Implement baseline heuristics.
3. Implement a hyper-heuristic environment.
4. Train a reinforcement learning agent.
5. Compare against fixed heuristic baselines.
6. Analyze performance on different TSP instance sizes and types.

## Repository Structure

```text
src/tsp_hh/      Core implementation
experiments/     Experiment scripts
data/            TSP instances
results/         Logs, metrics, and plots
notebooks/       Exploratory analysis
tests/           Unit tests