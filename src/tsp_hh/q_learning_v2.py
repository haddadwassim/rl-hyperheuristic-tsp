import numpy as np

from tsp_hh.hyper_env_v2 import HyperHeuristicStateV2
from tsp_hh.q_learning import QLearningAgent


def discretize_state_v2(
    state: HyperHeuristicStateV2,
    initial_length: float,
    max_steps: int,
) -> tuple[int, int, int, int]:
    """
    Discretize the V2 hyper-heuristic state.

    Features:
    1. current gap: how far current_length is from best_length
    2. last improvement: size of the latest improvement
    3. stagnation level
    4. progress through the episode
    """
    if initial_length <= 0:
        raise ValueError("initial_length must be positive")

    current_gap = (state.current_length - state.best_length) / initial_length
    last_improvement_ratio = state.last_improvement / initial_length
    stagnation_ratio = state.steps_without_improvement / max_steps
    progress_ratio = state.step_count / max_steps

    current_gap_bin = int(np.digitize(
        current_gap,
        bins=[0.0001, 0.001, 0.005, 0.01, 0.05],
    ))

    last_improvement_bin = int(np.digitize(
        last_improvement_ratio,
        bins=[0.0, 0.0001, 0.001, 0.005, 0.01],
    ))

    stagnation_bin = int(np.digitize(
        stagnation_ratio,
        bins=[0.05, 0.10, 0.25, 0.50, 0.75],
    ))

    progress_bin = int(np.digitize(
        progress_ratio,
        bins=[0.25, 0.50, 0.75],
    ))

    return current_gap_bin, last_improvement_bin, stagnation_bin, progress_bin


__all__ = [
    "QLearningAgent",
    "discretize_state_v2",
]