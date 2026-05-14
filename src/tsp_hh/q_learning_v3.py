import numpy as np

from tsp_hh.hyper_env_v3 import HyperHeuristicStateV3
from tsp_hh.q_learning import QLearningAgent


def discretize_state_v3(
    state: HyperHeuristicStateV3,
    max_steps: int,
) -> tuple[int, int, int, int, int]:
    """
    Discretize V3 hierarchical state for tabular Q-learning.

    Features:
    1. current gap between current and best
    2. last improvement ratio
    3. stagnation level
    4. progress through episode
    5. last category
    """
    initial = max(state.initial_length, 1e-12)

    current_gap = (state.current_length - state.best_length) / initial
    last_improvement_ratio = state.last_improvement / initial
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
        bins=[0.02, 0.05, 0.10, 0.25, 0.50, 0.75],
    ))

    progress_bin = int(np.digitize(
        progress_ratio,
        bins=[0.25, 0.50, 0.75],
    ))

    last_category_bin = int(state.last_category + 1)
    # last_category = -1,0,1,2 becomes 0,1,2,3

    return (
        current_gap_bin,
        last_improvement_bin,
        stagnation_bin,
        progress_bin,
        last_category_bin,
    )


__all__ = [
    "QLearningAgent",
    "discretize_state_v3",
]