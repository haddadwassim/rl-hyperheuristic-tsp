from dataclasses import dataclass, field

import numpy as np

from tsp_hh.hyper_env import HyperHeuristicState


def discretize_state(
    state: HyperHeuristicState,
    initial_length: float,
    max_steps: int,
) -> tuple[int, int, int]:
    """
    Convert a continuous-ish environment state into a small discrete state.

    We use three simple features:

    1. current_gap_bin:
       How much worse the current tour is compared to the best tour.

    2. stagnation_bin:
       How many steps have passed without improving the best tour.

    3. progress_bin:
       How far we are in the episode.

    This keeps tabular Q-learning simple and interpretable.
    """
    if initial_length <= 0:
        raise ValueError("initial_length must be positive")

    current_gap = (state.current_length - state.best_length) / initial_length
    stagnation_ratio = state.steps_without_improvement / max_steps
    progress_ratio = state.step_count / max_steps

    current_gap_bin = int(np.digitize(current_gap, bins=[0.001, 0.01, 0.05, 0.10]))
    stagnation_bin = int(np.digitize(stagnation_ratio, bins=[0.10, 0.25, 0.50, 0.75]))
    progress_bin = int(np.digitize(progress_ratio, bins=[0.25, 0.50, 0.75]))

    return current_gap_bin, stagnation_bin, progress_bin


@dataclass
class QLearningAgent:
    """
    Simple tabular Q-learning agent for hyper-heuristic action selection.
    """

    n_actions: int
    learning_rate: float = 0.1
    discount_factor: float = 0.95
    epsilon: float = 1.0
    epsilon_min: float = 0.05
    epsilon_decay: float = 0.995
    seed: int | None = None
    q_table: dict[tuple[int, int, int], np.ndarray] = field(default_factory=dict)

    def __post_init__(self):
        if self.n_actions < 2:
            raise ValueError("n_actions must be at least 2")

        self.rng = np.random.default_rng(self.seed)

    def get_q_values(self, state_key: tuple[int, int, int]) -> np.ndarray:
        """
        Return Q-values for a discrete state.

        If the state has not been visited before, initialize its values to zero.
        """
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.n_actions, dtype=float)

        return self.q_table[state_key]

    def select_action(self, state_key: tuple[int, int, int], training: bool = True) -> int:
        """
        Select an action using epsilon-greedy exploration.
        """
        q_values = self.get_q_values(state_key)

        if training and self.rng.random() < self.epsilon:
            return int(self.rng.integers(0, self.n_actions))

        return int(np.argmax(q_values))

    def update(
        self,
        state_key: tuple[int, int, int],
        action: int,
        reward: float,
        next_state_key: tuple[int, int, int],
        done: bool,
    ) -> None:
        """
        Apply the standard Q-learning update rule.
        """
        if not 0 <= action < self.n_actions:
            raise ValueError("Invalid action")

        q_values = self.get_q_values(state_key)
        next_q_values = self.get_q_values(next_state_key)

        current_q = q_values[action]

        if done:
            target = reward
        else:
            target = reward + self.discount_factor * np.max(next_q_values)

        q_values[action] = current_q + self.learning_rate * (target - current_q)

    def decay_epsilon(self) -> None:
        """
        Decay epsilon after each training episode.
        """
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)