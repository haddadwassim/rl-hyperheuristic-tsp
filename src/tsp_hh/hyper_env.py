from dataclasses import dataclass

import numpy as np

from tsp_hh.tour import tour_length, validate_tour
from tsp_hh.heuristics import (
    first_improvement_two_opt,
    best_improvement_two_opt,
    random_swap_move,
    random_insertion_move,
    perturb_tour,
    create_initial_tour,
)


@dataclass
class HyperHeuristicState:
    """
    State returned by the hyper-heuristic environment.

    The RL agent will eventually observe a numerical representation
    of the current search situation.
    """
    current_length: float
    best_length: float
    steps_without_improvement: int
    step_count: int
    n_cities: int


class TSPHyperHeuristicEnv:
    """
    Lightweight environment for learning a hyper-heuristic for TSP.

    The environment does not construct a tour from scratch.
    Instead, it starts from an initial tour and lets an agent choose
    which low-level heuristic to apply at each step.

    Actions
    -------
    0: first-improvement 2-opt
    1: best-improvement 2-opt
    2: random swap
    3: random insertion
    4: perturbation
    """

    ACTION_NAMES = {
        0: "first_2opt",
        1: "best_2opt",
        2: "random_swap",
        3: "random_insertion",
        4: "perturbation",
    }

    def __init__(
        self,
        distance_matrix: np.ndarray,
        initial_method: str = "nearest_neighbor",
        max_steps: int = 100,
        seed: int | None = None,
        perturbation_moves: int = 3,
    ):
        self.distance_matrix = np.asarray(distance_matrix, dtype=float)
        self.n_cities = self.distance_matrix.shape[0]

        if self.distance_matrix.shape != (self.n_cities, self.n_cities):
            raise ValueError("distance_matrix must be square")

        if max_steps < 1:
            raise ValueError("max_steps must be at least 1")

        self.initial_method = initial_method
        self.max_steps = max_steps
        self.perturbation_moves = perturbation_moves

        self.rng = np.random.default_rng(seed)
        self.seed = seed

        self.current_tour: np.ndarray | None = None
        self.best_tour: np.ndarray | None = None
        self.current_length: float | None = None
        self.best_length: float | None = None
        self.step_count = 0
        self.steps_without_improvement = 0

        self.reset()

    def reset(self) -> HyperHeuristicState:
        """
        Reset the environment to a fresh initial tour.
        """
        initial_seed = int(self.rng.integers(0, 1_000_000_000))

        self.current_tour = create_initial_tour(
            self.distance_matrix,
            method=self.initial_method,
            seed=initial_seed,
        )

        if not validate_tour(self.current_tour, self.n_cities):
            raise RuntimeError("Generated invalid initial tour")

        self.current_length = tour_length(self.current_tour, self.distance_matrix)

        self.best_tour = self.current_tour.copy()
        self.best_length = self.current_length

        self.step_count = 0
        self.steps_without_improvement = 0

        return self._get_state()

    def step(self, action: int) -> tuple[HyperHeuristicState, float, bool, dict]:
        """
        Apply one low-level heuristic.

        Returns
        -------
        state:
            New environment state.
        reward:
            Positive if the best solution improved.
        done:
            True if max_steps has been reached.
        info:
            Extra diagnostic information.
        """
        if action not in self.ACTION_NAMES:
            raise ValueError(f"Unknown action: {action}")

        old_current_length = self.current_length
        old_best_length = self.best_length

        new_tour = self._apply_action(action)
        new_length = tour_length(new_tour, self.distance_matrix)

        self.current_tour = new_tour
        self.current_length = new_length

        best_improved = False

        if new_length < self.best_length:
            self.best_tour = new_tour.copy()
            self.best_length = new_length
            self.steps_without_improvement = 0
            best_improved = True
        else:
            self.steps_without_improvement += 1

        self.step_count += 1

        reward = old_best_length - self.best_length

        done = self.step_count >= self.max_steps

        info = {
            "action_name": self.ACTION_NAMES[action],
            "old_current_length": old_current_length,
            "new_current_length": new_length,
            "old_best_length": old_best_length,
            "new_best_length": self.best_length,
            "best_improved": best_improved,
            "steps_without_improvement": self.steps_without_improvement,
            "step_count": self.step_count,
        }

        return self._get_state(), float(reward), done, info

    def _apply_action(self, action: int) -> np.ndarray:
        """
        Apply the selected low-level heuristic to the current tour.
        """
        if self.current_tour is None:
            raise RuntimeError("Environment must be reset before stepping")

        if action == 0:
            new_tour, _ = first_improvement_two_opt(
                self.current_tour,
                self.distance_matrix,
            )
            return new_tour

        if action == 1:
            new_tour, _ = best_improvement_two_opt(
                self.current_tour,
                self.distance_matrix,
            )
            return new_tour

        if action == 2:
            move_seed = int(self.rng.integers(0, 1_000_000_000))
            return random_swap_move(self.current_tour, seed=move_seed)

        if action == 3:
            move_seed = int(self.rng.integers(0, 1_000_000_000))
            return random_insertion_move(self.current_tour, seed=move_seed)

        if action == 4:
            move_seed = int(self.rng.integers(0, 1_000_000_000))
            return perturb_tour(
                self.current_tour,
                n_moves=self.perturbation_moves,
                seed=move_seed,
            )

        raise ValueError(f"Unknown action: {action}")

    def _get_state(self) -> HyperHeuristicState:
        """
        Return a compact state object.
        """
        return HyperHeuristicState(
            current_length=float(self.current_length),
            best_length=float(self.best_length),
            steps_without_improvement=int(self.steps_without_improvement),
            step_count=int(self.step_count),
            n_cities=int(self.n_cities),
        )

    def get_observation(self) -> np.ndarray:
        """
        Return a normalized numerical observation for RL.

        For now, this is intentionally simple.

        Features
        --------
        0: current_length / initial scale
        1: best_length / initial scale
        2: steps_without_improvement / max_steps
        3: step_count / max_steps
        4: n_cities / 1000
        """
        if self.current_length is None or self.best_length is None:
            raise RuntimeError("Environment must be reset before observation")

        scale = max(self.n_cities, 1)

        return np.array(
            [
                self.current_length / scale,
                self.best_length / scale,
                self.steps_without_improvement / self.max_steps,
                self.step_count / self.max_steps,
                self.n_cities / 1000.0,
            ],
            dtype=np.float32,
        )