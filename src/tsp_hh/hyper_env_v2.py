from dataclasses import dataclass

import numpy as np

from tsp_hh.tour import tour_length, validate_tour
from tsp_hh.heuristics import create_initial_tour, first_improvement_two_opt
from tsp_hh.bounded_heuristics import (
    random_two_opt_move,
    best_of_k_random_two_opt,
    best_of_k_random_swaps,
    best_of_k_random_insertions,
    perturb_then_best_of_k_two_opt,
)


@dataclass
class HyperHeuristicStateV2:
    current_length: float
    best_length: float
    last_improvement: float
    steps_without_improvement: int
    step_count: int
    n_cities: int


class TSPHyperHeuristicEnvV2:
    """
    Bounded hyper-heuristic environment for TSP.

    Unlike V1, this environment avoids giving the agent a dominant full
    best-improvement 2-opt action.

    Actions
    -------
    0: first-improvement 2-opt
    1: random 2-opt move
    2: best-of-k random 2-opt moves
    3: best-of-k random swap moves
    4: best-of-k random insertion moves
    5: perturbation + bounded 2-opt
    """

    ACTION_NAMES = {
        0: "first_2opt",
        1: "random_2opt",
        2: "bounded_2opt",
        3: "bounded_swap",
        4: "bounded_insertion",
        5: "perturb_bounded_2opt",
    }

    def __init__(
        self,
        distance_matrix: np.ndarray,
        initial_method: str = "nearest_neighbor",
        max_steps: int = 100,
        seed: int | None = None,
        k_neighbors: int = 10,
        perturbation_moves: int = 3,
        reward_scale: str = "initial_length",
        accept_worse_current: bool = False,
    ):
        self.distance_matrix = np.asarray(distance_matrix, dtype=float)
        self.n_cities = self.distance_matrix.shape[0]

        if self.distance_matrix.shape != (self.n_cities, self.n_cities):
            raise ValueError("distance_matrix must be square")

        if max_steps < 1:
            raise ValueError("max_steps must be at least 1")

        if k_neighbors < 1:
            raise ValueError("k_neighbors must be at least 1")

        self.initial_method = initial_method
        self.max_steps = max_steps
        self.k_neighbors = k_neighbors
        self.perturbation_moves = perturbation_moves
        self.reward_scale = reward_scale
        self.accept_worse_current = accept_worse_current

        self.rng = np.random.default_rng(seed)
        self.seed = seed

        self.current_tour: np.ndarray | None = None
        self.best_tour: np.ndarray | None = None
        self.current_length: float | None = None
        self.best_length: float | None = None
        self.initial_length: float | None = None
        self.last_improvement: float = 0.0

        self.step_count = 0
        self.steps_without_improvement = 0

        self.reset()

    def reset(self) -> HyperHeuristicStateV2:
        initial_seed = int(self.rng.integers(0, 1_000_000_000))

        self.current_tour = create_initial_tour(
            self.distance_matrix,
            method=self.initial_method,
            seed=initial_seed,
        )

        if not validate_tour(self.current_tour, self.n_cities):
            raise RuntimeError("Generated invalid initial tour")

        self.current_length = tour_length(self.current_tour, self.distance_matrix)
        self.initial_length = self.current_length

        self.best_tour = self.current_tour.copy()
        self.best_length = self.current_length

        self.last_improvement = 0.0
        self.step_count = 0
        self.steps_without_improvement = 0

        return self._get_state()

    def step(self, action: int) -> tuple[HyperHeuristicStateV2, float, bool, dict]:
        if action not in self.ACTION_NAMES:
            raise ValueError(f"Unknown action: {action}")

        old_current_length = self.current_length
        old_best_length = self.best_length

        candidate_tour = self._apply_action(action)
        candidate_length = tour_length(candidate_tour, self.distance_matrix)

        # By default, do not move current_tour to a worse solution.
        # This makes each action a bounded improvement attempt.
        if self.accept_worse_current or candidate_length <= self.current_length:
            self.current_tour = candidate_tour
            self.current_length = candidate_length

        best_improved = False

        if self.current_length < self.best_length:
            self.best_tour = self.current_tour.copy()
            self.best_length = self.current_length
            self.steps_without_improvement = 0
            best_improved = True
        else:
            self.steps_without_improvement += 1

        self.last_improvement = old_best_length - self.best_length
        self.step_count += 1

        reward = self._compute_reward(self.last_improvement)

        done = self.step_count >= self.max_steps

        info = {
            "action_name": self.ACTION_NAMES[action],
            "old_current_length": old_current_length,
            "candidate_length": candidate_length,
            "new_current_length": self.current_length,
            "old_best_length": old_best_length,
            "new_best_length": self.best_length,
            "raw_improvement": self.last_improvement,
            "reward": reward,
            "best_improved": best_improved,
            "steps_without_improvement": self.steps_without_improvement,
            "step_count": self.step_count,
        }

        return self._get_state(), float(reward), done, info

    def _apply_action(self, action: int) -> np.ndarray:
        if self.current_tour is None:
            raise RuntimeError("Environment must be reset before stepping")

        move_seed = int(self.rng.integers(0, 1_000_000_000))

        if action == 0:
            new_tour, _ = first_improvement_two_opt(
                self.current_tour,
                self.distance_matrix,
            )
            return new_tour

        if action == 1:
            return random_two_opt_move(
                self.current_tour,
                seed=move_seed,
            )

        if action == 2:
            new_tour, _ = best_of_k_random_two_opt(
                self.current_tour,
                self.distance_matrix,
                k=self.k_neighbors,
                seed=move_seed,
            )
            return new_tour

        if action == 3:
            new_tour, _ = best_of_k_random_swaps(
                self.current_tour,
                self.distance_matrix,
                k=self.k_neighbors,
                seed=move_seed,
            )
            return new_tour

        if action == 4:
            new_tour, _ = best_of_k_random_insertions(
                self.current_tour,
                self.distance_matrix,
                k=self.k_neighbors,
                seed=move_seed,
            )
            return new_tour

        if action == 5:
            new_tour, _ = perturb_then_best_of_k_two_opt(
                self.current_tour,
                self.distance_matrix,
                perturbation_moves=self.perturbation_moves,
                k=self.k_neighbors,
                seed=move_seed,
            )
            return new_tour

        raise ValueError(f"Unknown action: {action}")

    def _compute_reward(self, raw_improvement: float) -> float:
        if self.reward_scale == "none":
            return float(raw_improvement)

        if self.reward_scale == "initial_length":
            return float(raw_improvement / max(self.initial_length, 1e-12))

        raise ValueError(f"Unknown reward_scale: {self.reward_scale}")

    def _get_state(self) -> HyperHeuristicStateV2:
        return HyperHeuristicStateV2(
            current_length=float(self.current_length),
            best_length=float(self.best_length),
            last_improvement=float(self.last_improvement),
            steps_without_improvement=int(self.steps_without_improvement),
            step_count=int(self.step_count),
            n_cities=int(self.n_cities),
        )

    def get_observation(self) -> np.ndarray:
        if self.current_length is None or self.best_length is None:
            raise RuntimeError("Environment must be reset before observation")

        initial = max(self.initial_length, 1e-12)

        return np.array(
            [
                self.current_length / initial,
                self.best_length / initial,
                self.last_improvement / initial,
                self.steps_without_improvement / self.max_steps,
                self.step_count / self.max_steps,
                self.n_cities / 1000.0,
            ],
            dtype=np.float32,
        )