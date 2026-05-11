from dataclasses import dataclass

import numpy as np

from tsp_hh.tour import tour_length, validate_tour
from tsp_hh.heuristics_v3 import (
    construct_tour_v3,
    apply_improvement_v3,
    apply_perturbation_v3,
)


@dataclass
class HyperHeuristicStateV3:
    current_length: float
    best_length: float
    initial_length: float
    last_improvement: float
    steps_without_improvement: int
    step_count: int
    n_cities: int
    last_category: int


class TSPHyperHeuristicEnvV3:
    """
    Hierarchical hyper-heuristic environment.

    The agent chooses a high-level search category:

    0 = construction / restart proposal
    1 = improvement / intensification
    2 = perturbation / diversification

    A simple rule-based selector chooses the concrete heuristic inside
    the selected category.
    """

    CATEGORY_NAMES = {
        0: "construction",
        1: "improvement",
        2: "perturbation",
    }

    CONSTRUCTION_METHODS = ["random", "nearest_neighbor", "greedy"]
    IMPROVEMENT_METHODS = ["none", "two_opt", "three_opt"]
    PERTURBATION_METHODS = ["random_2opt", "city_swap", "insertion"]

    def __init__(
        self,
        distance_matrix: np.ndarray,
        initial_method: str = "greedy",
        max_steps: int = 100,
        seed: int | None = None,
        construction_starts: int = 10,
        two_opt_iterations: int = 20,
        three_opt_samples: int = 30,
        accept_worse_perturbation: bool = True,
        reward_scale: str = "initial_length",
        construction_penalty: float = 0.0,
    ):
        self.distance_matrix = np.asarray(distance_matrix, dtype=float)
        self.n_cities = self.distance_matrix.shape[0]

        if self.distance_matrix.shape != (self.n_cities, self.n_cities):
            raise ValueError("distance_matrix must be square")

        if max_steps < 1:
            raise ValueError("max_steps must be at least 1")

        if initial_method not in self.CONSTRUCTION_METHODS:
            raise ValueError(f"Unknown initial_method: {initial_method}")

        self.initial_method = initial_method
        self.max_steps = max_steps
        self.construction_starts = construction_starts
        self.two_opt_iterations = two_opt_iterations
        self.three_opt_samples = three_opt_samples
        self.accept_worse_perturbation = accept_worse_perturbation
        self.reward_scale = reward_scale
        self.construction_penalty = construction_penalty

        self.rng = np.random.default_rng(seed)
        self.seed = seed

        self.current_tour: np.ndarray | None = None
        self.best_tour: np.ndarray | None = None

        self.current_length: float | None = None
        self.best_length: float | None = None
        self.initial_length: float | None = None

        self.last_improvement: float = 0.0
        self.steps_without_improvement: int = 0
        self.step_count: int = 0
        self.last_category: int = -1

        self.category_counts = {
            name: 0 for name in self.CATEGORY_NAMES.values()
        }

        self.heuristic_counts = {}

        self.reset()

    def reset(self) -> HyperHeuristicStateV3:
        seed = int(self.rng.integers(0, 1_000_000_000))

        self.current_tour = construct_tour_v3(
            self.distance_matrix,
            method=self.initial_method,
            seed=seed,
            n_starts=self.construction_starts,
        )

        if not validate_tour(self.current_tour, self.n_cities):
            raise RuntimeError("Generated invalid initial tour")

        self.current_length = tour_length(self.current_tour, self.distance_matrix)

        self.best_tour = self.current_tour.copy()
        self.best_length = self.current_length
        self.initial_length = self.current_length

        self.last_improvement = 0.0
        self.steps_without_improvement = 0
        self.step_count = 0
        self.last_category = -1

        self.category_counts = {
            name: 0 for name in self.CATEGORY_NAMES.values()
        }
        self.heuristic_counts = {}

        return self._get_state()

    def step(self, category: int) -> tuple[HyperHeuristicStateV3, float, bool, dict]:
        if category not in self.CATEGORY_NAMES:
            raise ValueError(f"Unknown category: {category}")

        old_current_length = self.current_length
        old_best_length = self.best_length

        category_name = self.CATEGORY_NAMES[category]
        self.category_counts[category_name] += 1

        method_name = None
        candidate_tour = self.current_tour.copy()

        if category == 0:
            method_name, candidate_tour = self._apply_construction_category()

        elif category == 1:
            method_name, candidate_tour = self._apply_improvement_category()

        elif category == 2:
            method_name, candidate_tour = self._apply_perturbation_category()

        candidate_length = tour_length(candidate_tour, self.distance_matrix)

        accepted_current = self._accept_candidate(
            category=category,
            candidate_length=candidate_length,
        )

        if accepted_current:
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
        self.last_category = category

        reward = self._compute_reward(
            raw_improvement=self.last_improvement,
            category=category,
        )

        done = self.step_count >= self.max_steps

        info = {
            "category": category,
            "category_name": category_name,
            "heuristic_name": method_name,
            "old_current_length": old_current_length,
            "candidate_length": candidate_length,
            "new_current_length": self.current_length,
            "old_best_length": old_best_length,
            "new_best_length": self.best_length,
            "raw_improvement": self.last_improvement,
            "reward": reward,
            "accepted_current": accepted_current,
            "best_improved": best_improved,
            "steps_without_improvement": self.steps_without_improvement,
            "step_count": self.step_count,
        }

        return self._get_state(), float(reward), done, info

    def _apply_construction_category(self) -> tuple[str, np.ndarray]:
        method = self._select_construction_method()

        seed = int(self.rng.integers(0, 1_000_000_000))

        candidate = construct_tour_v3(
            self.distance_matrix,
            method=method,
            seed=seed,
            n_starts=self.construction_starts,
        )

        # Restart proposal is repaired lightly by 2-opt.
        candidate, _, _ = apply_improvement_v3(
            candidate,
            self.distance_matrix,
            method="two_opt",
            seed=seed,
            two_opt_iterations=max(1, self.two_opt_iterations // 2),
            three_opt_samples=self.three_opt_samples,
        )

        self._count_heuristic(f"construction:{method}")
        return method, candidate

    def _apply_improvement_category(self) -> tuple[str, np.ndarray]:
        method = self._select_improvement_method()

        seed = int(self.rng.integers(0, 1_000_000_000))

        candidate, _, _ = apply_improvement_v3(
            self.current_tour,
            self.distance_matrix,
            method=method,
            seed=seed,
            two_opt_iterations=self.two_opt_iterations,
            three_opt_samples=self.three_opt_samples,
        )

        self._count_heuristic(f"improvement:{method}")
        return method, candidate

    def _apply_perturbation_category(self) -> tuple[str, np.ndarray]:
        method = self._select_perturbation_method()

        seed = int(self.rng.integers(0, 1_000_000_000))

        candidate = apply_perturbation_v3(
            self.current_tour,
            method=method,
            seed=seed,
        )

        self._count_heuristic(f"perturbation:{method}")
        return method, candidate

    def _select_construction_method(self) -> str:
        # Keep construction strong and simple for now.
        if self.steps_without_improvement > 30:
            return "greedy"

        return "nearest_neighbor"

    def _select_improvement_method(self) -> str:
        # Start with 2-opt. Use sampled 3-opt when the search stagnates.
        if self.steps_without_improvement < 10:
            return "two_opt"

        if self.steps_without_improvement < 30:
            return "three_opt"

        return "none"

    def _select_perturbation_method(self) -> str:
        # Cycle among perturbation operators depending on stagnation.
        if self.steps_without_improvement < 10:
            return "insertion"

        if self.steps_without_improvement < 25:
            return "random_2opt"

        return "city_swap"

    def _accept_candidate(self, category: int, candidate_length: float) -> bool:
        # Construction is a restart proposal: accept only if it improves current.
        if category == 0:
            return candidate_length <= self.current_length

        # Improvement actions should never worsen.
        if category == 1:
            return candidate_length <= self.current_length

        # Perturbation may worsen current to escape local minima if enabled.
        if category == 2:
            if self.accept_worse_perturbation:
                return True
            return candidate_length <= self.current_length

        return False

    def _compute_reward(self, raw_improvement: float, category: int) -> float:
        if self.reward_scale == "none":
            reward = raw_improvement
        elif self.reward_scale == "initial_length":
            reward = raw_improvement / max(self.initial_length, 1e-12)
        else:
            raise ValueError(f"Unknown reward_scale: {self.reward_scale}")

        if category == 0:
            reward -= self.construction_penalty

        return float(reward)

    def _count_heuristic(self, name: str) -> None:
        self.heuristic_counts[name] = self.heuristic_counts.get(name, 0) + 1

    def _get_state(self) -> HyperHeuristicStateV3:
        return HyperHeuristicStateV3(
            current_length=float(self.current_length),
            best_length=float(self.best_length),
            initial_length=float(self.initial_length),
            last_improvement=float(self.last_improvement),
            steps_without_improvement=int(self.steps_without_improvement),
            step_count=int(self.step_count),
            n_cities=int(self.n_cities),
            last_category=int(self.last_category),
        )

    def get_observation(self) -> np.ndarray:
        initial = max(self.initial_length, 1e-12)

        return np.array(
            [
                self.current_length / initial,
                self.best_length / initial,
                self.last_improvement / initial,
                self.steps_without_improvement / self.max_steps,
                self.step_count / self.max_steps,
                self.n_cities / 1000.0,
                self.last_category / 3.0,
            ],
            dtype=np.float32,
        )