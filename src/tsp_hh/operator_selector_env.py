import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from tsp_hh.heuristics_v3 import construct_tour_v3, two_opt_local_search_limited
from tsp_hh.operators import apply_operator
from tsp_hh.tour import tour_length


TSPLIB_OPTIMA = {
    "eil51": 426.0,
    "berlin52": 7542.0,
    "st70": 675.0,
    "eil76": 538.0,
    "pr76": 108159.0,
    "rat99": 1211.0,
    "kroA100": 21282.0,
    "kroB100": 22141.0,
    "eil101": 629.0,
    "ch130": 6110.0,
}


@dataclass
class TSPInstanceData:
    name: str
    distance_matrix: np.ndarray
    optimum: float


class TSPOperatorSelectorEnv(gym.Env):
    """
    DRL environment for TSP heuristic/operator selection.

    Clean action space:
      0 = STOP
      1 = 2-opt limited
      2 = relocate best-of-k
      3 = swap best-of-k
      4 = perturb + 2-opt repair

    Internally mapped to tsp_hh.operators.apply_operator actions:
      env 0 -> op 0 STOP
      env 1 -> op 1 2opt_limited
      env 2 -> op 3 relocate_best_of_k
      env 3 -> op 4 swap_best_of_k
      env 4 -> op 5 perturb_then_2opt
    """

    metadata = {"render_modes": []}

    ENV_TO_OPERATOR_ACTION = {
        0: 0,
        1: 1,
        2: 3,
        3: 4,
        4: 5,
    }

    ACTION_NAMES = {
        0: "stop",
        1: "2opt_limited",
        2: "relocate_best_of_k",
        3: "swap_best_of_k",
        4: "perturb_then_2opt",
    }

    def __init__(
        self,
        instances: list[TSPInstanceData],
        max_steps: int = 10,
        two_opt_iterations: int = 50,
        max_trials: int = 100,
        perturb_swaps: int = 3,
        step_penalty: float = 0.0005,
        no_improve_penalty: float = 0.001,
        perturb_penalty: float = 0.001,
        start_mode: str = "mixed",
        seed: int = 42,
    ):
        super().__init__()

        self.instances = instances
        self.max_steps = max_steps
        self.two_opt_iterations = two_opt_iterations
        self.max_trials = max_trials
        self.perturb_swaps = perturb_swaps

        self.step_penalty = step_penalty
        self.no_improve_penalty = no_improve_penalty
        self.perturb_penalty = perturb_penalty

        self.start_mode = start_mode

        self.rng = np.random.default_rng(seed)

        self.action_space = spaces.Discrete(5)

        # Features:
        # n_norm, current_gap, length/initial_length, step/max_steps,
        # last_improvement_ratio, no_improve/max_steps, last_action_norm,
        # mean_edge, std_edge, max_edge, min_edge,
        # max_edge/mean_edge, std_edge/mean_edge
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(13,),
            dtype=np.float32,
        )

        self.instance = None
        self.distance_matrix = None
        self.optimum = None
        self.tour = None
        self.initial_length = None
        self.current_length = None
        self.current_step = None
        self.last_improvement = None
        self.no_improve_steps = None
        self.last_action = None
        self.action_counts = None
        self.runtime_sec = None

    def _edge_lengths(self, tour: list[int]) -> np.ndarray:
        arr = np.asarray(tour, dtype=int)
        nxt = np.roll(arr, -1)
        return self.distance_matrix[arr, nxt]

    def _get_obs(self) -> np.ndarray:
        n = len(self.tour)
        edges = self._edge_lengths(self.tour)

        mean_edge = float(edges.mean())
        std_edge = float(edges.std())
        max_edge = float(edges.max())
        min_edge = float(edges.min())

        current_gap = (self.current_length - self.optimum) / self.optimum

        obs = np.array(
            [
                n / 150.0,
                current_gap,
                self.current_length / max(self.initial_length, 1e-12),
                self.current_step / max(self.max_steps, 1),
                self.last_improvement / max(self.initial_length, 1e-12),
                self.no_improve_steps / max(self.max_steps, 1),
                self.last_action / 4.0,
                mean_edge,
                std_edge,
                max_edge,
                min_edge,
                max_edge / max(mean_edge, 1e-12),
                std_edge / max(mean_edge, 1e-12),
            ],
            dtype=np.float32,
        )

        return obs

    def _build_initial_tour(self, seed: int) -> list[int]:
        tour = construct_tour_v3(
            distance_matrix=self.distance_matrix,
            method="nearest_neighbor",
            seed=seed,
            n_starts=1,
        )
        tour = list(map(int, tour))

        if self.start_mode == "nearest_neighbor":
            return tour

        if self.start_mode == "nn_2opt":
            tour, _, _ = two_opt_local_search_limited(
                np.asarray(tour, dtype=int),
                self.distance_matrix,
                max_iterations=self.two_opt_iterations,
            )
            return list(map(int, tour))

        if self.start_mode == "mixed":
            # Half episodes start from NN, half from NN+2opt.
            if self.rng.random() < 0.5:
                return tour

            tour, _, _ = two_opt_local_search_limited(
                np.asarray(tour, dtype=int),
                self.distance_matrix,
                max_iterations=self.two_opt_iterations,
            )
            return list(map(int, tour))

        raise ValueError(f"Unknown start_mode: {self.start_mode}")

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        idx = int(self.rng.integers(0, len(self.instances)))
        self.instance = self.instances[idx]

        self.distance_matrix = self.instance.distance_matrix
        self.optimum = float(self.instance.optimum)

        local_seed = int(self.rng.integers(0, 1_000_000_000))

        self.tour = self._build_initial_tour(seed=local_seed)
        self.initial_length = tour_length(np.asarray(self.tour, dtype=int), self.distance_matrix)
        self.current_length = float(self.initial_length)

        self.best_tour = list(self.tour)
        self.best_length = float(self.current_length)

        self.current_step = 0
        self.last_improvement = 0.0
        self.no_improve_steps = 0
        self.last_action = 0
        self.runtime_sec = 0.0

        self.action_counts = {name: 0 for name in self.ACTION_NAMES.values()}

        info = {
            "instance": self.instance.name,
            "initial_length": self.initial_length,
            "initial_gap": (self.initial_length - self.optimum) / self.optimum,
        }

        return self._get_obs(), info

    def step(self, action: int):
        action = int(action)
        action_name = self.ACTION_NAMES[action]
        self.action_counts[action_name] += 1

        old_length = self.current_length
        old_gap = (old_length - self.optimum) / self.optimum

        terminated = False
        truncated = False

        if action == 0:
            current_gap = (self.current_length - self.optimum) / self.optimum
            reward = -current_gap
            terminated = True

            info = self._info(action_name=action_name, old_length=old_length)
            return self._get_obs(), float(reward), terminated, truncated, info

        op_action = self.ENV_TO_OPERATOR_ACTION[action]

        t0 = time.perf_counter()

        result = apply_operator(
            action=op_action,
            tour=self.tour,
            distance_matrix=self.distance_matrix,
            seed=int(self.rng.integers(0, 1_000_000_000)),
            two_opt_iterations=self.two_opt_iterations,
            max_trials=self.max_trials,
            perturb_swaps=self.perturb_swaps,
        )

        self.runtime_sec += time.perf_counter() - t0

        self.tour = result.tour
        self.current_length = float(result.length)

        if self.current_length < self.best_length:
            self.best_length = float(self.current_length)
            self.best_tour = list(self.tour)

        new_gap = (self.current_length - self.optimum) / self.optimum

        improvement_in_gap = old_gap - new_gap
        reward = improvement_in_gap

        reward -= self.step_penalty

        if action == 4:
            reward -= self.perturb_penalty

        if self.current_length >= old_length - 1e-12:
            reward -= self.no_improve_penalty
            self.no_improve_steps += 1
        else:
            self.no_improve_steps = 0

        self.last_improvement = old_length - self.current_length
        self.last_action = action

        self.current_step += 1

        if self.current_step >= self.max_steps:
            truncated = True
            reward += -new_gap

        info = self._info(action_name=action_name, old_length=old_length)

        return self._get_obs(), float(reward), terminated, truncated, info

    def _info(self, action_name: str, old_length: float) -> dict:
        current_gap = (self.current_length - self.optimum) / self.optimum
        best_gap = (self.best_length - self.optimum) / self.optimum

        return {
            "instance": self.instance.name,
            "action_name": action_name,
            "old_length": old_length,
            "current_length": self.current_length,
            "best_length": self.best_length,
            "optimum": self.optimum,
            "gap": best_gap,
            "gap_percent": best_gap * 100.0,
            "current_gap": current_gap,
            "current_gap_percent": current_gap * 100.0,
            "improvement": old_length - self.current_length,
            "step": self.current_step,
            "runtime_sec": self.runtime_sec,
            **{f"count_{k}": v for k, v in self.action_counts.items()},

        }