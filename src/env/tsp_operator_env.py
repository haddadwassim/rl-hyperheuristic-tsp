import numpy as np
import gymnasium as gym
from gymnasium import spaces

from src.tsp.generators import generate_instance
from src.tsp.tour import nearest_neighbor_tour, random_tour, tour_length
from src.tsp.metrics import tour_edge_statistics, count_crossing_edges
from src.operators.registry import apply_operator, num_actions, action_name


class TSPOperatorEnv(gym.Env):
    """
    Gymnasium environment for DRL-guided operator selection for TSP local search.

    The agent starts from an initial tour and repeatedly chooses one action:

        STOP, 2-opt, swap, relocate, perturb+2-opt

    The goal is to improve the tour while avoiding unnecessary search steps.
    """

    metadata = {"render_modes": []}

    def __init__(self, config: dict | None = None):
        super().__init__()

        if config is None:
            config = {}

        self.config = config

        tsp_config = config.get("tsp", {})
        env_config = config.get("environment", {})
        init_config = config.get("initial_solution", {})
        self.operator_config = config.get("operators", {})

        self.distribution = tsp_config.get("train_distribution", "uniform")
        self.num_nodes = tsp_config.get("train_num_nodes", 50)
        self.coordinate_range = tsp_config.get("coordinate_range", [0.0, 1.0])

        self.initial_solution_method = init_config.get("method", "nearest_neighbor")

        self.max_steps = env_config.get("max_steps", 100)

        self.stop_time_bonus_weight = env_config.get("stop_time_bonus_weight", 0.2)
        
        self.reward_scale = env_config.get("reward_scale", 10.0)
        self.stop_reward_weight = env_config.get("stop_reward_weight", 0.2)
        self.early_stop_no_improvement_penalty = env_config.get(
            "early_stop_no_improvement_penalty", 0.05
        )
        self.restore_best_on_failure = env_config.get("restore_best_on_failure", True)

        self.action_costs = config.get("action_costs", {})


        self.seed_value = config.get("project", {}).get("seed", None)
        self.rng = np.random.default_rng(self.seed_value)

        self.action_space = spaces.Discrete(num_actions())

        # Observation vector:
        # 0 current_length / initial_length
        # 1 best_length / initial_length
        # 2 relative_improvement
        # 3 step_fraction
        # 4 no_improvement_fraction
        # 5 last_improvement
        # 6 mean_edge / current_length
        # 7 std_edge / mean_edge
        # 8 max_edge / mean_edge
        # 9 crossing_ratio
        # 10 last_action_normalized
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(11,),
            dtype=np.float32,
        )

        self.instance = None
        self.current_tour = None
        self.current_length = None
        self.initial_length = None
        self.best_tour = None
        self.best_length = None

        self.current_step = 0
        self.steps_since_improvement = 0
        self.last_improvement = 0.0
        self.last_action = 0

    def reset(self, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)

        if seed is not None:
            self.rng = np.random.default_rng(seed)

        instance_seed = int(self.rng.integers(0, 1_000_000_000))

        self.instance = generate_instance(
            distribution=self.distribution,
            num_nodes=self.num_nodes,
            seed=instance_seed,
        )

        if self.initial_solution_method == "random":
            self.current_tour = random_tour(self.instance, rng=self.rng)
        else:
            self.current_tour = nearest_neighbor_tour(self.instance)

        self.current_length = tour_length(self.current_tour, self.instance)
        self.initial_length = self.current_length

        self.best_tour = self.current_tour.copy()
        self.best_length = self.current_length

        self.current_step = 0
        self.steps_since_improvement = 0
        self.last_improvement = 0.0
        self.last_action = 0

        observation = self._get_observation()

        info = self._get_info()
        return observation, info


    def step(self, action: int):
        action = int(action)

        if action < 0 or action >= num_actions():
            raise ValueError(f"Invalid action: {action}")

        action_label = action_name(action)

        terminated = False
        truncated = False

        old_best_length = self.best_length

        if action_label == "stop":
            terminated = True

            # Return the best solution found during the episode.
            self.current_tour = self.best_tour.copy()
            self.current_length = self.best_length

            final_improvement = (
                self.initial_length - self.best_length
            ) / self.initial_length

            if final_improvement <= 1e-12:
                reward = -self.early_stop_no_improvement_penalty
            else:
                remaining_budget_fraction = 1.0 - (
                    self.current_step / max(1, self.max_steps)
                )
                reward = (
                    self.stop_reward_weight * final_improvement
                    + self.stop_time_bonus_weight * remaining_budget_fraction
                )

            self.last_improvement = 0.0

        else:
            new_tour, new_length, improved, operator_info = apply_operator(
                action=action,
                tour=self.current_tour,
                instance=self.instance,
                config=self.operator_config,
                rng=self.rng,
            )

            # Update best archive if the action found a new best solution.
            if new_length < self.best_length:
                self.best_tour = new_tour.copy()
                self.best_length = new_length
                self.current_tour = new_tour
                self.current_length = new_length
                self.steps_since_improvement = 0
            else:
                self.steps_since_improvement += 1

                if self.restore_best_on_failure:
                    self.current_tour = self.best_tour.copy()
                    self.current_length = self.best_length
                else:
                    self.current_tour = new_tour
                    self.current_length = new_length

            best_improvement = (
                old_best_length - self.best_length
            ) / self.initial_length

            action_cost = self.action_costs.get(action_label, 0.01)

            reward = self.reward_scale * best_improvement - action_cost

            self.last_improvement = best_improvement

        self.current_step += 1
        self.last_action = action

        if self.current_step >= self.max_steps:
            truncated = True

        observation = self._get_observation()
        info = self._get_info()

        return observation, float(reward), terminated, truncated, info


    def _get_observation(self) -> np.ndarray:
        stats = tour_edge_statistics(self.current_tour, self.instance)

        mean_edge = stats["mean_edge_length"]
        std_edge = stats["std_edge_length"]
        max_edge = stats["max_edge_length"]

        current_ratio = self.current_length / self.initial_length
        best_ratio = self.best_length / self.initial_length
        relative_improvement = (self.initial_length - self.best_length) / self.initial_length

        step_fraction = self.current_step / max(1, self.max_steps)
        no_improvement_fraction = self.steps_since_improvement / max(1, self.max_steps)

        mean_edge_ratio = mean_edge / max(self.current_length, 1e-12)
        std_edge_ratio = std_edge / max(mean_edge, 1e-12)
        max_edge_ratio = max_edge / max(mean_edge, 1e-12)

        # O(n^2), but fine for initial experiments.
        crossings = count_crossing_edges(self.current_tour, self.instance)
        max_possible_edge_pairs = self.num_nodes * (self.num_nodes - 3) / 2
        crossing_ratio = crossings / max(1.0, max_possible_edge_pairs)

        last_action_normalized = self.last_action / max(1, num_actions() - 1)

        obs = np.array(
            [
                current_ratio,
                best_ratio,
                relative_improvement,
                step_fraction,
                no_improvement_fraction,
                self.last_improvement,
                mean_edge_ratio,
                std_edge_ratio,
                max_edge_ratio,
                crossing_ratio,
                last_action_normalized,
            ],
            dtype=np.float32,
        )

        return obs

    def _get_info(self) -> dict:
        return {
            "current_length": float(self.current_length),
            "initial_length": float(self.initial_length),
            "best_length": float(self.best_length),
            "relative_improvement": float(
                (self.initial_length - self.best_length) / self.initial_length
            ),
            "current_step": int(self.current_step),
            "steps_since_improvement": int(self.steps_since_improvement),
            "last_action": int(self.last_action),
            "last_action_name": action_name(self.last_action),
        }