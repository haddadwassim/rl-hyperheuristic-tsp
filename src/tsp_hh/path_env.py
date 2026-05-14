import numpy as np
import gymnasium as gym
from gymnasium import spaces


class TSPPathEnv(gym.Env):
    """
    TSP path-construction environment.

    The agent constructs a tour city by city.

    Observation:
        Dict with:
        - current_city: current city index
        - visited: binary vector of visited cities
        - step_count: number of selected cities
        - target_next_city: next city in teacher path, or -1 if unavailable

    Action:
        Choose next city.

    Mask:
        Valid actions are unvisited cities.
        At the final step, the episode ends and the return to start is added.

    Reward:
        - distance cost for every selected city
        - teacher bonus if the chosen city matches the optimal next city
        - final bonus if full tour exactly matches teacher path
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        coords: np.ndarray,
        optimal_tour: list[int],
        start_city: int = 0,
        teacher_bonus: float = 1.0,
        final_optimal_bonus: float = 10.0,
        distance_scale: float | None = None,
        max_steps: int | None = None,
        expose_teacher: bool = True,
    ):
        super().__init__()

        self.coords = np.asarray(coords, dtype=float)
        self.n_cities = self.coords.shape[0]
        self.start_city = int(start_city)

        if self.coords.shape != (self.n_cities, 2):
            raise ValueError("coords must have shape (n_cities, 2)")

        if sorted(optimal_tour) != list(range(self.n_cities)):
            raise ValueError("optimal_tour must contain every city exactly once")

        if optimal_tour[0] != self.start_city:
            raise ValueError("optimal_tour must start with start_city")

        self.optimal_tour = list(map(int, optimal_tour))
        self.teacher_bonus = teacher_bonus
        self.final_optimal_bonus = final_optimal_bonus

        self.distance_matrix = self._compute_distance_matrix(self.coords)

        if distance_scale is None:
            distance_scale = max(float(np.max(self.distance_matrix)), 1e-12)

        self.distance_scale = float(distance_scale)
        self.max_steps = max_steps or self.n_cities - 1

        self.expose_teacher = expose_teacher

        self.action_space = spaces.Discrete(self.n_cities)

        self.observation_space = spaces.Dict(
            {
                "current_city": spaces.Box(
                    low=0,
                    high=self.n_cities - 1,
                    shape=(1,),
                    dtype=np.int32,
                ),
                "current_city_onehot": spaces.Box(
                    low=0,
                    high=1,
                    shape=(self.n_cities,),
                    dtype=np.float32,
                ),
                "visited": spaces.Box(
                    low=0,
                    high=1,
                    shape=(self.n_cities,),
                    dtype=np.float32,
                ),
                "step_count": spaces.Box(
                    low=0,
                    high=self.n_cities,
                    shape=(1,),
                    dtype=np.float32,
                ),
                "target_next_city": spaces.Box(
                    low=-1,
                    high=self.n_cities - 1,
                    shape=(1,),
                    dtype=np.int32,
                ),
                "target_next_city_onehot": spaces.Box(
                    low=0,
                    high=1,
                    shape=(self.n_cities,),
                    dtype=np.float32,
                ),
                "coords": spaces.Box(
                    low=0,
                    high=1,
                    shape=(self.n_cities, 2),
                    dtype=np.float32,
                ),
                "distance_from_current": spaces.Box(
                    low=0,
                    high=np.inf,
                    shape=(self.n_cities,),
                    dtype=np.float32,
                ),
                "distance_to_start": spaces.Box(
                    low=0,
                    high=np.inf,
                    shape=(self.n_cities,),
                    dtype=np.float32,
                ),
            }
        )

        self.current_city = None
        self.visited = None
        self.path = None
        self.step_count = None
        self.total_distance = None

    def reset(self, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)

        self.current_city = self.start_city
        self.visited = np.zeros(self.n_cities, dtype=np.int8)
        self.visited[self.start_city] = 1

        self.path = [self.start_city]
        self.step_count = 0
        self.total_distance = 0.0

        return self._get_obs(), self._get_info()

    def step(self, action: int):
        action = int(action)

        terminated = False
        truncated = False

        valid_mask = self.action_masks()

        if not valid_mask[action]:
            # This should not happen with MaskablePPO, but keep safety.
            reward = -10.0
            terminated = True
            return self._get_obs(), reward, terminated, truncated, self._get_info()

        old_city = self.current_city
        new_city = action

        # Important: compute teacher target BEFORE updating the path.
        target = self._target_next_city_before_action()

        distance = self.distance_matrix[old_city, new_city]
        self.total_distance += distance

        self.current_city = new_city
        self.visited[new_city] = 1
        self.path.append(new_city)
        self.step_count += 1

        reward = -distance / self.distance_scale

        if new_city == target:
            reward += self.teacher_bonus

        if self.step_count >= self.max_steps or np.all(self.visited == 1):
            # Add return-to-start cost.
            return_distance = self.distance_matrix[self.current_city, self.start_city]
            self.total_distance += return_distance
            reward += -return_distance / self.distance_scale

            terminated = True

            if self.path == self.optimal_tour:
                reward += self.final_optimal_bonus

        return self._get_obs(), float(reward), terminated, truncated, self._get_info()

    def action_masks(self) -> np.ndarray:
        """
        Required by sb3-contrib MaskablePPO.

        True = valid action
        False = invalid action
        """
        if self.visited is None:
            mask = np.ones(self.n_cities, dtype=bool)
            mask[self.start_city] = False
            return mask

        mask = self.visited == 0
        return mask.astype(bool)

    def _target_next_city_before_action(self) -> int:
        """
        Return the teacher's next city for the current step.
        """
        next_index = len(self.path)

        if next_index >= len(self.optimal_tour):
            return -1

        return int(self.optimal_tour[next_index])

    def _get_obs(self) -> dict:
        current_onehot = np.zeros(self.n_cities, dtype=np.float32)
        current_onehot[self.current_city] = 1.0

        true_target = self._target_next_city_before_action()
        target = true_target if self.expose_teacher else -1

        target_onehot = np.zeros(self.n_cities, dtype=np.float32)

        if self.expose_teacher and target >= 0:
            target_onehot[target] = 1.0

        if target >= 0:
            target_onehot[target] = 1.0

        coords_min = np.min(self.coords, axis=0)
        coords_max = np.max(self.coords, axis=0)
        coords_range = np.maximum(coords_max - coords_min, 1e-12)
        coords_norm = (self.coords - coords_min) / coords_range

        distance_from_current = (
            self.distance_matrix[self.current_city] / self.distance_scale
        ).astype(np.float32)

        distance_to_start = (
            self.distance_matrix[:, self.start_city] / self.distance_scale
        ).astype(np.float32)

        return {
            "current_city": np.array([self.current_city], dtype=np.int32),
            "current_city_onehot": current_onehot,
            "visited": self.visited.astype(np.float32),
            "step_count": np.array(
                [self.step_count / max(self.n_cities - 1, 1)],
                dtype=np.float32,
            ),
            "target_next_city": np.array([target], dtype=np.int32),
            "target_next_city_onehot": target_onehot,
            "coords": coords_norm.astype(np.float32),
            "distance_from_current": distance_from_current,
            "distance_to_start": distance_to_start,
        }

    def _get_info(self) -> dict:
        return {
            "path": list(self.path),
            "total_distance": float(self.total_distance),
            "is_optimal_path": self.path == self.optimal_tour,
            "valid_actions": np.where(self.action_masks())[0].tolist(),
        }

    @staticmethod
    def _compute_distance_matrix(coords: np.ndarray) -> np.ndarray:
        diff = coords[:, None, :] - coords[None, :, :]
        return np.linalg.norm(diff, axis=-1)