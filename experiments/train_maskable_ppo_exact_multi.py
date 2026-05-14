import argparse
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.evaluation import evaluate_policy
from sb3_contrib.common.maskable.utils import get_action_masks
from stable_baselines3.common.monitor import Monitor

from tsp_hh.instances import generate_euclidean_instance
from tsp_hh.exact_tsp import held_karp_optimal_tour
from tsp_hh.path_env import TSPPathEnv


class MultiInstanceTSPPathEnv(gym.Env):
    """
    Multi-instance wrapper for TSPPathEnv.

    At every reset, one precomputed exact TSP instance is sampled.
    The underlying environment still exposes the teacher target.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        instances: list[dict],
        teacher_bonus: float = 3.0,
        final_optimal_bonus: float = 75.0,
        sample_mode: str = "random",
        expose_teacher: bool = False,
    ):
        super().__init__()

        if not instances:
            raise ValueError("instances must not be empty")

        self.instances = instances
        self.teacher_bonus = teacher_bonus
        self.final_optimal_bonus = final_optimal_bonus
        self.sample_mode = sample_mode
        self.expose_teacher = expose_teacher

        self.current_index = -1
        self.reset_counter = 0
        self.env: TSPPathEnv | None = None

        n_cities = instances[0]["coords"].shape[0]

        for item in instances:
            if item["coords"].shape[0] != n_cities:
                raise ValueError("All instances must have the same number of cities for this first version")

        # Same spaces as one TSPPathEnv.
        dummy = self._make_single_env(0)
        self.observation_space = dummy.observation_space
        self.action_space = dummy.action_space

    def _make_single_env(self, index: int) -> TSPPathEnv:
        item = self.instances[index]

        return TSPPathEnv(
            coords=item["coords"],
            optimal_tour=item["optimal_tour"],
            start_city=0,
            teacher_bonus=self.teacher_bonus,
            final_optimal_bonus=self.final_optimal_bonus,
            expose_teacher=self.expose_teacher,
        )

    def reset(self, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)

        if self.sample_mode == "round_robin":
            self.current_index = self.reset_counter % len(self.instances)
        elif self.sample_mode == "random":
            self.current_index = int(self.np_random.integers(0, len(self.instances)))
        else:
            raise ValueError(f"Unknown sample_mode: {self.sample_mode}")

        self.reset_counter += 1

        self.env = self._make_single_env(self.current_index)
        obs, info = self.env.reset(seed=seed)

        info["instance_index"] = self.current_index
        info["instance_seed"] = self.instances[self.current_index]["seed"]
        info["optimal_length"] = self.instances[self.current_index]["optimal_length"]

        return obs, info

    def step(self, action: int):
        if self.env is None:
            raise RuntimeError("Environment must be reset before step")

        obs, reward, terminated, truncated, info = self.env.step(action)

        info["instance_index"] = self.current_index
        info["instance_seed"] = self.instances[self.current_index]["seed"]
        info["optimal_length"] = self.instances[self.current_index]["optimal_length"]

        return obs, reward, terminated, truncated, info

    def action_masks(self):
        if self.env is None:
            mask = np.ones(self.action_space.n, dtype=bool)
            mask[0] = False
            return mask

        return self.env.action_masks()


def build_exact_instances(
    n_cities: int,
    seeds: list[int],
) -> list[dict]:
    instances = []

    for seed in seeds:
        instance = generate_euclidean_instance(
            n_cities=n_cities,
            seed=seed,
        )

        optimal_tour, optimal_length = held_karp_optimal_tour(
            instance.distance_matrix,
            start_city=0,
        )

        instances.append(
            {
                "seed": seed,
                "coords": instance.coords,
                "distance_matrix": instance.distance_matrix,
                "optimal_tour": optimal_tour,
                "optimal_length": optimal_length,
            }
        )

        print(
            f"Built instance seed={seed} | "
            f"optimal_length={optimal_length:.6f} | "
            f"optimal_tour={optimal_tour}"
        )

    return instances


def make_env(
    n_cities: int,
    seeds: list[int],
    teacher_bonus: float,
    final_optimal_bonus: float,
    sample_mode: str,
    expose_teacher: bool = False,
):
    instances = build_exact_instances(
        n_cities=n_cities,
        seeds=seeds,
    )

    env = MultiInstanceTSPPathEnv(
        instances=instances,
        teacher_bonus=teacher_bonus,
        final_optimal_bonus=final_optimal_bonus,
        sample_mode=sample_mode,
        expose_teacher=expose_teacher,
    )

    return Monitor(env), instances


def run_one_episode(model, env, deterministic: bool = True):
    obs, info = env.reset()

    done = False
    total_reward = 0.0

    while not done:
        action_masks = get_action_masks(env)

        action, _ = model.predict(
            obs,
            deterministic=deterministic,
            action_masks=action_masks,
        )

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated

    return info, total_reward


def evaluate_on_instances(
    model,
    instances: list[dict],
    n_eval_per_instance: int,
    teacher_bonus: float,
    final_optimal_bonus: float,
):
    rows = []

    for idx, item in enumerate(instances):
        single_env = TSPPathEnv(
            coords=item["coords"],
            optimal_tour=item["optimal_tour"],
            start_city=0,
            teacher_bonus=teacher_bonus,
            final_optimal_bonus=final_optimal_bonus,
            expose_teacher=False,  # Don't expose teacher during evaluation
        )

        env = Monitor(single_env)

        success = 0
        best_gap = float("inf")
        distances = []

        for _ in range(n_eval_per_instance):
            info, total_reward = run_one_episode(model, env, deterministic=True)

            distance = info["total_distance"]
            gap = (distance - item["optimal_length"]) / item["optimal_length"] * 100.0

            distances.append(distance)
            best_gap = min(best_gap, gap)

            if info["path"] == item["optimal_tour"]:
                success += 1

        mean_distance = float(np.mean(distances))
        mean_gap = (mean_distance - item["optimal_length"]) / item["optimal_length"] * 100.0

        rows.append(
            {
                "instance_index": idx,
                "seed": item["seed"],
                "optimal_length": item["optimal_length"],
                "success": success,
                "n_eval": n_eval_per_instance,
                "success_rate": success / n_eval_per_instance,
                "mean_distance": mean_distance,
                "mean_gap_percent": mean_gap,
                "best_gap_percent": best_gap,
            }
        )

    return rows


def parse_seed_list(seed_text: str) -> list[int]:
    return [int(x.strip()) for x in seed_text.split(",") if x.strip()]


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--n-cities", type=int, default=12)
    parser.add_argument("--train-seeds", type=str, default="42,43,44,45,46")
    parser.add_argument("--test-seeds", type=str, default="142,143,144")
    parser.add_argument("--timesteps", type=int, default=300_000)

    parser.add_argument("--teacher-bonus", type=float, default=3.0)
    parser.add_argument("--final-optimal-bonus", type=float, default=75.0)

    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--n-steps", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--ent-coef", type=float, default=0.01)

    parser.add_argument("--sample-mode", type=str, default="random", choices=["random", "round_robin"])
    parser.add_argument("--eval-per-instance", type=int, default=10)
    parser.add_argument("--model-seed", type=int, default=42)

    parser.add_argument(
        "--out-dir",
        type=str,
        default="results/maskable_ppo_exact_multi",
    )

    args = parser.parse_args()

    train_seeds = parse_seed_list(args.train_seeds)
    test_seeds = parse_seed_list(args.test_seeds)

    out_dir = Path(args.out_dir) / f"n{args.n_cities}_train{len(train_seeds)}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\nBuilding training instances")
    train_env, train_instances = make_env(
        n_cities=args.n_cities,
        seeds=train_seeds,
        teacher_bonus=args.teacher_bonus,
        final_optimal_bonus=args.final_optimal_bonus,
        sample_mode=args.sample_mode,
    )

    print("\nBuilding unseen test instances")
    test_instances = build_exact_instances(
        n_cities=args.n_cities,
        seeds=test_seeds,
    )

    model = MaskablePPO(
        policy="MultiInputPolicy",
        env=train_env,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        gamma=args.gamma,
        ent_coef=args.ent_coef,
        verbose=1,
        seed=args.model_seed,
    )

    model.learn(total_timesteps=args.timesteps)

    model_path = out_dir / "maskable_ppo_exact_multi"
    model.save(model_path)
    print(f"Saved model to: {model_path}")

    mean_reward, std_reward = evaluate_policy(
        model,
        train_env,
        n_eval_episodes=50,
        deterministic=True,
        warn=False,
    )

    print(f"\nTrain-env mean reward: {mean_reward:.3f} +/- {std_reward:.3f}")

    print("\nEvaluating on training instances")
    train_rows = evaluate_on_instances(
        model=model,
        instances=train_instances,
        n_eval_per_instance=args.eval_per_instance,
        teacher_bonus=args.teacher_bonus,
        final_optimal_bonus=args.final_optimal_bonus,
    )

    print("\nTraining instance results:")
    for row in train_rows:
        print(row)

    print("\nEvaluating on unseen test instances")
    test_rows = evaluate_on_instances(
        model=model,
        instances=test_instances,
        n_eval_per_instance=args.eval_per_instance,
        teacher_bonus=args.teacher_bonus,
        final_optimal_bonus=args.final_optimal_bonus,
    )

    print("\nUnseen test instance results:")
    for row in test_rows:
        print(row)

    import pandas as pd

    train_df = pd.DataFrame(train_rows)
    test_df = pd.DataFrame(test_rows)

    train_df.to_csv(out_dir / "train_eval.csv", index=False)
    test_df.to_csv(out_dir / "test_eval.csv", index=False)

    print(f"\nSaved train eval to: {out_dir / 'train_eval.csv'}")
    print(f"Saved test eval to: {out_dir / 'test_eval.csv'}")


if __name__ == "__main__":
    main()