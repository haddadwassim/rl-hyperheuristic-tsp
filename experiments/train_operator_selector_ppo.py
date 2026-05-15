import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from tsp_hh.operator_selector_env import (
    TSPInstanceData,
    TSPLIB_OPTIMA,
    TSPOperatorSelectorEnv,
)
from tsp_hh.tsplib_loader import load_tsplib_instance
from tsp_hh.heuristics_v3 import construct_tour_v3, two_opt_local_search_limited
from tsp_hh.tour import tour_length


def load_instances(data_dir: Path, names: list[str]) -> list[TSPInstanceData]:
    instances = []

    for name in names:
        path = data_dir / f"{name}.tsp"

        if not path.exists():
            raise FileNotFoundError(path)

        if name not in TSPLIB_OPTIMA:
            raise ValueError(f"No known optimum for {name}")

        inst = load_tsplib_instance(path)

        instances.append(
            TSPInstanceData(
                name=name,
                distance_matrix=inst.distance_matrix,
                optimum=TSPLIB_OPTIMA[name],
            )
        )

    return instances


def make_env(args, train_instances: list[TSPInstanceData]):
    env = TSPOperatorSelectorEnv(
        instances=train_instances,
        max_steps=args.max_steps,
        two_opt_iterations=args.two_opt_iterations,
        max_trials=args.max_trials,
        perturb_swaps=args.perturb_swaps,
        step_penalty=args.step_penalty,
        no_improve_penalty=args.no_improve_penalty,
        perturb_penalty=args.perturb_penalty,
        start_mode=args.start_mode,
        seed=args.seed,
    )

    return Monitor(env)


def evaluate_ppo(
    model: PPO,
    instances: list[TSPInstanceData],
    args,
) -> pd.DataFrame:
    rows = []

    for inst in instances:
        env = TSPOperatorSelectorEnv(
            instances=[inst],
            max_steps=args.max_steps,
            two_opt_iterations=args.two_opt_iterations,
            max_trials=args.max_trials,
            perturb_swaps=args.perturb_swaps,
            step_penalty=args.step_penalty,
            no_improve_penalty=args.no_improve_penalty,
            perturb_penalty=args.perturb_penalty,
            start_mode="nearest_neighbor",
            seed=args.seed,
        )

        obs, info = env.reset(seed=args.seed)

        terminated = False
        truncated = False

        total_reward = 0.0
        t0 = time.perf_counter()

        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))
            total_reward += float(reward)

        runtime_sec = time.perf_counter() - t0

        rows.append(
            {
                "instance": inst.name,
                "method": "ppo_operator_selector",
                "known_optimum": inst.optimum,
                "tour_length": info.get("best_length", info["current_length"]),
                "gap_to_optimum_percent": info["gap_percent"],
                "runtime_ms": runtime_sec * 1000.0,
                "total_reward": total_reward,
                "steps": info["step"],
                "count_stop": info.get("count_stop", 0),
                "count_2opt_limited": info.get("count_2opt_limited", 0),
                "count_relocate_best_of_k": info.get("count_relocate_best_of_k", 0),
                "count_swap_best_of_k": info.get("count_swap_best_of_k", 0),
                "count_perturb_then_2opt": info.get("count_perturb_then_2opt", 0),
            }
        )

    return pd.DataFrame(rows)


def evaluate_baselines(
    instances: list[TSPInstanceData],
    args,
) -> pd.DataFrame:
    rows = []

    for inst in instances:
        dm = inst.distance_matrix

        # NN only
        t0 = time.perf_counter()

        tour = construct_tour_v3(
            distance_matrix=dm,
            method="nearest_neighbor",
            seed=args.seed,
            n_starts=1,
        )
        tour = list(map(int, tour))
        length = tour_length(np.asarray(tour, dtype=int), dm)
        runtime_ms = (time.perf_counter() - t0) * 1000.0

        rows.append(
            {
                "instance": inst.name,
                "method": "nearest_neighbor",
                "known_optimum": inst.optimum,
                "tour_length": length,
                "gap_to_optimum_percent": (length - inst.optimum) / inst.optimum * 100.0,
                "runtime_ms": runtime_ms,
            }
        )

        # NN + 2opt
        t0 = time.perf_counter()

        repaired, _, _ = two_opt_local_search_limited(
            np.asarray(tour, dtype=int),
            dm,
            max_iterations=args.two_opt_iterations,
        )
        repaired = list(map(int, repaired))
        length = tour_length(np.asarray(repaired, dtype=int), dm)
        runtime_ms = (time.perf_counter() - t0) * 1000.0

        rows.append(
            {
                "instance": inst.name,
                "method": "nearest_neighbor_2opt",
                "known_optimum": inst.optimum,
                "tour_length": length,
                "gap_to_optimum_percent": (length - inst.optimum) / inst.optimum * 100.0,
                "runtime_ms": runtime_ms,
            }
        )

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data-dir", type=str, default="data/tsplib")

    parser.add_argument(
        "--train-instances",
        nargs="+",
        default=["eil51", "berlin52", "st70"],
    )

    parser.add_argument(
        "--test-instances",
        nargs="+",
        default=["eil76", "kroA100", "ch130"],
    )

    parser.add_argument("--total-timesteps", type=int, default=100_000)

    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--two-opt-iterations", type=int, default=50)
    parser.add_argument("--max-trials", type=int, default=100)
    parser.add_argument("--perturb-swaps", type=int, default=3)

    parser.add_argument("--step-penalty", type=float, default=0.0005)
    parser.add_argument("--no-improve-penalty", type=float, default=0.001)
    parser.add_argument("--perturb-penalty", type=float, default=0.001)

    parser.add_argument(
        "--start-mode",
        type=str,
        default="mixed",
        choices=["nearest_neighbor", "nn_2opt", "mixed"],
    )

    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--n-steps", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--ent-coef", type=float, default=0.01)

    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--out-dir", type=str, default="results/operator_selector_ppo")

    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data_dir = Path(args.data_dir)

    train_instances = load_instances(data_dir, args.train_instances)
    test_instances = load_instances(data_dir, args.test_instances)

    env = make_env(args, train_instances)

    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        gamma=args.gamma,
        ent_coef=args.ent_coef,
        verbose=1,
        seed=args.seed,
        tensorboard_log=str(out_dir / "tb"),
        device="cpu",
    )

    model.learn(total_timesteps=args.total_timesteps)

    model_path = out_dir / "ppo_operator_selector"
    model.save(str(model_path))

    print(f"\nSaved model to: {model_path}")

    train_eval = evaluate_ppo(model, train_instances, args)
    test_eval = evaluate_ppo(model, test_instances, args)

    train_base = evaluate_baselines(train_instances, args)
    test_base = evaluate_baselines(test_instances, args)

    train_results = pd.concat([train_eval, train_base], ignore_index=True)
    test_results = pd.concat([test_eval, test_base], ignore_index=True)

    train_results.to_csv(out_dir / "train_eval.csv", index=False)
    test_results.to_csv(out_dir / "test_eval.csv", index=False)

    print("\nTraining instance results:")
    print(
        train_results.sort_values(["instance", "gap_to_optimum_percent"])
        .to_string(index=False)
    )

    print("\nTest instance results:")
    print(
        test_results.sort_values(["instance", "gap_to_optimum_percent"])
        .to_string(index=False)
    )

    print(f"\nSaved results to: {out_dir}")


if __name__ == "__main__":
    main()