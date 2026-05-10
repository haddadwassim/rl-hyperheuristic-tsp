import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from tsp_hh.instances import generate_euclidean_instance
from tsp_hh.hyper_env import TSPHyperHeuristicEnv


def run_random_hh_on_instance(
    n_cities: int,
    seed: int,
    max_steps: int,
    initial_method: str,
) -> dict:
    """
    Run a random hyper-heuristic on one TSP instance.

    At each step, the policy randomly selects one low-level heuristic action.
    """
    instance = generate_euclidean_instance(n_cities=n_cities, seed=seed)

    env = TSPHyperHeuristicEnv(
        distance_matrix=instance.distance_matrix,
        initial_method=initial_method,
        max_steps=max_steps,
        seed=seed,
    )

    rng = np.random.default_rng(seed)

    initial_length = env.best_length
    action_counts = {action_name: 0 for action_name in env.ACTION_NAMES.values()}

    start_time = time.perf_counter()

    done = False

    while not done:
        action = int(rng.integers(0, len(env.ACTION_NAMES)))

        _, reward, done, info = env.step(action)

        action_counts[info["action_name"]] += 1

    runtime_sec = time.perf_counter() - start_time

    final_best_length = env.best_length
    total_improvement = initial_length - final_best_length

    result = {
        "method": "random_hh",
        "n_cities": n_cities,
        "seed": seed,
        "initial_method": initial_method,
        "max_steps": max_steps,
        "initial_length": initial_length,
        "best_length": final_best_length,
        "improvement": total_improvement,
        "runtime_sec": runtime_sec,
    }

    for action_name, count in action_counts.items():
        result[f"count_{action_name}"] = count

    return result


def summarize_results(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate random hyper-heuristic results by instance size.
    """
    summary = (
        df.groupby(["n_cities", "initial_method", "max_steps"])
        .agg(
            mean_initial_length=("initial_length", "mean"),
            mean_best_length=("best_length", "mean"),
            std_best_length=("best_length", "std"),
            mean_improvement=("improvement", "mean"),
            mean_runtime_sec=("runtime_sec", "mean"),
        )
        .reset_index()
        .sort_values(["n_cities", "mean_best_length"])
    )

    return summary


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--n-cities",
        type=int,
        nargs="+",
        default=[20, 50, 100],
        help="List of TSP instance sizes.",
    )

    parser.add_argument(
        "--n-instances",
        type=int,
        default=10,
        help="Number of random instances per size.",
    )

    parser.add_argument(
        "--max-steps",
        type=int,
        default=50,
        help="Number of hyper-heuristic decisions per instance.",
    )

    parser.add_argument(
        "--initial-method",
        type=str,
        default="nearest_neighbor",
        choices=["nearest_neighbor", "random"],
        help="Initial tour construction method.",
    )

    parser.add_argument(
        "--out-dir",
        type=str,
        default="results/random_hh",
        help="Directory where CSV results are saved.",
    )

    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = []

    for n_cities in args.n_cities:
        for seed in range(args.n_instances):
            row = run_random_hh_on_instance(
                n_cities=n_cities,
                seed=seed,
                max_steps=args.max_steps,
                initial_method=args.initial_method,
            )

            all_results.append(row)

            print(
                f"Finished n_cities={n_cities}, seed={seed}, "
                f"best_length={row['best_length']:.3f}, "
                f"improvement={row['improvement']:.3f}"
            )

    df = pd.DataFrame(all_results)
    summary = summarize_results(df)

    raw_path = out_dir / "random_hh_raw.csv"
    summary_path = out_dir / "random_hh_summary.csv"

    df.to_csv(raw_path, index=False)
    summary.to_csv(summary_path, index=False)

    print("\nRaw results saved to:")
    print(raw_path)

    print("\nSummary saved to:")
    print(summary_path)

    print("\nSummary:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()