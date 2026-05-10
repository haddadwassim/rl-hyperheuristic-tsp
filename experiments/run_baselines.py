import argparse
import time
from pathlib import Path

import pandas as pd

from tsp_hh.instances import generate_euclidean_instance
from tsp_hh.tour import tour_length, random_tour
from tsp_hh.heuristics import (
    nearest_neighbor_tour,
    repeated_two_opt,
)


def run_single_instance(
    n_cities: int,
    seed: int,
    two_opt_max_iterations: int,
) -> list[dict]:
    """
    Run baseline methods on one random Euclidean TSP instance.
    """
    instance = generate_euclidean_instance(n_cities=n_cities, seed=seed)
    distance_matrix = instance.distance_matrix

    results = []

    # ------------------------------------------------------------
    # Baseline 1: Random tour
    # ------------------------------------------------------------
    start_time = time.perf_counter()

    tour = random_tour(n_cities=n_cities, seed=seed)
    length = tour_length(tour, distance_matrix)

    elapsed = time.perf_counter() - start_time

    results.append({
        "method": "random",
        "n_cities": n_cities,
        "seed": seed,
        "tour_length": length,
        "improvement": 0.0,
        "two_opt_iterations": 0,
        "runtime_sec": elapsed,
    })

    # ------------------------------------------------------------
    # Baseline 2: Nearest neighbor
    # ------------------------------------------------------------
    start_time = time.perf_counter()

    tour = nearest_neighbor_tour(distance_matrix, start_city=0)
    length = tour_length(tour, distance_matrix)

    elapsed = time.perf_counter() - start_time

    results.append({
        "method": "nearest_neighbor",
        "n_cities": n_cities,
        "seed": seed,
        "tour_length": length,
        "improvement": 0.0,
        "two_opt_iterations": 0,
        "runtime_sec": elapsed,
    })

    # ------------------------------------------------------------
    # Baseline 3: Random tour + repeated 2-opt
    # ------------------------------------------------------------
    start_time = time.perf_counter()

    initial_tour = random_tour(n_cities=n_cities, seed=seed)
    initial_length = tour_length(initial_tour, distance_matrix)

    improved_tour, improvement, n_iterations = repeated_two_opt(
        initial_tour,
        distance_matrix,
        max_iterations=two_opt_max_iterations,
        use_best_improvement=False,
    )

    final_length = tour_length(improved_tour, distance_matrix)
    elapsed = time.perf_counter() - start_time

    results.append({
        "method": "random_2opt",
        "n_cities": n_cities,
        "seed": seed,
        "tour_length": final_length,
        "improvement": initial_length - final_length,
        "two_opt_iterations": n_iterations,
        "runtime_sec": elapsed,
    })

    # ------------------------------------------------------------
    # Baseline 4: Nearest neighbor + repeated 2-opt
    # ------------------------------------------------------------
    start_time = time.perf_counter()

    initial_tour = nearest_neighbor_tour(distance_matrix, start_city=0)
    initial_length = tour_length(initial_tour, distance_matrix)

    improved_tour, improvement, n_iterations = repeated_two_opt(
        initial_tour,
        distance_matrix,
        max_iterations=two_opt_max_iterations,
        use_best_improvement=False,
    )

    final_length = tour_length(improved_tour, distance_matrix)
    elapsed = time.perf_counter() - start_time

    results.append({
        "method": "nearest_neighbor_2opt",
        "n_cities": n_cities,
        "seed": seed,
        "tour_length": final_length,
        "improvement": initial_length - final_length,
        "two_opt_iterations": n_iterations,
        "runtime_sec": elapsed,
    })

    return results


def summarize_results(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate results by method and number of cities.
    """
    summary = (
        df.groupby(["n_cities", "method"])
        .agg(
            mean_length=("tour_length", "mean"),
            std_length=("tour_length", "std"),
            mean_runtime_sec=("runtime_sec", "mean"),
            mean_two_opt_iterations=("two_opt_iterations", "mean"),
        )
        .reset_index()
        .sort_values(["n_cities", "mean_length"])
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
        "--two-opt-max-iterations",
        type=int,
        default=200,
        help="Maximum number of repeated 2-opt improving moves.",
    )

    parser.add_argument(
        "--out-dir",
        type=str,
        default="results/baselines",
        help="Directory where CSV results are saved.",
    )

    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = []

    for n_cities in args.n_cities:
        for seed in range(args.n_instances):
            rows = run_single_instance(
                n_cities=n_cities,
                seed=seed,
                two_opt_max_iterations=args.two_opt_max_iterations,
            )
            all_results.extend(rows)

            print(f"Finished n_cities={n_cities}, seed={seed}")

    df = pd.DataFrame(all_results)
    summary = summarize_results(df)

    raw_path = out_dir / "baseline_raw.csv"
    summary_path = out_dir / "baseline_summary.csv"

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