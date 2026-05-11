import argparse
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd

from tsp_hh.tsplib_loader import list_tsplib_files, load_tsplib_instance
from tsp_hh.tour import tour_length, validate_tour
from tsp_hh.heuristics_v3 import (
    construct_tour_v3,
    two_opt_local_search_limited,
    three_opt_local_search_sampled,
    random_two_opt_perturbation,
    city_swap_perturbation,
    insertion_perturbation,
)


TSPLIB_OPTIMA = {
    "eil51": 426.0,
    "berlin52": 7542.0,
    "st70": 675.0,
    "eil76": 538.0,
    "kroA100": 21282.0,
    "ch130": 6110.0,
}


def nn_2opt_baseline(
    distance_matrix: np.ndarray,
    seed: int,
    construction_starts: int,
    two_opt_iterations: int,
) -> tuple[np.ndarray, float]:
    tour = construct_tour_v3(
        distance_matrix,
        method="greedy",
        seed=seed,
        n_starts=construction_starts,
    )

    tour, _, _ = two_opt_local_search_limited(
        tour,
        distance_matrix,
        max_iterations=two_opt_iterations,
    )

    return tour, tour_length(tour, distance_matrix)


def nn_3opt_sampled_baseline(
    distance_matrix: np.ndarray,
    seed: int,
    construction_starts: int,
    two_opt_iterations: int,
    three_opt_samples: int,
    rounds: int,
) -> tuple[np.ndarray, float]:
    rng = np.random.default_rng(seed)

    tour = construct_tour_v3(
        distance_matrix,
        method="greedy",
        seed=seed,
        n_starts=construction_starts,
    )

    tour, _, _ = two_opt_local_search_limited(
        tour,
        distance_matrix,
        max_iterations=two_opt_iterations,
    )

    for _ in range(rounds):
        move_seed = int(rng.integers(0, 1_000_000_000))

        candidate, improvement, _ = three_opt_local_search_sampled(
            tour,
            distance_matrix,
            samples=three_opt_samples,
            seed=move_seed,
        )

        if improvement > 0:
            tour = candidate

            tour, _, _ = two_opt_local_search_limited(
                tour,
                distance_matrix,
                max_iterations=two_opt_iterations,
            )

    return tour, tour_length(tour, distance_matrix)


def iterated_local_search_baseline(
    distance_matrix: np.ndarray,
    seed: int,
    construction_starts: int,
    two_opt_iterations: int,
    iterations: int,
) -> tuple[np.ndarray, float]:
    rng = np.random.default_rng(seed)

    current = construct_tour_v3(
        distance_matrix,
        method="greedy",
        seed=seed,
        n_starts=construction_starts,
    )

    current, _, _ = two_opt_local_search_limited(
        current,
        distance_matrix,
        max_iterations=two_opt_iterations,
    )

    current_length = tour_length(current, distance_matrix)

    best = current.copy()
    best_length = current_length

    perturbations = [
        random_two_opt_perturbation,
        city_swap_perturbation,
        insertion_perturbation,
    ]

    for _ in range(iterations):
        perturb_fn = perturbations[int(rng.integers(0, len(perturbations)))]
        move_seed = int(rng.integers(0, 1_000_000_000))

        candidate = perturb_fn(current, seed=move_seed)

        candidate, _, _ = two_opt_local_search_limited(
            candidate,
            distance_matrix,
            max_iterations=two_opt_iterations,
        )

        candidate_length = tour_length(candidate, distance_matrix)

        # Accept if improved current.
        if candidate_length <= current_length:
            current = candidate
            current_length = candidate_length

        # Track global best.
        if candidate_length < best_length:
            best = candidate.copy()
            best_length = candidate_length
            current = candidate
            current_length = candidate_length

    return best, best_length


def simulated_annealing_baseline(
    distance_matrix: np.ndarray,
    seed: int,
    construction_starts: int,
    iterations: int,
    initial_temperature: float,
    cooling_rate: float,
) -> tuple[np.ndarray, float]:
    rng = np.random.default_rng(seed)

    current = construct_tour_v3(
        distance_matrix,
        method="greedy",
        seed=seed,
        n_starts=construction_starts,
    )

    current_length = tour_length(current, distance_matrix)

    best = current.copy()
    best_length = current_length

    temperature = initial_temperature

    perturbations = [
        random_two_opt_perturbation,
        city_swap_perturbation,
        insertion_perturbation,
    ]

    for _ in range(iterations):
        perturb_fn = perturbations[int(rng.integers(0, len(perturbations)))]
        move_seed = int(rng.integers(0, 1_000_000_000))

        candidate = perturb_fn(current, seed=move_seed)
        candidate_length = tour_length(candidate, distance_matrix)

        delta = candidate_length - current_length

        if delta <= 0:
            accept = True
        else:
            probability = math.exp(-delta / max(temperature, 1e-12))
            accept = rng.random() < probability

        if accept:
            current = candidate
            current_length = candidate_length

        if current_length < best_length:
            best = current.copy()
            best_length = current_length

        temperature *= cooling_rate

    return best, best_length


def run_method(
    instance_name: str,
    instance,
    optimum: float,
    method: str,
    run_seed: int,
    args,
) -> dict:
    start_time = time.perf_counter()

    if method == "nn_2opt":
        tour, length = nn_2opt_baseline(
            instance.distance_matrix,
            seed=run_seed,
            construction_starts=args.construction_starts,
            two_opt_iterations=args.two_opt_iterations,
        )

    elif method == "nn_3opt_sampled":
        tour, length = nn_3opt_sampled_baseline(
            instance.distance_matrix,
            seed=run_seed,
            construction_starts=args.construction_starts,
            two_opt_iterations=args.two_opt_iterations,
            three_opt_samples=args.three_opt_samples,
            rounds=args.three_opt_rounds,
        )

    elif method == "iterated_local_search":
        tour, length = iterated_local_search_baseline(
            instance.distance_matrix,
            seed=run_seed,
            construction_starts=args.construction_starts,
            two_opt_iterations=args.two_opt_iterations,
            iterations=args.ils_iterations,
        )

    elif method == "simulated_annealing":
        tour, length = simulated_annealing_baseline(
            instance.distance_matrix,
            seed=run_seed,
            construction_starts=args.construction_starts,
            iterations=args.sa_iterations,
            initial_temperature=args.sa_initial_temperature,
            cooling_rate=args.sa_cooling_rate,
        )

    else:
        raise ValueError(f"Unknown method: {method}")

    runtime_sec = time.perf_counter() - start_time

    if not validate_tour(tour, instance.coords.shape[0]):
        raise RuntimeError(f"{method} produced invalid tour on {instance_name}")

    gap = (length - optimum) / optimum * 100.0

    return {
        "instance": instance_name,
        "n_cities": instance.coords.shape[0],
        "run_seed": run_seed,
        "method": method,
        "optimum": optimum,
        "tour_length": length,
        "gap_to_optimum_percent": gap,
        "runtime_sec": runtime_sec,
    }


def summarize_results(raw: pd.DataFrame) -> pd.DataFrame:
    summary = (
        raw.groupby(["instance", "n_cities", "method"])
        .agg(
            mean_length=("tour_length", "mean"),
            std_length=("tour_length", "std"),
            best_length=("tour_length", "min"),
            worst_length=("tour_length", "max"),
            mean_gap_to_optimum_percent=("gap_to_optimum_percent", "mean"),
            best_gap_to_optimum_percent=("gap_to_optimum_percent", "min"),
            mean_runtime_sec=("runtime_sec", "mean"),
        )
        .reset_index()
    )

    best_by_instance = summary.groupby("instance")["mean_length"].transform("min")

    summary["gap_to_best_observed_percent"] = (
        (summary["mean_length"] - best_by_instance) / best_by_instance * 100.0
    )

    return summary.sort_values(["n_cities", "instance", "mean_length"])


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data-dir", type=str, default="data/tsplib")
    parser.add_argument("--out-dir", type=str, default="results/external_baselines_tsplib")
    parser.add_argument("--instances", type=str, nargs="*", default=None)
    parser.add_argument("--n-runs", type=int, default=10)
    parser.add_argument("--seed-offset", type=int, default=40_000)

    parser.add_argument(
        "--methods",
        type=str,
        nargs="+",
        default=[
            "nn_2opt",
            "nn_3opt_sampled",
            "iterated_local_search",
            "simulated_annealing",
        ],
    )

    parser.add_argument("--construction-starts", type=int, default=10)
    parser.add_argument("--two-opt-iterations", type=int, default=30)

    parser.add_argument("--three-opt-samples", type=int, default=30)
    parser.add_argument("--three-opt-rounds", type=int, default=10)

    parser.add_argument("--ils-iterations", type=int, default=100)

    parser.add_argument("--sa-iterations", type=int, default=2000)
    parser.add_argument("--sa-initial-temperature", type=float, default=1000.0)
    parser.add_argument("--sa-cooling-rate", type=float, default=0.995)

    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tsp_files = list_tsplib_files(args.data_dir)

    if args.instances:
        wanted = {x if x.endswith(".tsp") else f"{x}.tsp" for x in args.instances}
        tsp_files = [p for p in tsp_files if p.name in wanted]

    if not tsp_files:
        raise FileNotFoundError("No TSPLIB .tsp files found")

    all_rows = []

    for path in tsp_files:
        instance_name = path.stem

        if instance_name not in TSPLIB_OPTIMA:
            print(f"Skipping {instance_name}: no known optimum")
            continue

        instance = load_tsplib_instance(path)
        optimum = TSPLIB_OPTIMA[instance_name]

        print(f"\nRunning external baselines on {instance_name} (n={instance.coords.shape[0]})")

        for run_idx in range(args.n_runs):
            run_seed = args.seed_offset + run_idx

            for method in args.methods:
                row = run_method(
                    instance_name=instance_name,
                    instance=instance,
                    optimum=optimum,
                    method=method,
                    run_seed=run_seed,
                    args=args,
                )

                all_rows.append(row)

                print(
                    f"  run={run_idx + 1}/{args.n_runs} | "
                    f"{method:22s} | "
                    f"gap={row['gap_to_optimum_percent']:.3f}% | "
                    f"time={row['runtime_sec']:.3f}s"
                )

    raw = pd.DataFrame(all_rows)
    summary = summarize_results(raw)

    raw_path = out_dir / "external_baselines_tsplib_raw.csv"
    summary_path = out_dir / "external_baselines_tsplib_summary.csv"

    raw.to_csv(raw_path, index=False)
    summary.to_csv(summary_path, index=False)

    print("\nExternal baseline comparison finished.")
    print(f"Raw saved to: {raw_path}")
    print(f"Summary saved to: {summary_path}")

    print("\nSummary:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()