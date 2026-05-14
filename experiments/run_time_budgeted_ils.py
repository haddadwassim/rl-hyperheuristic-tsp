import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from tsp_hh.tsplib_loader import list_tsplib_files, load_tsplib_instance
from tsp_hh.tour import tour_length, validate_tour
from tsp_hh.heuristics_v3 import (
    construct_tour_v3,
    two_opt_local_search_limited,
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


def load_v3_time_budgets(v3_summary_path: Path) -> dict[str, float]:
    """
    Use the mean runtime of teacher_guided_q_learning_v3 as the ILS time budget
    for each TSPLIB instance.
    """
    if not v3_summary_path.exists():
        raise FileNotFoundError(f"V3 summary not found: {v3_summary_path}")

    summary = pd.read_csv(v3_summary_path)

    q = summary[summary["method"] == "teacher_guided_q_learning_v3"].copy()

    if q.empty:
        raise ValueError("No teacher_guided_q_learning_v3 rows found in V3 summary")

    budgets = {
        row["instance"]: float(row["mean_runtime_sec"])
        for _, row in q.iterrows()
    }

    return budgets


def iterated_local_search_time_budgeted(
    distance_matrix: np.ndarray,
    seed: int,
    construction_starts: int,
    two_opt_iterations: int,
    time_budget_sec: float,
) -> tuple[np.ndarray, float, int, float]:
    """
    Run ILS until the given time budget is reached.

    Pattern:
    greedy construction
    2-opt repair
    repeat:
        perturb
        2-opt repair
        accept if improved
    """
    rng = np.random.default_rng(seed)

    start_time = time.perf_counter()

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

    iterations = 0

    while True:
        elapsed = time.perf_counter() - start_time

        if elapsed >= time_budget_sec:
            break

        perturb_fn = perturbations[int(rng.integers(0, len(perturbations)))]
        move_seed = int(rng.integers(0, 1_000_000_000))

        candidate = perturb_fn(current, seed=move_seed)

        candidate, _, _ = two_opt_local_search_limited(
            candidate,
            distance_matrix,
            max_iterations=two_opt_iterations,
        )

        candidate_length = tour_length(candidate, distance_matrix)

        if candidate_length <= current_length:
            current = candidate
            current_length = candidate_length

        if candidate_length < best_length:
            best = candidate.copy()
            best_length = candidate_length
            current = candidate
            current_length = candidate_length

        iterations += 1

    runtime_sec = time.perf_counter() - start_time

    return best, best_length, iterations, runtime_sec


def run_one(
    instance_name: str,
    instance,
    optimum: float,
    run_seed: int,
    time_budget_sec: float,
    construction_starts: int,
    two_opt_iterations: int,
) -> dict:
    tour, length, iterations, runtime_sec = iterated_local_search_time_budgeted(
        distance_matrix=instance.distance_matrix,
        seed=run_seed,
        construction_starts=construction_starts,
        two_opt_iterations=two_opt_iterations,
        time_budget_sec=time_budget_sec,
    )

    if not validate_tour(tour, instance.coords.shape[0]):
        raise RuntimeError(f"Invalid tour for {instance_name}")

    gap = (length - optimum) / optimum * 100.0

    return {
        "instance": instance_name,
        "n_cities": instance.coords.shape[0],
        "run_seed": run_seed,
        "method": "ils_time_budgeted",
        "optimum": optimum,
        "time_budget_sec": time_budget_sec,
        "tour_length": length,
        "gap_to_optimum_percent": gap,
        "iterations_completed": iterations,
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
            mean_iterations=("iterations_completed", "mean"),
            mean_budget_sec=("time_budget_sec", "mean"),
            mean_runtime_sec=("runtime_sec", "mean"),
        )
        .reset_index()
    )

    return summary.sort_values(["n_cities", "instance", "mean_length"])


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data-dir", type=str, default="data/tsplib")
    parser.add_argument(
        "--v3-summary-path",
        type=str,
        default="results/tsplib_v3_teacher_medium/tsplib_v3_summary.csv",
        help="V3 summary file used to extract per-instance runtime budgets.",
    )
    parser.add_argument("--out-dir", type=str, default="results/time_budgeted_ils")
    parser.add_argument("--instances", type=str, nargs="*", default=None)
    parser.add_argument("--n-runs", type=int, default=10)
    parser.add_argument("--seed-offset", type=int, default=50_000)

    parser.add_argument("--construction-starts", type=int, default=10)
    parser.add_argument("--two-opt-iterations", type=int, default=30)

    parser.add_argument(
        "--budget-scale",
        type=float,
        default=1.0,
        help="Multiplier for the V3 runtime budget. Use 1.0 for equal runtime, 0.5 or 2.0 for sensitivity.",
    )

    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    budgets = load_v3_time_budgets(Path(args.v3_summary_path))

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

        if instance_name not in budgets:
            print(f"Skipping {instance_name}: no V3 runtime budget found")
            continue

        instance = load_tsplib_instance(path)
        optimum = TSPLIB_OPTIMA[instance_name]

        time_budget_sec = budgets[instance_name] * args.budget_scale

        print(
            f"\nRunning time-budgeted ILS on {instance_name} "
            f"(n={instance.coords.shape[0]}, budget={time_budget_sec:.3f}s)"
        )

        for run_idx in range(args.n_runs):
            run_seed = args.seed_offset + run_idx

            row = run_one(
                instance_name=instance_name,
                instance=instance,
                optimum=optimum,
                run_seed=run_seed,
                time_budget_sec=time_budget_sec,
                construction_starts=args.construction_starts,
                two_opt_iterations=args.two_opt_iterations,
            )

            all_rows.append(row)

            print(
                f"  run={run_idx + 1}/{args.n_runs} | "
                f"gap={row['gap_to_optimum_percent']:.3f}% | "
                f"iters={row['iterations_completed']} | "
                f"time={row['runtime_sec']:.3f}s"
            )

    raw = pd.DataFrame(all_rows)
    summary = summarize_results(raw)

    raw_path = out_dir / "time_budgeted_ils_raw.csv"
    summary_path = out_dir / "time_budgeted_ils_summary.csv"

    raw.to_csv(raw_path, index=False)
    summary.to_csv(summary_path, index=False)

    print("\nTime-budgeted ILS finished.")
    print(f"Raw saved to: {raw_path}")
    print(f"Summary saved to: {summary_path}")

    print("\nSummary:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()