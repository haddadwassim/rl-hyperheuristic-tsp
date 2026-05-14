import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from tsp_hh.bc_policy import NextCityPolicy
from tsp_hh.exact_tsp import held_karp_optimal_tour
from tsp_hh.heuristics_v3 import construct_tour_v3, two_opt_local_search_limited
from tsp_hh.tour import tour_length
from tsp_hh.tsplib_loader import list_tsplib_files, load_tsplib_instance

from train_behavior_cloning_exact import (
    normalize_coords,
    compute_distance_matrix,
    rollout_policy,
    repair_tour_with_2opt,
)


def sample_subinstance(
    coords: np.ndarray,
    n_cities: int,
    seed: int,
) -> tuple[np.ndarray, list[int]]:
    rng = np.random.default_rng(seed)

    if coords.shape[0] < n_cities:
        raise ValueError("TSPLIB instance has fewer cities than requested subinstance size")

    selected = sorted(rng.choice(coords.shape[0], size=n_cities, replace=False).tolist())
    sub_coords = coords[selected]

    return sub_coords, selected


def build_tsplib_subinstance_meta(
    instance_name: str,
    coords: np.ndarray,
    n_cities: int,
    sample_seed: int,
) -> dict:
    sub_coords_raw, selected_indices = sample_subinstance(
        coords=coords,
        n_cities=n_cities,
        seed=sample_seed,
    )

    coords_norm = normalize_coords(sub_coords_raw)
    distance_matrix = compute_distance_matrix(coords_norm)

    optimal_tour, optimal_length = held_karp_optimal_tour(
        distance_matrix,
        start_city=0,
    )

    return {
        "instance": instance_name,
        "sample_seed": sample_seed,
        "selected_indices": selected_indices,
        "coords": coords_norm,
        "distance_matrix": distance_matrix,
        "optimal_tour": optimal_tour,
        "optimal_length": optimal_length,
    }


def evaluate_bc(
    model: NextCityPolicy,
    meta: dict,
    device: torch.device,
    repair_2opt: bool,
    two_opt_iterations: int,
) -> dict:
    start_time = time.perf_counter()

    path, length = rollout_policy(
        model=model,
        coords=meta["coords"],
        distance_matrix=meta["distance_matrix"],
        start_city=0,
        device=device,
    )

    if repair_2opt:
        path, length = repair_tour_with_2opt(
            path=path,
            distance_matrix=meta["distance_matrix"],
            max_iterations=two_opt_iterations,
        )

    runtime_sec = time.perf_counter() - start_time

    gap = (length - meta["optimal_length"]) / meta["optimal_length"] * 100.0

    return {
        "instance": meta["instance"],
        "sample_seed": meta["sample_seed"],
        "method": "bc_policy_2opt" if repair_2opt else "bc_policy",
        "optimal_length": meta["optimal_length"],
        "predicted_length": length,
        "gap_percent": gap,
        "exact_path": path == meta["optimal_tour"],
        "runtime_sec": runtime_sec,
        "runtime_ms": runtime_sec * 1000.0,
        "predicted_path": path,
        "optimal_tour": meta["optimal_tour"],
        "selected_indices": meta["selected_indices"],
    }


def evaluate_baseline(
    meta: dict,
    method: str,
    repair_2opt: bool,
    two_opt_iterations: int,
) -> dict:
    start_time = time.perf_counter()

    tour = construct_tour_v3(
        distance_matrix=meta["distance_matrix"],
        method=method,
        seed=meta["sample_seed"],
        n_starts=1,
    )

    if repair_2opt:
        tour, _, _ = two_opt_local_search_limited(
            np.asarray(tour, dtype=int),
            meta["distance_matrix"],
            max_iterations=two_opt_iterations,
        )

    length = tour_length(np.asarray(tour), meta["distance_matrix"])
    runtime_sec = time.perf_counter() - start_time

    gap = (length - meta["optimal_length"]) / meta["optimal_length"] * 100.0
    tour_list = list(map(int, tour))

    method_name = f"{method}_2opt" if repair_2opt else method

    return {
        "instance": meta["instance"],
        "sample_seed": meta["sample_seed"],
        "method": method_name,
        "optimal_length": meta["optimal_length"],
        "predicted_length": length,
        "gap_percent": gap,
        "exact_path": tour_list == meta["optimal_tour"],
        "runtime_sec": runtime_sec,
        "runtime_ms": runtime_sec * 1000.0,
        "predicted_path": tour_list,
        "optimal_tour": meta["optimal_tour"],
        "selected_indices": meta["selected_indices"],
    }


def summarize(raw: pd.DataFrame) -> pd.DataFrame:
    summary = (
        raw.groupby(["method"])
        .agg(
            n_cases=("gap_percent", "count"),
            mean_gap=("gap_percent", "mean"),
            median_gap=("gap_percent", "median"),
            best_gap=("gap_percent", "min"),
            worst_gap=("gap_percent", "max"),
            exact_rate=("exact_path", "mean"),
            mean_runtime_ms=("runtime_ms", "mean"),
        )
        .reset_index()
        .sort_values("mean_gap")
    )

    return summary


def summarize_by_instance(raw: pd.DataFrame) -> pd.DataFrame:
    summary = (
        raw.groupby(["instance", "method"])
        .agg(
            n_cases=("gap_percent", "count"),
            mean_gap=("gap_percent", "mean"),
            median_gap=("gap_percent", "median"),
            exact_rate=("exact_path", "mean"),
            mean_runtime_ms=("runtime_ms", "mean"),
        )
        .reset_index()
        .sort_values(["instance", "mean_gap"])
    )

    return summary


def load_model(
    model_path: Path,
    hidden_dim: int,
    device: torch.device,
) -> NextCityPolicy:
    model = NextCityPolicy(
        feature_dim=9,
        hidden_dim=hidden_dim,
    ).to(device)

    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    return model


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data-dir", type=str, default="data/tsplib")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--hidden-dim", type=int, default=128)

    parser.add_argument("--n-cities", type=int, default=10)
    parser.add_argument("--samples-per-instance", type=int, default=20)
    parser.add_argument("--sample-seed-offset", type=int, default=10_000)

    parser.add_argument(
        "--instances",
        type=str,
        nargs="*",
        default=["eil51", "berlin52", "st70", "eil76", "kroA100", "ch130"],
    )

    parser.add_argument("--two-opt-repair-iterations", type=int, default=50)
    parser.add_argument("--out-dir", type=str, default="results/bc_tsplib_subinstances")

    args = parser.parse_args()

    out_dir = Path(args.out_dir) / f"n{args.n_cities}"
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_model(
        model_path=Path(args.model_path),
        hidden_dim=args.hidden_dim,
        device=device,
    )

    wanted = {name if name.endswith(".tsp") else f"{name}.tsp" for name in args.instances}
    tsp_files = [p for p in list_tsplib_files(args.data_dir) if p.name in wanted]

    if not tsp_files:
        raise FileNotFoundError("No matching TSPLIB files found")

    rows = []

    for path in tsp_files:
        instance_name = path.stem
        instance = load_tsplib_instance(path)

        print(f"\nEvaluating TSPLIB subinstances from {instance_name}")

        for sample_idx in range(args.samples_per_instance):
            sample_seed = args.sample_seed_offset + sample_idx

            meta = build_tsplib_subinstance_meta(
                instance_name=instance_name,
                coords=instance.coords,
                n_cities=args.n_cities,
                sample_seed=sample_seed,
            )

            case_rows = [
                evaluate_bc(
                    model=model,
                    meta=meta,
                    device=device,
                    repair_2opt=False,
                    two_opt_iterations=args.two_opt_repair_iterations,
                ),
                evaluate_bc(
                    model=model,
                    meta=meta,
                    device=device,
                    repair_2opt=True,
                    two_opt_iterations=args.two_opt_repair_iterations,
                ),
                evaluate_baseline(
                    meta=meta,
                    method="random",
                    repair_2opt=False,
                    two_opt_iterations=args.two_opt_repair_iterations,
                ),
                evaluate_baseline(
                    meta=meta,
                    method="nearest_neighbor",
                    repair_2opt=False,
                    two_opt_iterations=args.two_opt_repair_iterations,
                ),
                evaluate_baseline(
                    meta=meta,
                    method="nearest_neighbor",
                    repair_2opt=True,
                    two_opt_iterations=args.two_opt_repair_iterations,
                ),
            ]

            rows.extend(case_rows)

            best_row = min(case_rows, key=lambda r: r["gap_percent"])

            print(
                f"  sample {sample_idx + 1}/{args.samples_per_instance} | "
                f"best={best_row['method']} | "
                f"gap={best_row['gap_percent']:.3f}%"
            )

    raw = pd.DataFrame(rows)
    summary = summarize(raw)
    by_instance = summarize_by_instance(raw)

    raw_path = out_dir / "tsplib_subinstances_raw.csv"
    summary_path = out_dir / "tsplib_subinstances_summary.csv"
    by_instance_path = out_dir / "tsplib_subinstances_by_instance.csv"

    raw.to_csv(raw_path, index=False)
    summary.to_csv(summary_path, index=False)
    by_instance.to_csv(by_instance_path, index=False)

    print("\nOverall summary:")
    print(summary.to_string(index=False))

    print("\nSummary by TSPLIB source instance:")
    print(by_instance.to_string(index=False))

    print(f"\nSaved raw results to: {raw_path}")
    print(f"Saved summary to: {summary_path}")
    print(f"Saved by-instance summary to: {by_instance_path}")


if __name__ == "__main__":
    main()