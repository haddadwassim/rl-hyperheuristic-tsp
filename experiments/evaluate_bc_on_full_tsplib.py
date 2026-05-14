import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from tsp_hh.bc_policy import NextCityPolicy
from tsp_hh.heuristics_v3 import construct_tour_v3, two_opt_local_search_limited
from tsp_hh.tour import tour_length
from tsp_hh.tsplib_loader import list_tsplib_files, load_tsplib_instance

from train_behavior_cloning_exact import (
    normalize_coords,
    compute_distance_matrix,
    rollout_policy,
    repair_tour_with_2opt,
)


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


def build_full_tsplib_meta(instance_name: str, instance) -> dict:
    coords_model = normalize_coords(instance.coords)
    model_distance_matrix = compute_distance_matrix(coords_model)

    eval_distance_matrix = instance.distance_matrix

    if instance_name not in TSPLIB_OPTIMA:
        raise ValueError(f"No known optimum for {instance_name}")

    return {
        "instance": instance_name,
        "coords_model": coords_model,
        "model_distance_matrix": model_distance_matrix,
        "eval_distance_matrix": eval_distance_matrix,
        "known_optimum": TSPLIB_OPTIMA[instance_name],
        "n_cities": coords_model.shape[0],
    }


def evaluate_bc(
    model: NextCityPolicy,
    meta: dict,
    device: torch.device,
    repair_2opt: bool,
    two_opt_iterations: int,
) -> dict:
    start_time = time.perf_counter()

    path, _ = rollout_policy(
        model=model,
        coords=meta["coords_model"],
        distance_matrix=meta["model_distance_matrix"],
        start_city=0,
        device=device,
    )
    length = tour_length(np.asarray(path), meta["eval_distance_matrix"])

    if repair_2opt:
        path, length = repair_tour_with_2opt(
            path=path,
            distance_matrix=meta["eval_distance_matrix"],
            max_iterations=two_opt_iterations,
        )

    runtime_sec = time.perf_counter() - start_time

    gap = (length - meta["known_optimum"]) / meta["known_optimum"] * 100.0

    return {
        "instance": meta["instance"],
        "n_cities": meta["n_cities"],
        "method": "bc_policy_2opt" if repair_2opt else "bc_policy",
        "known_optimum": meta["known_optimum"],
        "tour_length": length,
        "gap_to_optimum_percent": gap,
        "runtime_sec": runtime_sec,
        "runtime_ms": runtime_sec * 1000.0,
        "path": path,
    }


def evaluate_baseline(
    meta: dict,
    method: str,
    repair_2opt: bool,
    two_opt_iterations: int,
) -> dict:
    start_time = time.perf_counter()

    tour = construct_tour_v3(
        distance_matrix=meta["eval_distance_matrix"],
        method=method,
        seed=42,
        n_starts=1,
    )

    if repair_2opt:
        tour, _, _ = two_opt_local_search_limited(
            np.asarray(tour, dtype=int),
            meta["eval_distance_matrix"],
            max_iterations=two_opt_iterations,
        )

    length = tour_length(np.asarray(tour), meta["eval_distance_matrix"])

    runtime_sec = time.perf_counter() - start_time

    gap = (length - meta["known_optimum"]) / meta["known_optimum"] * 100.0

    method_name = f"{method}_2opt" if repair_2opt else method

    return {
        "instance": meta["instance"],
        "n_cities": meta["n_cities"],
        "method": method_name,
        "known_optimum": meta["known_optimum"],
        "tour_length": length,
        "gap_to_optimum_percent": gap,
        "runtime_sec": runtime_sec,
        "runtime_ms": runtime_sec * 1000.0,
        "path": list(map(int, tour)),
    }


def summarize(raw: pd.DataFrame) -> pd.DataFrame:
    return (
        raw.sort_values(["instance", "gap_to_optimum_percent"])
        .reset_index(drop=True)
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data-dir", type=str, default="data/tsplib")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--hidden-dim", type=int, default=256)

    parser.add_argument(
        "--instances",
        type=str,
        nargs="*",
        default=[
            "eil51",
            "berlin52",
            "st70",
            "eil76",
            "pr76",
            "rat99",
            "kroA100",
            "kroB100",
            "eil101",
            "ch130",
        ],
    )

    parser.add_argument("--two-opt-repair-iterations", type=int, default=300)
    parser.add_argument("--out-dir", type=str, default="results/bc_full_tsplib")

    args = parser.parse_args()

    out_dir = Path(args.out_dir)
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

        if instance_name not in TSPLIB_OPTIMA:
            print(f"Skipping {instance_name}: no known optimum")
            continue

        instance = load_tsplib_instance(path)

        meta = build_full_tsplib_meta(
            instance_name=instance_name,
            instance=instance,
        )

        print(f"\nEvaluating full TSPLIB instance: {instance_name} (n={meta['n_cities']})")

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

        for row in sorted(case_rows, key=lambda r: r["gap_to_optimum_percent"]):
            print(
                f"  {row['method']:22s} | "
                f"gap={row['gap_to_optimum_percent']:.3f}% | "
                f"len={row['tour_length']:.3f} | "
                f"time={row['runtime_ms']:.2f} ms"
            )

    raw = pd.DataFrame(rows)
    summary = summarize(raw)

    raw_path = out_dir / "full_tsplib_raw.csv"
    summary_path = out_dir / "full_tsplib_summary.csv"

    raw.to_csv(raw_path, index=False)
    summary.to_csv(summary_path, index=False)

    print("\nFull TSPLIB summary:")
    print(summary.to_string(index=False))

    print(f"\nSaved raw results to: {raw_path}")
    print(f"Saved summary to: {summary_path}")


if __name__ == "__main__":
    main()