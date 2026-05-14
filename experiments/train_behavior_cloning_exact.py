import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from tsp_hh.instances import generate_euclidean_instance
from tsp_hh.exact_tsp import held_karp_optimal_tour
from tsp_hh.tour import tour_length
from tsp_hh.bc_policy import NextCityPolicy
from tsp_hh.heuristics_v3 import (
    construct_tour_v3,
    two_opt_local_search_limited,
)


# ============================================================
# Data / feature utilities
# ============================================================

def normalize_coords(coords: np.ndarray) -> np.ndarray:
    coords = np.asarray(coords, dtype=np.float32)

    min_xy = coords.min(axis=0)
    max_xy = coords.max(axis=0)
    scale = np.maximum(max_xy - min_xy, 1e-12)

    return (coords - min_xy) / scale


def compute_distance_matrix(coords: np.ndarray) -> np.ndarray:
    diff = coords[:, None, :] - coords[None, :, :]
    return np.linalg.norm(diff, axis=-1)


def build_candidate_features(
    coords: np.ndarray,
    distance_matrix: np.ndarray,
    current_city: int,
    start_city: int,
    visited: np.ndarray,
) -> np.ndarray:
    """
    Build candidate-level features.

    For each candidate city i:
      - candidate coordinates
      - current city coordinates
      - start city coordinates
      - distance(current, i)
      - distance(i, start)
      - visited flag
    """
    n_cities = coords.shape[0]

    current_xy = coords[current_city]
    start_xy = coords[start_city]

    current_xy_repeated = np.repeat(current_xy[None, :], n_cities, axis=0)
    start_xy_repeated = np.repeat(start_xy[None, :], n_cities, axis=0)

    max_dist = max(float(distance_matrix.max()), 1e-12)

    dist_from_current = distance_matrix[current_city] / max_dist
    dist_to_start = distance_matrix[:, start_city] / max_dist

    features = np.concatenate(
        [
            coords,
            current_xy_repeated,
            start_xy_repeated,
            dist_from_current[:, None],
            dist_to_start[:, None],
            visited[:, None].astype(np.float32),
        ],
        axis=1,
    )

    return features.astype(np.float32)


# ============================================================
# Exact data generation
# ============================================================

def build_examples_for_instance(
    n_cities: int,
    seed: int,
    start_city: int = 0,
) -> tuple[list[np.ndarray], list[np.ndarray], list[int], dict]:
    instance = generate_euclidean_instance(n_cities=n_cities, seed=seed)

    coords = normalize_coords(instance.coords)
    distance_matrix = compute_distance_matrix(coords)

    optimal_tour, optimal_length = held_karp_optimal_tour(
        distance_matrix,
        start_city=start_city,
    )

    X = []
    masks = []
    y = []

    visited = np.zeros(n_cities, dtype=np.float32)
    visited[start_city] = 1.0

    for step in range(1, n_cities):
        current_city = optimal_tour[step - 1]
        target_city = optimal_tour[step]

        features = build_candidate_features(
            coords=coords,
            distance_matrix=distance_matrix,
            current_city=current_city,
            start_city=start_city,
            visited=visited,
        )

        valid_mask = visited == 0

        X.append(features)
        masks.append(valid_mask.astype(bool))
        y.append(target_city)

        visited[target_city] = 1.0

    meta = {
        "seed": seed,
        "coords": coords,
        "distance_matrix": distance_matrix,
        "optimal_tour": optimal_tour,
        "optimal_length": optimal_length,
    }

    return X, masks, y, meta


def build_dataset(
    n_cities: int,
    seeds: list[int],
) -> tuple[TensorDataset, list[dict]]:
    X_all = []
    masks_all = []
    y_all = []
    metas = []

    for seed in seeds:
        X, masks, y, meta = build_examples_for_instance(
            n_cities=n_cities,
            seed=seed,
            start_city=0,
        )

        X_all.extend(X)
        masks_all.extend(masks)
        y_all.extend(y)
        metas.append(meta)

    X_tensor = torch.tensor(np.stack(X_all), dtype=torch.float32)
    masks_tensor = torch.tensor(np.stack(masks_all), dtype=torch.bool)
    y_tensor = torch.tensor(np.asarray(y_all), dtype=torch.long)

    dataset = TensorDataset(X_tensor, masks_tensor, y_tensor)

    return dataset, metas


# ============================================================
# Training utilities
# ============================================================

def masked_cross_entropy(
    logits: torch.Tensor,
    valid_mask: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    masked_logits = logits.masked_fill(~valid_mask, -1e9)
    return F.cross_entropy(masked_logits, target)


@torch.no_grad()
def evaluate_next_city_accuracy(
    model: NextCityPolicy,
    dataset: TensorDataset,
    device: torch.device,
    batch_size: int = 512,
) -> float:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    correct = 0
    total = 0

    model.eval()

    for X, masks, y in loader:
        X = X.to(device)
        masks = masks.to(device)
        y = y.to(device)

        logits = model(X)
        logits = logits.masked_fill(~masks, -1e9)

        pred = logits.argmax(dim=1)

        correct += int((pred == y).sum().item())
        total += int(y.numel())

    return correct / max(total, 1)


# ============================================================
# Rollout / repair
# ============================================================

@torch.no_grad()
def rollout_policy(
    model: NextCityPolicy,
    coords: np.ndarray,
    distance_matrix: np.ndarray,
    start_city: int,
    device: torch.device,
) -> tuple[list[int], float]:
    model.eval()

    n_cities = coords.shape[0]

    visited = np.zeros(n_cities, dtype=np.float32)
    visited[start_city] = 1.0

    path = [start_city]
    current_city = start_city

    for _ in range(n_cities - 1):
        features = build_candidate_features(
            coords=coords,
            distance_matrix=distance_matrix,
            current_city=current_city,
            start_city=start_city,
            visited=visited,
        )

        valid_mask = visited == 0

        X = torch.tensor(features[None, :, :], dtype=torch.float32).to(device)
        mask = torch.tensor(valid_mask[None, :], dtype=torch.bool).to(device)

        logits = model(X)
        logits = logits.masked_fill(~mask, -1e9)

        next_city = int(logits.argmax(dim=1).item())

        path.append(next_city)
        visited[next_city] = 1.0
        current_city = next_city

    length = tour_length(np.asarray(path), distance_matrix)

    return path, float(length)


def repair_tour_with_2opt(
    path: list[int],
    distance_matrix: np.ndarray,
    max_iterations: int,
) -> tuple[list[int], float]:
    repaired_tour, _, _ = two_opt_local_search_limited(
        np.asarray(path, dtype=int),
        distance_matrix,
        max_iterations=max_iterations,
    )

    repaired_path = list(map(int, repaired_tour))
    repaired_length = tour_length(np.asarray(repaired_path), distance_matrix)

    return repaired_path, float(repaired_length)


@torch.no_grad()
def evaluate_bc_policy(
    model: NextCityPolicy,
    metas: list[dict],
    device: torch.device,
    repair_2opt: bool = False,
    two_opt_iterations: int = 50,
) -> pd.DataFrame:
    rows = []

    for meta in metas:
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

        optimal_length = meta["optimal_length"]
        gap = (length - optimal_length) / optimal_length * 100.0

        rows.append(
            {
                "seed": meta["seed"],
                "method": "bc_policy_2opt" if repair_2opt else "bc_policy",
                "optimal_length": optimal_length,
                "predicted_length": length,
                "gap_percent": gap,
                "exact_path": path == meta["optimal_tour"],
                "predicted_path": path,
                "optimal_tour": meta["optimal_tour"],
                "runtime_sec": runtime_sec,
                "runtime_ms": runtime_sec * 1000.0,
            }
        )

    return pd.DataFrame(rows)


def evaluate_constructive_baseline(
    metas: list[dict],
    method: str,
    repair_2opt: bool = False,
    two_opt_iterations: int = 50,
) -> pd.DataFrame:
    rows = []

    method_name = f"{method}_2opt" if repair_2opt else method

    for meta in metas:
        seed = meta["seed"]
        distance_matrix = meta["distance_matrix"]
        optimal_length = meta["optimal_length"]

        start_time = time.perf_counter()

        tour = construct_tour_v3(
            distance_matrix=distance_matrix,
            method=method,
            seed=seed,
            n_starts=1,
        )

        if repair_2opt:
            tour, _, _ = two_opt_local_search_limited(
                np.asarray(tour, dtype=int),
                distance_matrix,
                max_iterations=two_opt_iterations,
            )

        length = tour_length(np.asarray(tour), distance_matrix)
        runtime_sec = time.perf_counter() - start_time

        gap = (length - optimal_length) / optimal_length * 100.0
        tour_list = list(map(int, tour))

        rows.append(
            {
                "seed": seed,
                "method": method_name,
                "optimal_length": optimal_length,
                "predicted_length": length,
                "gap_percent": gap,
                "exact_path": tour_list == meta["optimal_tour"],
                "predicted_path": tour_list,
                "optimal_tour": meta["optimal_tour"],
                "runtime_sec": runtime_sec,
                "runtime_ms": runtime_sec * 1000.0,
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# Reporting
# ============================================================

def summarize_method(df: pd.DataFrame, method_name: str) -> dict:
    return {
        "method": method_name,
        "mean_gap": df["gap_percent"].mean(),
        "median_gap": df["gap_percent"].median(),
        "best_gap": df["gap_percent"].min(),
        "worst_gap": df["gap_percent"].max(),
        "exact_rate": df["exact_path"].mean(),
        "mean_runtime_ms": df["runtime_ms"].mean(),
    }


def print_method_summary(name: str, df: pd.DataFrame) -> None:
    row = summarize_method(df, name)

    print(
        f"{name:24s} | "
        f"mean_gap={row['mean_gap']:.3f}% | "
        f"median_gap={row['median_gap']:.3f}% | "
        f"best_gap={row['best_gap']:.3f}% | "
        f"worst_gap={row['worst_gap']:.3f}% | "
        f"exact_rate={row['exact_rate']:.3f} | "
        f"mean_runtime_ms={row['mean_runtime_ms']:.3f}"
    )


def parse_seed_range(text: str) -> list[int]:
    """
    Supports:
      "42:142" -> 42..141
      "42,43,44"
    """
    text = text.strip()

    if ":" in text:
        start, end = text.split(":")
        return list(range(int(start), int(end)))

    return [int(x.strip()) for x in text.split(",") if x.strip()]


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--n-cities", type=int, default=10)
    parser.add_argument("--train-seeds", type=str, default="42:542")
    parser.add_argument("--test-seeds", type=str, default="1000:1100")

    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)

    parser.add_argument("--two-opt-repair-iterations", type=int, default=50)

    parser.add_argument("--out-dir", type=str, default="results/behavior_cloning_exact_clean")

    args = parser.parse_args()

    out_dir = Path(args.out_dir) / f"n{args.n_cities}"
    out_dir.mkdir(parents=True, exist_ok=True)

    train_seeds = parse_seed_range(args.train_seeds)
    test_seeds = parse_seed_range(args.test_seeds)

    print(f"Building training dataset: n={args.n_cities}, instances={len(train_seeds)}")
    train_dataset, train_metas = build_dataset(args.n_cities, train_seeds)

    print(f"Building test dataset: n={args.n_cities}, instances={len(test_seeds)}")
    test_dataset, test_metas = build_dataset(args.n_cities, test_seeds)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = NextCityPolicy(
        feature_dim=9,
        hidden_dim=args.hidden_dim,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    logs = []

    for epoch in range(1, args.epochs + 1):
        model.train()

        total_loss = 0.0
        n_batches = 0

        for X, masks, y in train_loader:
            X = X.to(device)
            masks = masks.to(device)
            y = y.to(device)

            logits = model(X)
            loss = masked_cross_entropy(logits, masks, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())
            n_batches += 1

        train_acc = evaluate_next_city_accuracy(model, train_dataset, device)
        test_acc = evaluate_next_city_accuracy(model, test_dataset, device)

        # Raw BC rollout only during training, for fast progress tracking.
        test_rollout = evaluate_bc_policy(
            model=model,
            metas=test_metas,
            device=device,
            repair_2opt=False,
        )

        row = {
            "epoch": epoch,
            "loss": total_loss / max(n_batches, 1),
            "train_next_city_acc": train_acc,
            "test_next_city_acc": test_acc,
            "test_raw_mean_gap": test_rollout["gap_percent"].mean(),
            "test_raw_exact_rate": test_rollout["exact_path"].mean(),
        }

        logs.append(row)

        print(
            f"Epoch {epoch:03d} | "
            f"loss={row['loss']:.4f} | "
            f"train_acc={train_acc:.3f} | "
            f"test_acc={test_acc:.3f} | "
            f"raw_gap={row['test_raw_mean_gap']:.2f}% | "
            f"raw_exact={row['test_raw_exact_rate']:.3f}"
        )

    # Save model and training log.
    torch.save(model.state_dict(), out_dir / "bc_policy.pt")
    pd.DataFrame(logs).to_csv(out_dir / "training_log.csv", index=False)

    # Final evaluation.
    bc_eval = evaluate_bc_policy(
        model=model,
        metas=test_metas,
        device=device,
        repair_2opt=False,
    )

    bc_2opt_eval = evaluate_bc_policy(
        model=model,
        metas=test_metas,
        device=device,
        repair_2opt=True,
        two_opt_iterations=args.two_opt_repair_iterations,
    )

    random_eval = evaluate_constructive_baseline(
        metas=test_metas,
        method="random",
        repair_2opt=False,
    )

    nn_eval = evaluate_constructive_baseline(
        metas=test_metas,
        method="nearest_neighbor",
        repair_2opt=False,
    )

    nn_2opt_eval = evaluate_constructive_baseline(
        metas=test_metas,
        method="nearest_neighbor",
        repair_2opt=True,
        two_opt_iterations=args.two_opt_repair_iterations,
    )

    # Save detailed evaluations.
    bc_eval.to_csv(out_dir / "test_bc_policy.csv", index=False)
    bc_2opt_eval.to_csv(out_dir / "test_bc_policy_2opt.csv", index=False)
    random_eval.to_csv(out_dir / "test_random.csv", index=False)
    nn_eval.to_csv(out_dir / "test_nearest_neighbor.csv", index=False)
    nn_2opt_eval.to_csv(out_dir / "test_nearest_neighbor_2opt.csv", index=False)

    # Print clean summaries.
    print("\nFinal method summaries:")
    print_method_summary("bc_policy", bc_eval)
    print_method_summary("bc_policy_2opt", bc_2opt_eval)
    print_method_summary("random", random_eval)
    print_method_summary("nearest_neighbor", nn_eval)
    print_method_summary("nearest_neighbor_2opt", nn_2opt_eval)

    comparison_rows = [
        summarize_method(bc_eval, "bc_policy"),
        summarize_method(bc_2opt_eval, "bc_policy_2opt"),
        summarize_method(random_eval, "random"),
        summarize_method(nn_eval, "nearest_neighbor"),
        summarize_method(nn_2opt_eval, "nearest_neighbor_2opt"),
    ]

    comparison_df = pd.DataFrame(comparison_rows)
    comparison_df.to_csv(out_dir / "test_method_comparison.csv", index=False)

    print("\nMethod comparison:")
    print(comparison_df.to_string(index=False))

    print(f"\nSaved model and results to: {out_dir}")


if __name__ == "__main__":
    main()