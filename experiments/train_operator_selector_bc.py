import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from tsp_hh.instances import generate_euclidean_instance
from tsp_hh.heuristics_v3 import construct_tour_v3, two_opt_local_search_limited
from tsp_hh.tour import tour_length
from tsp_hh.operators import apply_operator, ACTION_NAMES


# ============================================================
# Basic utilities
# ============================================================

def compute_distance_matrix(coords: np.ndarray) -> np.ndarray:
    diff = coords[:, None, :] - coords[None, :, :]
    return np.linalg.norm(diff, axis=-1)


def normalize_coords(coords: np.ndarray) -> np.ndarray:
    coords = np.asarray(coords, dtype=np.float32)
    min_xy = coords.min(axis=0)
    max_xy = coords.max(axis=0)
    scale = np.maximum(max_xy - min_xy, 1e-12)
    return (coords - min_xy) / scale


def parse_int_list(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


# ============================================================
# State features
# ============================================================

def edge_lengths(tour: list[int], distance_matrix: np.ndarray) -> np.ndarray:
    arr = np.asarray(tour, dtype=int)
    nxt = np.roll(arr, -1)
    return distance_matrix[arr, nxt]


def build_state_features(
    tour: list[int],
    distance_matrix: np.ndarray,
    initial_length: float,
    current_step: int,
    max_steps: int,
    last_improvement: float,
    no_improve_steps: int,
    last_action: int,
) -> np.ndarray:
    n = len(tour)
    length = tour_length(np.asarray(tour, dtype=int), distance_matrix)
    edges = edge_lengths(tour, distance_matrix)

    mean_edge = float(edges.mean())
    std_edge = float(edges.std())
    max_edge = float(edges.max())
    min_edge = float(edges.min())

    sorted_edges = np.sort(edges)
    top_10_percent = sorted_edges[int(0.9 * len(sorted_edges)) :]
    long_edge_mean = float(top_10_percent.mean()) if len(top_10_percent) > 0 else max_edge

    features = np.array(
        [
            n / 150.0,
            length / max(initial_length, 1e-12),
            current_step / max(max_steps, 1),
            last_improvement / max(initial_length, 1e-12),
            no_improve_steps / max(max_steps, 1),
            last_action / max(len(ACTION_NAMES) - 1, 1),
            mean_edge,
            std_edge,
            max_edge,
            min_edge,
            long_edge_mean,
            max_edge / max(mean_edge, 1e-12),
            std_edge / max(mean_edge, 1e-12),
        ],
        dtype=np.float32,
    )

    return features


# ============================================================
# Teacher action
# ============================================================

def choose_teacher_action(
    tour: list[int],
    distance_matrix: np.ndarray,
    seed: int,
    two_opt_iterations: int,
    max_trials: int,
    perturb_swaps: int,
    min_improvement: float = 1e-12,
) -> tuple[int, dict[int, float]]:
    """
    Try every non-stop operator and choose the one with best immediate improvement.
    If no operator improves, teacher chooses STOP.
    """
    improvements = {}

    best_action = 0
    best_improvement = 0.0

    for action in ACTION_NAMES:
        if action == 0:
            continue

        result = apply_operator(
            action=action,
            tour=tour,
            distance_matrix=distance_matrix,
            seed=seed + action * 1009,
            two_opt_iterations=two_opt_iterations,
            max_trials=max_trials,
            perturb_swaps=perturb_swaps,
        )

        improvements[action] = result.improvement

        if result.improvement > best_improvement + min_improvement:
            best_improvement = result.improvement
            best_action = action

    return best_action, improvements


# ============================================================
# Dataset generation
# ============================================================

def build_initial_tour(
    distance_matrix: np.ndarray,
    seed: int,
    initial_method: str,
) -> list[int]:
    if initial_method == "nearest_neighbor":
        tour = construct_tour_v3(
            distance_matrix=distance_matrix,
            method="nearest_neighbor",
            seed=seed,
            n_starts=1,
        )
        return list(map(int, tour))

    if initial_method == "random":
        tour = construct_tour_v3(
            distance_matrix=distance_matrix,
            method="random",
            seed=seed,
            n_starts=1,
        )
        return list(map(int, tour))

    if initial_method == "nearest_neighbor_perturbed":
        tour = construct_tour_v3(
            distance_matrix=distance_matrix,
            method="nearest_neighbor",
            seed=seed,
            n_starts=1,
        )
        tour = list(map(int, tour))

        rng = np.random.default_rng(seed)
        n = len(tour)

        for _ in range(5):
            i, j = rng.choice(np.arange(1, n), size=2, replace=False)
            tour[i], tour[j] = tour[j], tour[i]

        return tour

    if initial_method == "random_2opt_light":
        tour = construct_tour_v3(
            distance_matrix=distance_matrix,
            method="random",
            seed=seed,
            n_starts=1,
        )

        tour, _, _ = two_opt_local_search_limited(
            np.asarray(tour, dtype=int),
            distance_matrix,
            max_iterations=10,
        )

        return list(map(int, tour))

    raise ValueError(f"Unknown initial_method: {initial_method}")


def build_teacher_dataset(
    n_city_choices: list[int],
    n_instances: int,
    seed_offset: int,
    max_steps: int,
    two_opt_iterations: int,
    max_trials: int,
    perturb_swaps: int,
) -> tuple[TensorDataset, pd.DataFrame]:
    X_rows = []
    y_rows = []
    log_rows = []

    rng = np.random.default_rng(seed_offset)

    for instance_idx in range(n_instances):
        n_cities = int(rng.choice(n_city_choices))
        seed = seed_offset + instance_idx

        instance = generate_euclidean_instance(n_cities=n_cities, seed=seed)
        coords = normalize_coords(instance.coords)
        distance_matrix = compute_distance_matrix(coords)

        initial_methods = [
            "nearest_neighbor",
            "random",
            "nearest_neighbor_perturbed",
            "random_2opt_light",
        ]

        initial_method = initial_methods[instance_idx % len(initial_methods)]

        tour = build_initial_tour(
            distance_matrix=distance_matrix,
            seed=seed,
            initial_method=initial_method,
        )
        tour = list(map(int, tour))

        initial_length = tour_length(np.asarray(tour, dtype=int), distance_matrix)

        last_improvement = 0.0
        no_improve_steps = 0
        last_action = 0

        for step in range(max_steps):
            current_length = tour_length(np.asarray(tour, dtype=int), distance_matrix)

            features = build_state_features(
                tour=tour,
                distance_matrix=distance_matrix,
                initial_length=initial_length,
                current_step=step,
                max_steps=max_steps,
                last_improvement=last_improvement,
                no_improve_steps=no_improve_steps,
                last_action=last_action,
            )

            teacher_action, improvements = choose_teacher_action(
                tour=tour,
                distance_matrix=distance_matrix,
                seed=seed + step * 100_000,
                two_opt_iterations=two_opt_iterations,
                max_trials=max_trials,
                perturb_swaps=perturb_swaps,
            )

            X_rows.append(features)
            y_rows.append(teacher_action)

            log_rows.append(
                {
                    "instance_idx": instance_idx,
                    "seed": seed,
                    "n_cities": n_cities,
                    "step": step,
                    "current_length": current_length,
                    "initial_length": initial_length,
                    "teacher_action": teacher_action,
                    "teacher_action_name": ACTION_NAMES[teacher_action],
                    "initial_method": initial_method,
                    **{f"improvement_{ACTION_NAMES[a]}": improvements.get(a, 0.0) for a in ACTION_NAMES},
                }
            )

            if teacher_action == 0:
                break

            result = apply_operator(
                action=teacher_action,
                tour=tour,
                distance_matrix=distance_matrix,
                seed=seed + step * 100_000 + teacher_action * 1009,
                two_opt_iterations=two_opt_iterations,
                max_trials=max_trials,
                perturb_swaps=perturb_swaps,
            )

            tour = result.tour
            last_improvement = result.improvement
            last_action = teacher_action

            if result.improved:
                no_improve_steps = 0
            else:
                no_improve_steps += 1

        if (instance_idx + 1) % 25 == 0:
            print(f"Built {instance_idx + 1}/{n_instances} teacher trajectories")

    X = torch.tensor(np.stack(X_rows), dtype=torch.float32)
    y = torch.tensor(np.asarray(y_rows), dtype=torch.long)

    dataset = TensorDataset(X, y)
    logs = pd.DataFrame(log_rows)

    return dataset, logs


# ============================================================
# Model
# ============================================================

class OperatorSelectorMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, n_actions: int):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@torch.no_grad()
def evaluate_action_accuracy(
    model: OperatorSelectorMLP,
    dataset: TensorDataset,
    device: torch.device,
    batch_size: int,
) -> float:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    correct = 0
    total = 0

    model.eval()

    for X, y in loader:
        X = X.to(device)
        y = y.to(device)

        logits = model(X)
        pred = logits.argmax(dim=1)

        correct += int((pred == y).sum().item())
        total += int(y.numel())

    return correct / max(total, 1)


# ============================================================
# Rollout evaluation
# ============================================================

@torch.no_grad()
def rollout_selector(
    model: OperatorSelectorMLP,
    distance_matrix: np.ndarray,
    seed: int,
    max_steps: int,
    two_opt_iterations: int,
    max_trials: int,
    perturb_swaps: int,
    device: torch.device,
) -> dict:
    tour = construct_tour_v3(
        distance_matrix=distance_matrix,
        method="nearest_neighbor",
        seed=seed,
        n_starts=1,
    )
    tour = list(map(int, tour))

    initial_length = tour_length(np.asarray(tour, dtype=int), distance_matrix)

    last_improvement = 0.0
    no_improve_steps = 0
    last_action = 0

    action_counts = {name: 0 for name in ACTION_NAMES.values()}

    start_time = time.perf_counter()

    for step in range(max_steps):
        features = build_state_features(
            tour=tour,
            distance_matrix=distance_matrix,
            initial_length=initial_length,
            current_step=step,
            max_steps=max_steps,
            last_improvement=last_improvement,
            no_improve_steps=no_improve_steps,
            last_action=last_action,
        )

        X = torch.tensor(features[None, :], dtype=torch.float32).to(device)
        logits = model(X)
        action = int(logits.argmax(dim=1).item())

        action_name = ACTION_NAMES[action]
        action_counts[action_name] += 1

        if action == 0:
            break

        result = apply_operator(
            action=action,
            tour=tour,
            distance_matrix=distance_matrix,
            seed=seed + step * 100_000 + action * 1009,
            two_opt_iterations=two_opt_iterations,
            max_trials=max_trials,
            perturb_swaps=perturb_swaps,
        )

        tour = result.tour
        last_improvement = result.improvement
        last_action = action

        if result.improved:
            no_improve_steps = 0
        else:
            no_improve_steps += 1

    runtime_sec = time.perf_counter() - start_time

    final_length = tour_length(np.asarray(tour, dtype=int), distance_matrix)

    return {
        "initial_length": initial_length,
        "final_length": final_length,
        "improvement": initial_length - final_length,
        "improvement_percent": (initial_length - final_length) / max(initial_length, 1e-12) * 100.0,
        "runtime_sec": runtime_sec,
        "runtime_ms": runtime_sec * 1000.0,
        "steps_used": sum(action_counts.values()),
        **{f"count_{k}": v for k, v in action_counts.items()},
    }


def evaluate_fixed_strategy(
    distance_matrix: np.ndarray,
    seed: int,
    strategy: str,
    max_steps: int,
    two_opt_iterations: int,
    max_trials: int,
    perturb_swaps: int,
) -> dict:
    tour = construct_tour_v3(
        distance_matrix=distance_matrix,
        method="nearest_neighbor",
        seed=seed,
        n_starts=1,
    )
    tour = list(map(int, tour))

    initial_length = tour_length(np.asarray(tour, dtype=int), distance_matrix)

    start_time = time.perf_counter()

    if strategy == "nn_only":
        final_tour = tour

    elif strategy == "nn_2opt":
        final_tour, _, _ = two_opt_local_search_limited(
            np.asarray(tour, dtype=int),
            distance_matrix,
            max_iterations=two_opt_iterations,
        )
        final_tour = list(map(int, final_tour))

    elif strategy == "cycle":
        final_tour = tour
        actions = [1, 2, 3, 4, 5]

        for step in range(max_steps):
            action = actions[step % len(actions)]
            result = apply_operator(
                action=action,
                tour=final_tour,
                distance_matrix=distance_matrix,
                seed=seed + step * 100_000 + action * 1009,
                two_opt_iterations=two_opt_iterations,
                max_trials=max_trials,
                perturb_swaps=perturb_swaps,
            )
            final_tour = result.tour

    elif strategy == "random_operator":
        final_tour = tour
        rng = np.random.default_rng(seed)

        for step in range(max_steps):
            action = int(rng.choice([1, 2, 3, 4, 5]))

            result = apply_operator(
                action=action,
                tour=final_tour,
                distance_matrix=distance_matrix,
                seed=seed + step * 100_000 + action * 1009,
                two_opt_iterations=two_opt_iterations,
                max_trials=max_trials,
                perturb_swaps=perturb_swaps,
            )
            final_tour = result.tour

    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    runtime_sec = time.perf_counter() - start_time

    final_length = tour_length(np.asarray(final_tour, dtype=int), distance_matrix)

    return {
        "initial_length": initial_length,
        "final_length": final_length,
        "improvement": initial_length - final_length,
        "improvement_percent": (initial_length - final_length) / max(initial_length, 1e-12) * 100.0,
        "runtime_sec": runtime_sec,
        "runtime_ms": runtime_sec * 1000.0,
    }


def evaluate_on_generated_instances(
    model: OperatorSelectorMLP,
    n_city_choices: list[int],
    n_instances: int,
    seed_offset: int,
    max_steps: int,
    two_opt_iterations: int,
    max_trials: int,
    perturb_swaps: int,
    device: torch.device,
) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(seed_offset)

    for instance_idx in range(n_instances):
        n_cities = int(rng.choice(n_city_choices))
        seed = seed_offset + instance_idx

        instance = generate_euclidean_instance(n_cities=n_cities, seed=seed)
        coords = normalize_coords(instance.coords)
        distance_matrix = compute_distance_matrix(coords)

        method_results = {}

        selector_result = rollout_selector(
            model=model,
            distance_matrix=distance_matrix,
            seed=seed,
            max_steps=max_steps,
            two_opt_iterations=two_opt_iterations,
            max_trials=max_trials,
            perturb_swaps=perturb_swaps,
            device=device,
        )
        method_results["bc_operator_selector"] = selector_result

        for strategy in ["nn_only", "nn_2opt", "cycle", "random_operator"]:
            method_results[strategy] = evaluate_fixed_strategy(
                distance_matrix=distance_matrix,
                seed=seed,
                strategy=strategy,
                max_steps=max_steps,
                two_opt_iterations=two_opt_iterations,
                max_trials=max_trials,
                perturb_swaps=perturb_swaps,
            )

        for method, result in method_results.items():
            rows.append(
                {
                    "seed": seed,
                    "n_cities": n_cities,
                    "method": method,
                    **result,
                }
            )

    return pd.DataFrame(rows)


def summarize_eval(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["n_cities", "method"])
        .agg(
            n_cases=("final_length", "count"),
            mean_final_length=("final_length", "mean"),
            mean_improvement_percent=("improvement_percent", "mean"),
            mean_runtime_ms=("runtime_ms", "mean"),
        )
        .reset_index()
        .sort_values(["n_cities", "mean_final_length"])
    )


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--n-city-choices", type=str, default="50,70,100,130")
    parser.add_argument("--train-instances", type=int, default=300)
    parser.add_argument("--test-instances", type=int, default=60)
    parser.add_argument("--train-seed-offset", type=int, default=70_000)
    parser.add_argument("--test-seed-offset", type=int, default=90_000)

    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--two-opt-iterations", type=int, default=50)
    parser.add_argument("--max-trials", type=int, default=100)
    parser.add_argument("--perturb-swaps", type=int, default=3)

    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)

    parser.add_argument("--out-dir", type=str, default="results/operator_selector_bc")

    args = parser.parse_args()

    n_city_choices = parse_int_list(args.n_city_choices)
    max_n = max(n_city_choices)

    out_dir = Path(args.out_dir) / f"nmax{max_n}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Building training teacher dataset")
    train_dataset, train_teacher_log = build_teacher_dataset(
        n_city_choices=n_city_choices,
        n_instances=args.train_instances,
        seed_offset=args.train_seed_offset,
        max_steps=args.max_steps,
        two_opt_iterations=args.two_opt_iterations,
        max_trials=args.max_trials,
        perturb_swaps=args.perturb_swaps,
    )

    print("\nBuilding test teacher dataset")
    test_dataset, test_teacher_log = build_teacher_dataset(
        n_city_choices=n_city_choices,
        n_instances=args.test_instances,
        seed_offset=args.test_seed_offset,
        max_steps=args.max_steps,
        two_opt_iterations=args.two_opt_iterations,
        max_trials=args.max_trials,
        perturb_swaps=args.perturb_swaps,
    )

    train_teacher_log.to_csv(out_dir / "train_teacher_log.csv", index=False)
    test_teacher_log.to_csv(out_dir / "test_teacher_log.csv", index=False)

    print("\nTeacher action distribution:")
    print(train_teacher_log["teacher_action_name"].value_counts().to_string())

    input_dim = train_dataset.tensors[0].shape[1]
    n_actions = len(ACTION_NAMES)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = OperatorSelectorMLP(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        n_actions=n_actions,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
    )

    logs = []

    for epoch in range(1, args.epochs + 1):
        model.train()

        total_loss = 0.0
        n_batches = 0

        for X, y in train_loader:
            X = X.to(device)
            y = y.to(device)

            logits = model(X)
            loss = F.cross_entropy(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())
            n_batches += 1

        train_acc = evaluate_action_accuracy(
            model=model,
            dataset=train_dataset,
            device=device,
            batch_size=args.batch_size,
        )

        test_acc = evaluate_action_accuracy(
            model=model,
            dataset=test_dataset,
            device=device,
            batch_size=args.batch_size,
        )

        row = {
            "epoch": epoch,
            "loss": total_loss / max(n_batches, 1),
            "train_acc": train_acc,
            "test_acc": test_acc,
        }
        logs.append(row)

        print(
            f"Epoch {epoch:03d} | "
            f"loss={row['loss']:.4f} | "
            f"train_acc={train_acc:.3f} | "
            f"test_acc={test_acc:.3f}"
        )

    torch.save(model.state_dict(), out_dir / "operator_selector_bc.pt")
    pd.DataFrame(logs).to_csv(out_dir / "training_log.csv", index=False)

    print("\nEvaluating rollout on generated test instances")
    eval_raw = evaluate_on_generated_instances(
        model=model,
        n_city_choices=n_city_choices,
        n_instances=args.test_instances,
        seed_offset=args.test_seed_offset + 10_000,
        max_steps=args.max_steps,
        two_opt_iterations=args.two_opt_iterations,
        max_trials=args.max_trials,
        perturb_swaps=args.perturb_swaps,
        device=device,
    )

    eval_summary = summarize_eval(eval_raw)

    eval_raw.to_csv(out_dir / "generated_eval_raw.csv", index=False)
    eval_summary.to_csv(out_dir / "generated_eval_summary.csv", index=False)

    print("\nGenerated evaluation summary:")
    print(eval_summary.to_string(index=False))

    print(f"\nSaved model and results to: {out_dir}")


if __name__ == "__main__":
    main()