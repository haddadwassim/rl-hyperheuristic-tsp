import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from tsp_hh.instances import generate_euclidean_instance
from tsp_hh.bc_policy import NextCityPolicy
from tsp_hh.tour import tour_length
from tsp_hh.heuristics_v3 import construct_tour_v3, two_opt_local_search_limited

from train_behavior_cloning_exact import (
    normalize_coords,
    compute_distance_matrix,
    build_candidate_features,
    rollout_policy,
    repair_tour_with_2opt,
    masked_cross_entropy,
)


def parse_int_list(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def build_teacher_tour(
    distance_matrix: np.ndarray,
    seed: int,
    teacher_starts: int,
    teacher_2opt_iterations: int,
) -> tuple[list[int], float]:
    """
    Teacher = greedy multi-start construction + 2-opt repair.
    This is not exact optimal, but it gives scalable demonstrations.
    """
    tour = construct_tour_v3(
        distance_matrix=distance_matrix,
        method="greedy",
        seed=seed,
        n_starts=teacher_starts,
    )

    tour, _, _ = two_opt_local_search_limited(
        np.asarray(tour, dtype=int),
        distance_matrix,
        max_iterations=teacher_2opt_iterations,
    )

    tour = list(map(int, tour))
    length = tour_length(np.asarray(tour), distance_matrix)

    return tour, float(length)


def build_examples_from_teacher_tour(
    coords: np.ndarray,
    distance_matrix: np.ndarray,
    teacher_tour: list[int],
    max_n_cities: int,
    start_city: int = 0,
) -> tuple[list[np.ndarray], list[np.ndarray], list[int]]:
    """
    Convert one teacher tour into state -> next-city examples.

    Because we train on several problem sizes, each example is padded to
    max_n_cities. The valid mask marks real unvisited cities only.
    """
    n_cities = coords.shape[0]

    teacher_tour = list(map(int, teacher_tour))

    if len(teacher_tour) != n_cities:
        raise ValueError(
            f"Invalid teacher tour length: expected {n_cities}, got {len(teacher_tour)}"
        )

    if sorted(teacher_tour) != list(range(n_cities)):
        raise ValueError("Teacher tour is not a valid permutation")

    if start_city not in teacher_tour:
        raise ValueError(f"start_city={start_city} not found in teacher tour")

    # Important:
    # construct_tour_v3 may return a valid TSP cycle starting from any city.
    # Our learning examples always start from start_city=0, so rotate the
    # cyclic tour representation to start at 0.
    if teacher_tour[0] != start_city:
        start_pos = teacher_tour.index(start_city)
        teacher_tour = teacher_tour[start_pos:] + teacher_tour[:start_pos]

    X = []
    masks = []
    y = []

    visited = np.zeros(n_cities, dtype=np.float32)
    visited[start_city] = 1.0

    for step in range(1, n_cities):
        current_city = teacher_tour[step - 1]
        target_city = teacher_tour[step]

        features = build_candidate_features(
            coords=coords,
            distance_matrix=distance_matrix,
            current_city=current_city,
            start_city=start_city,
            visited=visited,
        )

        valid_mask = visited == 0

        padded_features = np.zeros((max_n_cities, features.shape[1]), dtype=np.float32)
        padded_mask = np.zeros(max_n_cities, dtype=bool)

        padded_features[:n_cities] = features
        padded_mask[:n_cities] = valid_mask

        X.append(padded_features)
        masks.append(padded_mask)
        y.append(target_city)

        visited[target_city] = 1.0

    return X, masks, y


def build_heuristic_teacher_dataset(
    n_city_choices: list[int],
    n_instances: int,
    seed_offset: int,
    teacher_starts: int,
    teacher_2opt_iterations: int,
    max_n_cities: int,
) -> tuple[TensorDataset, list[dict]]:
    X_all = []
    masks_all = []
    y_all = []
    metas = []

    rng = np.random.default_rng(seed_offset)

    for instance_idx in range(n_instances):
        n_cities = int(rng.choice(n_city_choices))
        seed = seed_offset + instance_idx

        instance = generate_euclidean_instance(
            n_cities=n_cities,
            seed=seed,
        )

        coords = normalize_coords(instance.coords)
        distance_matrix = compute_distance_matrix(coords)

        start_time = time.perf_counter()

        teacher_tour, teacher_length = build_teacher_tour(
            distance_matrix=distance_matrix,
            seed=seed,
            teacher_starts=teacher_starts,
            teacher_2opt_iterations=teacher_2opt_iterations,
        )

        teacher_time = time.perf_counter() - start_time

        X, masks, y = build_examples_from_teacher_tour(
            coords=coords,
            distance_matrix=distance_matrix,
            teacher_tour=teacher_tour,
            max_n_cities=max_n_cities,
            start_city=0,
        )

        X_all.extend(X)
        masks_all.extend(masks)
        y_all.extend(y)

        metas.append(
            {
                "seed": seed,
                "n_cities": n_cities,
                "coords": coords,
                "distance_matrix": distance_matrix,
                "teacher_tour": teacher_tour,
                "teacher_length": teacher_length,
                "teacher_time_sec": teacher_time,
            }
        )

        if (instance_idx + 1) % 25 == 0:
            print(
                f"Built {instance_idx + 1}/{n_instances} teacher instances | "
                f"last_n={n_cities} | "
                f"last_teacher_len={teacher_length:.3f} | "
                f"teacher_time={teacher_time:.3f}s"
            )

    X_tensor = torch.tensor(np.stack(X_all), dtype=torch.float32)
    masks_tensor = torch.tensor(np.stack(masks_all), dtype=torch.bool)
    y_tensor = torch.tensor(np.asarray(y_all), dtype=torch.long)

    dataset = TensorDataset(X_tensor, masks_tensor, y_tensor)

    return dataset, metas


def sanity_check_dataset(
    dataset: TensorDataset,
    name: str,
    max_print: int = 10,
) -> None:
    X, masks, y = dataset.tensors

    print(f"\nSanity checking dataset: {name}")
    print(f"X shape: {tuple(X.shape)}")
    print(f"masks shape: {tuple(masks.shape)}")
    print(f"y shape: {tuple(y.shape)}")

    n_examples = y.shape[0]

    # Check target range.
    target_in_range = (y >= 0) & (y < masks.shape[1])
    n_bad_range = int((~target_in_range).sum().item())

    print(f"Targets out of range: {n_bad_range}/{n_examples}")

    # Critical check: target must be valid under the action mask.
    row_ids = torch.arange(n_examples)
    target_valid = masks[row_ids, y]
    n_invalid_targets = int((~target_valid).sum().item())

    print(f"Targets masked invalid: {n_invalid_targets}/{n_examples}")

    if n_invalid_targets > 0:
        bad_indices = torch.where(~target_valid)[0][:max_print]

        print("\nExamples with invalid target:")
        for idx in bad_indices.tolist():
            print(
                f"idx={idx} | "
                f"target={int(y[idx].item())} | "
                f"mask_target={bool(masks[idx, y[idx]].item())} | "
                f"valid_actions={torch.where(masks[idx])[0].tolist()}"
            )

    # Check padded actions are invalid.
    valid_counts = masks.sum(dim=1)
    print(f"Valid action count: min={int(valid_counts.min())}, max={int(valid_counts.max())}")

    if n_bad_range > 0 or n_invalid_targets > 0:
        raise RuntimeError(f"Dataset sanity check failed for {name}")

    print(f"Dataset sanity check passed: {name}")


@torch.no_grad()
def evaluate_next_city_accuracy(
    model: NextCityPolicy,
    dataset: TensorDataset,
    device: torch.device,
    batch_size: int,
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


@torch.no_grad()
def evaluate_rollout_vs_teacher(
    model: NextCityPolicy,
    metas: list[dict],
    device: torch.device,
    two_opt_iterations: int,
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

        raw_time = time.perf_counter() - start_time

        start_time = time.perf_counter()

        repaired_path, repaired_length = repair_tour_with_2opt(
            path=path,
            distance_matrix=meta["distance_matrix"],
            max_iterations=two_opt_iterations,
        )

        repair_time = time.perf_counter() - start_time

        teacher_length = meta["teacher_length"]

        rows.append(
            {
                "seed": meta["seed"],
                "n_cities": meta["n_cities"],
                "teacher_length": teacher_length,
                "bc_length": length,
                "bc_2opt_length": repaired_length,
                "bc_gap_to_teacher_percent": (length - teacher_length) / teacher_length * 100.0,
                "bc_2opt_gap_to_teacher_percent": (repaired_length - teacher_length) / teacher_length * 100.0,
                "raw_runtime_ms": raw_time * 1000.0,
                "repair_runtime_ms": repair_time * 1000.0,
                "total_runtime_ms": (raw_time + repair_time) * 1000.0,
                "matches_teacher_path": path == meta["teacher_tour"],
            }
        )

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--n-city-choices", type=str, default="50,70,100,130")
    parser.add_argument("--train-instances", type=int, default=300)
    parser.add_argument("--test-instances", type=int, default=60)
    parser.add_argument("--train-seed-offset", type=int, default=20_000)
    parser.add_argument("--test-seed-offset", type=int, default=50_000)

    parser.add_argument("--teacher-starts", type=int, default=10)
    parser.add_argument("--teacher-2opt-iterations", type=int, default=300)

    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)

    parser.add_argument("--eval-2opt-iterations", type=int, default=300)

    parser.add_argument("--out-dir", type=str, default="results/behavior_cloning_heuristic_teacher")

    args = parser.parse_args()

    n_city_choices = parse_int_list(args.n_city_choices)
    max_n_cities = max(n_city_choices)

    out_dir = Path(args.out_dir) / f"nmax{max_n_cities}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Building heuristic-teacher training dataset")
    train_dataset, train_metas = build_heuristic_teacher_dataset(
        n_city_choices=n_city_choices,
        n_instances=args.train_instances,
        seed_offset=args.train_seed_offset,
        teacher_starts=args.teacher_starts,
        teacher_2opt_iterations=args.teacher_2opt_iterations,
        max_n_cities=max_n_cities,
    )
    sanity_check_dataset(train_dataset, "train")

    print("\nBuilding heuristic-teacher test dataset")
    test_dataset, test_metas = build_heuristic_teacher_dataset(
        n_city_choices=n_city_choices,
        n_instances=args.test_instances,
        seed_offset=args.test_seed_offset,
        teacher_starts=args.teacher_starts,
        teacher_2opt_iterations=args.teacher_2opt_iterations,
        max_n_cities=max_n_cities,
    )
    sanity_check_dataset(test_dataset, "test")

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

        train_acc = evaluate_next_city_accuracy(
            model=model,
            dataset=train_dataset,
            device=device,
            batch_size=args.batch_size,
        )

        test_acc = evaluate_next_city_accuracy(
            model=model,
            dataset=test_dataset,
            device=device,
            batch_size=args.batch_size,
        )

        # Keep rollout evaluation small during training.
        test_rollout = evaluate_rollout_vs_teacher(
            model=model,
            metas=test_metas[:10],
            device=device,
            two_opt_iterations=args.eval_2opt_iterations,
        )

        row = {
            "epoch": epoch,
            "loss": total_loss / max(n_batches, 1),
            "train_next_city_acc": train_acc,
            "test_next_city_acc": test_acc,
            "test_bc_2opt_gap_to_teacher": test_rollout["bc_2opt_gap_to_teacher_percent"].mean(),
        }

        logs.append(row)

        print(
            f"Epoch {epoch:03d} | "
            f"loss={row['loss']:.4f} | "
            f"train_acc={train_acc:.3f} | "
            f"test_acc={test_acc:.3f} | "
            f"bc2opt_vs_teacher={row['test_bc_2opt_gap_to_teacher']:.2f}%"
        )

    torch.save(model.state_dict(), out_dir / "bc_policy_heuristic_teacher.pt")
    pd.DataFrame(logs).to_csv(out_dir / "training_log.csv", index=False)

    final_eval = evaluate_rollout_vs_teacher(
        model=model,
        metas=test_metas,
        device=device,
        two_opt_iterations=args.eval_2opt_iterations,
    )

    final_eval.to_csv(out_dir / "test_rollout_vs_teacher.csv", index=False)

    print("\nFinal rollout vs heuristic teacher:")
    print(
        final_eval[
            [
                "n_cities",
                "bc_gap_to_teacher_percent",
                "bc_2opt_gap_to_teacher_percent",
                "total_runtime_ms",
            ]
        ]
        .groupby("n_cities")
        .mean()
        .reset_index()
        .to_string(index=False)
    )

    print(f"\nSaved model and results to: {out_dir}")


if __name__ == "__main__":
    main()