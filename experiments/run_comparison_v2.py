import argparse
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd

from tsp_hh.instances import generate_euclidean_instance
from tsp_hh.hyper_env_v2 import TSPHyperHeuristicEnvV2
from tsp_hh.q_learning_v2 import QLearningAgent, discretize_state_v2


def load_q_learning_agent_v2(path: Path, seed: int | None = None) -> QLearningAgent:
    with path.open("rb") as f:
        payload = pickle.load(f)

    agent = QLearningAgent(
        n_actions=payload["n_actions"],
        learning_rate=payload["learning_rate"],
        discount_factor=payload["discount_factor"],
        epsilon=0.0,
        epsilon_min=0.0,
        epsilon_decay=1.0,
        seed=seed,
    )

    agent.q_table = payload["q_table"]
    return agent


def run_policy_v2(
    instance,
    n_cities: int,
    seed: int,
    max_steps: int,
    initial_method: str,
    k_neighbors: int,
    perturbation_moves: int,
    reward_scale: str,
    accept_worse_current: bool,
    method_name: str,
    policy_type: str,
    fixed_action: int | None = None,
    agent: QLearningAgent | None = None,
) -> dict:
    env = TSPHyperHeuristicEnvV2(
        distance_matrix=instance.distance_matrix,
        initial_method=initial_method,
        max_steps=max_steps,
        seed=seed,
        k_neighbors=k_neighbors,
        perturbation_moves=perturbation_moves,
        reward_scale=reward_scale,
        accept_worse_current=accept_worse_current,
    )

    rng = np.random.default_rng(seed)
    state = env.reset()

    initial_length = env.best_length
    action_counts = {name: 0 for name in env.ACTION_NAMES.values()}

    start_time = time.perf_counter()

    done = False
    step = 0

    while not done:
        if policy_type == "random":
            action = int(rng.integers(0, len(env.ACTION_NAMES)))

        elif policy_type == "cycle":
            action = step % len(env.ACTION_NAMES)

        elif policy_type == "fixed":
            if fixed_action is None:
                raise ValueError("fixed_action must be provided for fixed policy")
            action = fixed_action

        elif policy_type == "q_learning":
            if agent is None:
                raise ValueError("agent must be provided for q_learning policy")

            state_key = discretize_state_v2(
                state=state,
                initial_length=initial_length,
                max_steps=max_steps,
            )
            action = agent.select_action(state_key, training=False)

        else:
            raise ValueError(f"Unknown policy_type: {policy_type}")

        next_state, reward, done, info = env.step(action)

        action_counts[info["action_name"]] += 1
        state = next_state
        step += 1

    runtime_sec = time.perf_counter() - start_time

    row = {
        "method": method_name,
        "n_cities": n_cities,
        "seed": seed,
        "initial_method": initial_method,
        "max_steps": max_steps,
        "k_neighbors": k_neighbors,
        "perturbation_moves": perturbation_moves,
        "reward_scale": reward_scale,
        "accept_worse_current": accept_worse_current,
        "initial_length": initial_length,
        "tour_length": env.best_length,
        "improvement": initial_length - env.best_length,
        "runtime_sec": runtime_sec,
    }

    for action_name, count in action_counts.items():
        row[f"count_{action_name}"] = count

    return row


def summarize_results(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby(["n_cities", "method"])
        .agg(
            mean_length=("tour_length", "mean"),
            std_length=("tour_length", "std"),
            best_length=("tour_length", "min"),
            worst_length=("tour_length", "max"),
            mean_improvement=("improvement", "mean"),
            mean_runtime_sec=("runtime_sec", "mean"),
        )
        .reset_index()
        .sort_values(["n_cities", "mean_length"])
    )

    best_by_size = summary.groupby("n_cities")["mean_length"].transform("min")

    summary["gap_to_best_observed_percent"] = (
        (summary["mean_length"] - best_by_size) / best_by_size * 100.0
    )

    return summary


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--q-table-path", type=str, default="results/q_learning_v2/q_table_v2.pkl")
    parser.add_argument("--n-cities", type=int, nargs="+", default=[20, 50, 100])
    parser.add_argument("--n-instances", type=int, default=30)
    parser.add_argument("--seed-offset", type=int, default=10_000)

    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--initial-method", type=str, default="nearest_neighbor", choices=["nearest_neighbor", "random"])
    parser.add_argument("--k-neighbors", type=int, default=10)
    parser.add_argument("--perturbation-moves", type=int, default=3)
    parser.add_argument("--reward-scale", type=str, default="initial_length", choices=["initial_length", "none"])
    parser.add_argument("--accept-worse-current", action="store_true")

    parser.add_argument("--agent-seed", type=int, default=42)
    parser.add_argument("--out-dir", type=str, default="results/comparison_v2")

    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    q_agent = load_q_learning_agent_v2(Path(args.q_table_path), seed=args.agent_seed)

    all_rows = []

    for n_cities in args.n_cities:
        for i in range(args.n_instances):
            seed = args.seed_offset + i
            instance = generate_euclidean_instance(n_cities=n_cities, seed=seed)

            rows = [
                run_policy_v2(
                    instance=instance,
                    n_cities=n_cities,
                    seed=seed,
                    max_steps=args.max_steps,
                    initial_method=args.initial_method,
                    k_neighbors=args.k_neighbors,
                    perturbation_moves=args.perturbation_moves,
                    reward_scale=args.reward_scale,
                    accept_worse_current=args.accept_worse_current,
                    method_name="random_hh_v2",
                    policy_type="random",
                ),
                run_policy_v2(
                    instance=instance,
                    n_cities=n_cities,
                    seed=seed,
                    max_steps=args.max_steps,
                    initial_method=args.initial_method,
                    k_neighbors=args.k_neighbors,
                    perturbation_moves=args.perturbation_moves,
                    reward_scale=args.reward_scale,
                    accept_worse_current=args.accept_worse_current,
                    method_name="cycle_hh_v2",
                    policy_type="cycle",
                ),
                run_policy_v2(
                    instance=instance,
                    n_cities=n_cities,
                    seed=seed,
                    max_steps=args.max_steps,
                    initial_method=args.initial_method,
                    k_neighbors=args.k_neighbors,
                    perturbation_moves=args.perturbation_moves,
                    reward_scale=args.reward_scale,
                    accept_worse_current=args.accept_worse_current,
                    method_name="always_first_2opt_v2",
                    policy_type="fixed",
                    fixed_action=0,
                ),
                run_policy_v2(
                    instance=instance,
                    n_cities=n_cities,
                    seed=seed,
                    max_steps=args.max_steps,
                    initial_method=args.initial_method,
                    k_neighbors=args.k_neighbors,
                    perturbation_moves=args.perturbation_moves,
                    reward_scale=args.reward_scale,
                    accept_worse_current=args.accept_worse_current,
                    method_name="always_random_2opt_v2",
                    policy_type="fixed",
                    fixed_action=1,
                ),
                run_policy_v2(
                    instance=instance,
                    n_cities=n_cities,
                    seed=seed,
                    max_steps=args.max_steps,
                    initial_method=args.initial_method,
                    k_neighbors=args.k_neighbors,
                    perturbation_moves=args.perturbation_moves,
                    reward_scale=args.reward_scale,
                    accept_worse_current=args.accept_worse_current,
                    method_name="always_bounded_2opt_v2",
                    policy_type="fixed",
                    fixed_action=2,
                ),
                run_policy_v2(
                    instance=instance,
                    n_cities=n_cities,
                    seed=seed,
                    max_steps=args.max_steps,
                    initial_method=args.initial_method,
                    k_neighbors=args.k_neighbors,
                    perturbation_moves=args.perturbation_moves,
                    reward_scale=args.reward_scale,
                    accept_worse_current=args.accept_worse_current,
                    method_name="always_bounded_swap_v2",
                    policy_type="fixed",
                    fixed_action=3,
                ),
                run_policy_v2(
                    instance=instance,
                    n_cities=n_cities,
                    seed=seed,
                    max_steps=args.max_steps,
                    initial_method=args.initial_method,
                    k_neighbors=args.k_neighbors,
                    perturbation_moves=args.perturbation_moves,
                    reward_scale=args.reward_scale,
                    accept_worse_current=args.accept_worse_current,
                    method_name="always_bounded_insertion_v2",
                    policy_type="fixed",
                    fixed_action=4,
                ),
                run_policy_v2(
                    instance=instance,
                    n_cities=n_cities,
                    seed=seed,
                    max_steps=args.max_steps,
                    initial_method=args.initial_method,
                    k_neighbors=args.k_neighbors,
                    perturbation_moves=args.perturbation_moves,
                    reward_scale=args.reward_scale,
                    accept_worse_current=args.accept_worse_current,
                    method_name="always_perturb_bounded_2opt_v2",
                    policy_type="fixed",
                    fixed_action=5,
                ),
                run_policy_v2(
                    instance=instance,
                    n_cities=n_cities,
                    seed=seed,
                    max_steps=args.max_steps,
                    initial_method=args.initial_method,
                    k_neighbors=args.k_neighbors,
                    perturbation_moves=args.perturbation_moves,
                    reward_scale=args.reward_scale,
                    accept_worse_current=args.accept_worse_current,
                    method_name="q_learning_hh_v2",
                    policy_type="q_learning",
                    agent=q_agent,
                ),
            ]

            all_rows.extend(rows)

            best_row = min(rows, key=lambda row: row["tour_length"])

            print(
                f"Finished n={n_cities}, seed={seed}, "
                f"best={best_row['method']}, length={best_row['tour_length']:.3f}"
            )

    raw = pd.DataFrame(all_rows)
    summary = summarize_results(raw)

    raw_path = out_dir / "comparison_v2_raw.csv"
    summary_path = out_dir / "comparison_v2_summary.csv"

    raw.to_csv(raw_path, index=False)
    summary.to_csv(summary_path, index=False)

    print("\nV2 comparison finished.")
    print(f"Raw results saved to: {raw_path}")
    print(f"Summary saved to: {summary_path}")

    print("\nSummary:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()