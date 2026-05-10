import argparse
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd

from tsp_hh.instances import generate_euclidean_instance
from tsp_hh.hyper_env import TSPHyperHeuristicEnv
from tsp_hh.q_learning import QLearningAgent, discretize_state


def load_q_learning_agent(path: Path, seed: int | None = None) -> QLearningAgent:
    """
    Load a trained Q-learning agent from a pickle file.
    """
    with path.open("rb") as f:
        payload = pickle.load(f)

    agent = QLearningAgent(
        n_actions=payload["n_actions"],
        learning_rate=payload["learning_rate"],
        discount_factor=payload["discount_factor"],
        epsilon=0.0,          # Important: no exploration during evaluation
        epsilon_min=0.0,
        epsilon_decay=1.0,
        seed=seed,
    )

    agent.q_table = payload["q_table"]

    return agent


def run_learned_hh_on_instance(
    agent: QLearningAgent,
    n_cities: int,
    seed: int,
    max_steps: int,
    initial_method: str,
) -> dict:
    """
    Evaluate the learned Q-learning hyper-heuristic on one TSP instance.
    """
    instance = generate_euclidean_instance(n_cities=n_cities, seed=seed)

    env = TSPHyperHeuristicEnv(
        distance_matrix=instance.distance_matrix,
        initial_method=initial_method,
        max_steps=max_steps,
        seed=seed,
    )

    state = env.reset()

    initial_length = env.best_length
    action_counts = {name: 0 for name in env.ACTION_NAMES.values()}

    start_time = time.perf_counter()

    done = False

    while not done:
        state_key = discretize_state(
            state=state,
            initial_length=initial_length,
            max_steps=max_steps,
        )

        action = agent.select_action(state_key, training=False)

        next_state, reward, done, info = env.step(action)

        action_counts[info["action_name"]] += 1
        state = next_state

    runtime_sec = time.perf_counter() - start_time

    result = {
        "method": "q_learning_hh",
        "n_cities": n_cities,
        "seed": seed,
        "initial_method": initial_method,
        "max_steps": max_steps,
        "initial_length": initial_length,
        "tour_length": env.best_length,
        "improvement": initial_length - env.best_length,
        "runtime_sec": runtime_sec,
    }

    for action_name, count in action_counts.items():
        result[f"count_{action_name}"] = count

    return result


def run_random_hh_on_instance(
    n_cities: int,
    seed: int,
    max_steps: int,
    initial_method: str,
) -> dict:
    """
    Run the random hyper-heuristic on one TSP instance.
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
    action_counts = {name: 0 for name in env.ACTION_NAMES.values()}

    start_time = time.perf_counter()

    done = False

    while not done:
        action = int(rng.integers(0, len(env.ACTION_NAMES)))

        _, _, done, info = env.step(action)

        action_counts[info["action_name"]] += 1

    runtime_sec = time.perf_counter() - start_time

    result = {
        "method": "random_hh",
        "n_cities": n_cities,
        "seed": seed,
        "initial_method": initial_method,
        "max_steps": max_steps,
        "initial_length": initial_length,
        "tour_length": env.best_length,
        "improvement": initial_length - env.best_length,
        "runtime_sec": runtime_sec,
    }

    for action_name, count in action_counts.items():
        result[f"count_{action_name}"] = count

    return result


def summarize_results(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby(["n_cities", "method"])
        .agg(
            mean_initial_length=("initial_length", "mean"),
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

    parser.add_argument(
        "--q-table-path",
        type=str,
        default="results/q_learning/q_table.pkl",
        help="Path to trained Q-table.",
    )

    parser.add_argument(
        "--n-cities",
        type=int,
        nargs="+",
        default=[20, 50, 100],
        help="Evaluation instance sizes.",
    )

    parser.add_argument(
        "--n-instances",
        type=int,
        default=30,
        help="Number of unseen instances per size.",
    )

    parser.add_argument(
        "--seed-offset",
        type=int,
        default=10_000,
        help="Offset used to create unseen evaluation seeds.",
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
        default="results/q_learning_eval",
        help="Output directory.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Agent evaluation seed.",
    )

    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    q_table_path = Path(args.q_table_path)

    agent = load_q_learning_agent(q_table_path, seed=args.seed)

    all_results = []

    for n_cities in args.n_cities:
        for i in range(args.n_instances):
            eval_seed = args.seed_offset + i

            q_row = run_learned_hh_on_instance(
                agent=agent,
                n_cities=n_cities,
                seed=eval_seed,
                max_steps=args.max_steps,
                initial_method=args.initial_method,
            )

            random_row = run_random_hh_on_instance(
                n_cities=n_cities,
                seed=eval_seed,
                max_steps=args.max_steps,
                initial_method=args.initial_method,
            )

            all_results.append(q_row)
            all_results.append(random_row)

            best_method = min(
                [q_row, random_row],
                key=lambda row: row["tour_length"],
            )["method"]

            print(
                f"Finished n_cities={n_cities}, seed={eval_seed}, "
                f"q_length={q_row['tour_length']:.3f}, "
                f"random_length={random_row['tour_length']:.3f}, "
                f"best={best_method}"
            )

    df = pd.DataFrame(all_results)
    summary = summarize_results(df)

    raw_path = out_dir / "q_learning_eval_raw.csv"
    summary_path = out_dir / "q_learning_eval_summary.csv"

    df.to_csv(raw_path, index=False)
    summary.to_csv(summary_path, index=False)

    print("\nEvaluation finished.")
    print(f"Raw results saved to: {raw_path}")
    print(f"Summary saved to: {summary_path}")

    print("\nSummary:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()