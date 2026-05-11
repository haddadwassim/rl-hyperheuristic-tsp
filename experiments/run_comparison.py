import argparse
import time
from pathlib import Path
import pickle

import numpy as np
import pandas as pd

from tsp_hh.instances import generate_euclidean_instance
from tsp_hh.tour import tour_length, random_tour
from tsp_hh.heuristics import nearest_neighbor_tour, repeated_two_opt
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
        epsilon=0.0,
        epsilon_min=0.0,
        epsilon_decay=1.0,
        seed=seed,
    )

    agent.q_table = payload["q_table"]

    return agent


def run_random_baseline(instance, n_cities: int, seed: int) -> dict:
    distance_matrix = instance.distance_matrix

    start_time = time.perf_counter()

    tour = random_tour(n_cities=n_cities, seed=seed)
    length = tour_length(tour, distance_matrix)

    runtime_sec = time.perf_counter() - start_time

    return {
        "method": "random",
        "n_cities": n_cities,
        "seed": seed,
        "tour_length": length,
        "improvement": 0.0,
        "runtime_sec": runtime_sec,
    }


def run_q_learning_hh_baseline(
    instance,
    n_cities: int,
    seed: int,
    hh_max_steps: int,
    initial_method: str,
    agent: QLearningAgent,
) -> dict:
    env = TSPHyperHeuristicEnv(
        distance_matrix=instance.distance_matrix,
        initial_method=initial_method,
        max_steps=hh_max_steps,
        seed=seed,
    )

    state = env.reset()

    initial_length = env.best_length
    action_counts = {action_name: 0 for action_name in env.ACTION_NAMES.values()}

    start_time = time.perf_counter()

    done = False

    while not done:
        state_key = discretize_state(
            state=state,
            initial_length=initial_length,
            max_steps=hh_max_steps,
        )

        action = agent.select_action(state_key, training=False)

        next_state, _, done, info = env.step(action)

        action_counts[info["action_name"]] += 1
        state = next_state

    runtime_sec = time.perf_counter() - start_time

    result = {
        "method": "q_learning_hh",
        "n_cities": n_cities,
        "seed": seed,
        "tour_length": env.best_length,
        "improvement": initial_length - env.best_length,
        "hh_steps": hh_max_steps,
        "runtime_sec": runtime_sec,
    }

    for action_name, count in action_counts.items():
        result[f"count_{action_name}"] = count

    return result


def run_nearest_neighbor_baseline(instance, n_cities: int, seed: int) -> dict:
    distance_matrix = instance.distance_matrix

    start_time = time.perf_counter()

    tour = nearest_neighbor_tour(distance_matrix, start_city=0)
    length = tour_length(tour, distance_matrix)

    runtime_sec = time.perf_counter() - start_time

    return {
        "method": "nearest_neighbor",
        "n_cities": n_cities,
        "seed": seed,
        "tour_length": length,
        "improvement": 0.0,
        "runtime_sec": runtime_sec,
    }


def run_random_2opt_baseline(
    instance,
    n_cities: int,
    seed: int,
    two_opt_max_iterations: int,
) -> dict:
    distance_matrix = instance.distance_matrix

    start_time = time.perf_counter()

    initial_tour = random_tour(n_cities=n_cities, seed=seed)
    initial_length = tour_length(initial_tour, distance_matrix)

    improved_tour, _, n_iterations = repeated_two_opt(
        initial_tour,
        distance_matrix,
        max_iterations=two_opt_max_iterations,
        use_best_improvement=False,
    )

    final_length = tour_length(improved_tour, distance_matrix)
    runtime_sec = time.perf_counter() - start_time

    return {
        "method": "random_2opt",
        "n_cities": n_cities,
        "seed": seed,
        "tour_length": final_length,
        "improvement": initial_length - final_length,
        "two_opt_iterations": n_iterations,
        "runtime_sec": runtime_sec,
    }


def run_nearest_neighbor_2opt_baseline(
    instance,
    n_cities: int,
    seed: int,
    two_opt_max_iterations: int,
) -> dict:
    distance_matrix = instance.distance_matrix

    start_time = time.perf_counter()

    initial_tour = nearest_neighbor_tour(distance_matrix, start_city=0)
    initial_length = tour_length(initial_tour, distance_matrix)

    improved_tour, _, n_iterations = repeated_two_opt(
        initial_tour,
        distance_matrix,
        max_iterations=two_opt_max_iterations,
        use_best_improvement=False,
    )

    final_length = tour_length(improved_tour, distance_matrix)
    runtime_sec = time.perf_counter() - start_time

    return {
        "method": "nearest_neighbor_2opt",
        "n_cities": n_cities,
        "seed": seed,
        "tour_length": final_length,
        "improvement": initial_length - final_length,
        "two_opt_iterations": n_iterations,
        "runtime_sec": runtime_sec,
    }


def run_random_hh_baseline(
    instance,
    n_cities: int,
    seed: int,
    hh_max_steps: int,
    initial_method: str,
) -> dict:
    env = TSPHyperHeuristicEnv(
        distance_matrix=instance.distance_matrix,
        initial_method=initial_method,
        max_steps=hh_max_steps,
        seed=seed,
    )

    rng = np.random.default_rng(seed)

    initial_length = env.best_length
    action_counts = {action_name: 0 for action_name in env.ACTION_NAMES.values()}

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
        "tour_length": env.best_length,
        "improvement": initial_length - env.best_length,
        "hh_steps": hh_max_steps,
        "runtime_sec": runtime_sec,
    }

    for action_name, count in action_counts.items():
        result[f"count_{action_name}"] = count

    return result

def run_fixed_hh_baseline(
    instance,
    n_cities: int,
    seed: int,
    hh_max_steps: int,
    initial_method: str,
    fixed_action: int | None,
    method_name: str,
) -> dict:
    """
    Run a fixed-policy hyper-heuristic baseline.

    If fixed_action is an integer, the same action is selected at every step.
    If fixed_action is None, actions are cycled in order: 0, 1, 2, 3, 4, ...
    """
    env = TSPHyperHeuristicEnv(
        distance_matrix=instance.distance_matrix,
        initial_method=initial_method,
        max_steps=hh_max_steps,
        seed=seed,
    )

    initial_length = env.best_length
    action_counts = {action_name: 0 for action_name in env.ACTION_NAMES.values()}

    start_time = time.perf_counter()

    done = False
    step = 0

    while not done:
        if fixed_action is None:
            action = step % len(env.ACTION_NAMES)
        else:
            action = fixed_action

        _, _, done, info = env.step(action)

        action_counts[info["action_name"]] += 1
        step += 1

    runtime_sec = time.perf_counter() - start_time

    result = {
        "method": method_name,
        "n_cities": n_cities,
        "seed": seed,
        "tour_length": env.best_length,
        "improvement": initial_length - env.best_length,
        "hh_steps": hh_max_steps,
        "runtime_sec": runtime_sec,
    }

    for action_name, count in action_counts.items():
        result[f"count_{action_name}"] = count

    return result


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

    return summary


def add_relative_gap(summary: pd.DataFrame) -> pd.DataFrame:
    """
    Add relative gap to the best method found in each n_cities group.

    Since we do not yet have optimal solutions, this is not an optimality gap.
    It is only a relative gap to the best observed method in our experiment.
    """
    summary = summary.copy()

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
        default=None,
        help="Optional path to a trained Q-table. If provided, q_learning_hh is included.",
    )

    parser.add_argument(
        "--agent-seed",
        type=int,
        default=42,
        help="Seed for the Q-learning agent during evaluation.",
    )

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
        "--hh-max-steps",
        type=int,
        default=50,
        help="Number of random hyper-heuristic decisions per instance.",
    )

    parser.add_argument(
        "--two-opt-long-max-iterations",
        type=int,
        default=1000,
        help="Maximum number of repeated 2-opt moves for long 2-opt baselines.",
    )

    parser.add_argument(
        "--hh-initial-method",
        type=str,
        default="nearest_neighbor",
        choices=["nearest_neighbor", "random"],
        help="Initial tour method for the random hyper-heuristic.",
    )

    parser.add_argument(
        "--out-dir",
        type=str,
        default="results/comparison",
        help="Directory where CSV results are saved.",
    )

    args = parser.parse_args()

    q_agent = None

    if args.q_table_path is not None:
        q_table_path = Path(args.q_table_path)

        if not q_table_path.exists():
            raise FileNotFoundError(f"Q-table not found: {q_table_path}")

        q_agent = load_q_learning_agent(q_table_path, seed=args.agent_seed)
        print(f"Loaded Q-learning agent from: {q_table_path}")
        print(f"Loaded Q-table states: {len(q_agent.q_table)}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = []

    methods = [
        "random",
        "nearest_neighbor",
        "random_2opt",
        "nearest_neighbor_2opt",
        "random_2opt_long",
        "nearest_neighbor_2opt_long",
        "random_hh",
        "always_first_2opt",
        "always_best_2opt",
        "always_random_swap",
        "always_random_insertion",
        "always_perturbation",
        "cycle_hh",
    ]

    if q_agent is not None:
        methods.append("q_learning_hh")

    for n_cities in args.n_cities:
        for seed in range(args.n_instances):
            instance = generate_euclidean_instance(n_cities=n_cities, seed=seed)

            rows = [
                run_random_baseline(instance, n_cities, seed),
                run_nearest_neighbor_baseline(instance, n_cities, seed),

                run_random_2opt_baseline(
                    instance,
                    n_cities,
                    seed,
                    args.two_opt_max_iterations,
                ),

                run_nearest_neighbor_2opt_baseline(
                    instance,
                    n_cities,
                    seed,
                    args.two_opt_max_iterations,
                ),

                {
                    **run_random_2opt_baseline(
                        instance,
                        n_cities,
                        seed,
                        args.two_opt_long_max_iterations,
                    ),
                    "method": "random_2opt_long",
                },

                {
                    **run_nearest_neighbor_2opt_baseline(
                        instance,
                        n_cities,
                        seed,
                        args.two_opt_long_max_iterations,
                    ),
                    "method": "nearest_neighbor_2opt_long",
                },

                run_random_hh_baseline(
                    instance,
                    n_cities,
                    seed,
                    args.hh_max_steps,
                    args.hh_initial_method,
                ),

                run_fixed_hh_baseline(
                    instance=instance,
                    n_cities=n_cities,
                    seed=seed,
                    hh_max_steps=args.hh_max_steps,
                    initial_method=args.hh_initial_method,
                    fixed_action=0,
                    method_name="always_first_2opt",
                ),

                run_fixed_hh_baseline(
                    instance=instance,
                    n_cities=n_cities,
                    seed=seed,
                    hh_max_steps=args.hh_max_steps,
                    initial_method=args.hh_initial_method,
                    fixed_action=1,
                    method_name="always_best_2opt",
                ),

                run_fixed_hh_baseline(
                    instance=instance,
                    n_cities=n_cities,
                    seed=seed,
                    hh_max_steps=args.hh_max_steps,
                    initial_method=args.hh_initial_method,
                    fixed_action=2,
                    method_name="always_random_swap",
                ),

                run_fixed_hh_baseline(
                    instance=instance,
                    n_cities=n_cities,
                    seed=seed,
                    hh_max_steps=args.hh_max_steps,
                    initial_method=args.hh_initial_method,
                    fixed_action=3,
                    method_name="always_random_insertion",
                ),

                run_fixed_hh_baseline(
                    instance=instance,
                    n_cities=n_cities,
                    seed=seed,
                    hh_max_steps=args.hh_max_steps,
                    initial_method=args.hh_initial_method,
                    fixed_action=4,
                    method_name="always_perturbation",
                ),

                run_fixed_hh_baseline(
                    instance=instance,
                    n_cities=n_cities,
                    seed=seed,
                    hh_max_steps=args.hh_max_steps,
                    initial_method=args.hh_initial_method,
                    fixed_action=None,
                    method_name="cycle_hh",
                ),
            ]

            if q_agent is not None:
                rows.append(
                    run_q_learning_hh_baseline(
                        instance=instance,
                        n_cities=n_cities,
                        seed=seed,
                        hh_max_steps=args.hh_max_steps,
                        initial_method=args.hh_initial_method,
                        agent=q_agent,
                    )
                )

            all_results.extend(rows)

            best_row = min(rows, key=lambda row: row["tour_length"])

            print(
                f"Finished n_cities={n_cities}, seed={seed}, "
                f"best_method={best_row['method']}, "
                f"best_length={best_row['tour_length']:.3f}"
            )

    df = pd.DataFrame(all_results)

    summary = summarize_results(df)
    summary = add_relative_gap(summary)

    raw_path = out_dir / "comparison_raw.csv"
    summary_path = out_dir / "comparison_summary.csv"

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