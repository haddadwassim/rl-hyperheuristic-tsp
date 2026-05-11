import argparse
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd

from tsp_hh.tsplib_loader import list_tsplib_files, load_tsplib_instance
from tsp_hh.hyper_env_v3 import TSPHyperHeuristicEnvV3
from tsp_hh.q_learning_v3 import QLearningAgent, discretize_state_v3
from tsp_hh.rule_based_teacher_v3 import RuleBasedTeacherV3, TeacherConfigV3


TSPLIB_OPTIMA = {
    "eil51": 426.0,
    "berlin52": 7542.0,
    "st70": 675.0,
    "eil76": 538.0,
    "kroA100": 21282.0,
    "ch130": 6110.0,
}


def load_q_learning_agent_v3(path: Path, seed: int | None = None) -> QLearningAgent:
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


def run_policy_on_instance(
    instance_name: str,
    instance,
    optimum: float,
    run_seed: int,
    method_name: str,
    policy_type: str,
    max_steps: int,
    initial_method: str,
    construction_starts: int,
    two_opt_iterations: int,
    three_opt_samples: int,
    accept_worse_perturbation: bool,
    reward_scale: str,
    construction_penalty: float,
    agent: QLearningAgent | None = None,
    teacher: RuleBasedTeacherV3 | None = None,
    fixed_category: int | None = None,
) -> dict:
    env = TSPHyperHeuristicEnvV3(
        distance_matrix=instance.distance_matrix,
        initial_method=initial_method,
        max_steps=max_steps,
        seed=run_seed,
        construction_starts=construction_starts,
        two_opt_iterations=two_opt_iterations,
        three_opt_samples=three_opt_samples,
        accept_worse_perturbation=accept_worse_perturbation,
        reward_scale=reward_scale,
        construction_penalty=construction_penalty,
    )

    rng = np.random.default_rng(run_seed)
    state = env.reset()

    initial_length = env.best_length
    initial_gap = (initial_length - optimum) / optimum * 100.0

    start_time = time.perf_counter()

    done = False
    step = 0

    while not done:
        if policy_type == "random":
            category = int(rng.integers(0, 3))

        elif policy_type == "cycle":
            category = step % 3

        elif policy_type == "fixed":
            if fixed_category is None:
                raise ValueError("fixed_category must be provided")
            category = fixed_category

        elif policy_type == "teacher":
            if teacher is None:
                raise ValueError("teacher must be provided")
            category = teacher.select_category(state)

        elif policy_type == "q_learning":
            if agent is None:
                raise ValueError("agent must be provided")
            state_key = discretize_state_v3(state, max_steps=max_steps)
            category = agent.select_action(state_key, training=False)

        else:
            raise ValueError(f"Unknown policy_type: {policy_type}")

        next_state, reward, done, info = env.step(category)
        state = next_state
        step += 1

    runtime_sec = time.perf_counter() - start_time

    final_gap = (env.best_length - optimum) / optimum * 100.0

    row = {
        "instance": instance_name,
        "n_cities": instance.coords.shape[0],
        "run_seed": run_seed,
        "method": method_name,
        "optimum": optimum,
        "initial_length": initial_length,
        "tour_length": env.best_length,
        "improvement": initial_length - env.best_length,
        "initial_gap_percent": initial_gap,
        "gap_to_optimum_percent": final_gap,
        "gap_improvement_percent": initial_gap - final_gap,
        "runtime_sec": runtime_sec,
        "max_steps": max_steps,
        "construction_starts": construction_starts,
        "two_opt_iterations": two_opt_iterations,
        "three_opt_samples": three_opt_samples,
    }

    for name, count in env.category_counts.items():
        row[f"count_category_{name}"] = count

    for name, count in env.heuristic_counts.items():
        clean = name.replace(":", "_")
        row[f"count_heuristic_{clean}"] = count

    return row


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
            mean_gap_improvement_percent=("gap_improvement_percent", "mean"),
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
    parser.add_argument("--q-table-path", type=str, default="results/q_learning_v3_teacher_medium/q_table_v3_teacher.pkl")
    parser.add_argument("--out-dir", type=str, default="results/tsplib_v3_teacher_medium")
    parser.add_argument("--instances", type=str, nargs="*", default=None)
    parser.add_argument("--n-runs", type=int, default=10)
    parser.add_argument("--seed-offset", type=int, default=30_000)

    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--initial-method", type=str, default="greedy", choices=["random", "nearest_neighbor", "greedy"])
    parser.add_argument("--construction-starts", type=int, default=10)
    parser.add_argument("--two-opt-iterations", type=int, default=30)
    parser.add_argument("--three-opt-samples", type=int, default=30)
    parser.add_argument("--accept-worse-perturbation", action="store_true")
    parser.add_argument("--reward-scale", type=str, default="initial_length", choices=["initial_length", "none"])
    parser.add_argument("--construction-penalty", type=float, default=0.0)

    parser.add_argument("--agent-seed", type=int, default=42)

    parser.add_argument("--mild-stagnation", type=int, default=5)
    parser.add_argument("--medium-stagnation", type=int, default=15)
    parser.add_argument("--strong-stagnation", type=int, default=35)
    parser.add_argument("--restart-stagnation", type=int, default=60)

    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    agent = load_q_learning_agent_v3(Path(args.q_table_path), seed=args.agent_seed)

    teacher = RuleBasedTeacherV3(
        TeacherConfigV3(
            mild_stagnation=args.mild_stagnation,
            medium_stagnation=args.medium_stagnation,
            strong_stagnation=args.strong_stagnation,
            restart_stagnation=args.restart_stagnation,
            force_improve_after_perturbation=True,
        )
    )

    tsp_files = list_tsplib_files(args.data_dir)

    if args.instances:
        wanted = {x if x.endswith(".tsp") else f"{x}.tsp" for x in args.instances}
        tsp_files = [p for p in tsp_files if p.name in wanted]

    if not tsp_files:
        raise FileNotFoundError("No TSPLIB instances found")

    all_rows = []

    for path in tsp_files:
        instance_name = path.stem

        if instance_name not in TSPLIB_OPTIMA:
            print(f"Skipping {instance_name}: no known optimum")
            continue

        instance = load_tsplib_instance(path)
        optimum = TSPLIB_OPTIMA[instance_name]

        print(f"\nRunning {instance_name} (n={instance.coords.shape[0]})")

        for run_idx in range(args.n_runs):
            run_seed = args.seed_offset + run_idx

            rows = [
                run_policy_on_instance(
                    instance_name, instance, optimum, run_seed,
                    "random_category_v3", "random",
                    args.max_steps, args.initial_method, args.construction_starts,
                    args.two_opt_iterations, args.three_opt_samples,
                    args.accept_worse_perturbation, args.reward_scale,
                    args.construction_penalty,
                ),
                run_policy_on_instance(
                    instance_name, instance, optimum, run_seed,
                    "cycle_category_v3", "cycle",
                    args.max_steps, args.initial_method, args.construction_starts,
                    args.two_opt_iterations, args.three_opt_samples,
                    args.accept_worse_perturbation, args.reward_scale,
                    args.construction_penalty,
                ),
                run_policy_on_instance(
                    instance_name, instance, optimum, run_seed,
                    "always_construction_v3", "fixed",
                    args.max_steps, args.initial_method, args.construction_starts,
                    args.two_opt_iterations, args.three_opt_samples,
                    args.accept_worse_perturbation, args.reward_scale,
                    args.construction_penalty,
                    fixed_category=0,
                ),
                run_policy_on_instance(
                    instance_name, instance, optimum, run_seed,
                    "always_improvement_v3", "fixed",
                    args.max_steps, args.initial_method, args.construction_starts,
                    args.two_opt_iterations, args.three_opt_samples,
                    args.accept_worse_perturbation, args.reward_scale,
                    args.construction_penalty,
                    fixed_category=1,
                ),
                run_policy_on_instance(
                    instance_name, instance, optimum, run_seed,
                    "always_perturbation_v3", "fixed",
                    args.max_steps, args.initial_method, args.construction_starts,
                    args.two_opt_iterations, args.three_opt_samples,
                    args.accept_worse_perturbation, args.reward_scale,
                    args.construction_penalty,
                    fixed_category=2,
                ),
                run_policy_on_instance(
                    instance_name, instance, optimum, run_seed,
                    "teacher_policy_v3", "teacher",
                    args.max_steps, args.initial_method, args.construction_starts,
                    args.two_opt_iterations, args.three_opt_samples,
                    args.accept_worse_perturbation, args.reward_scale,
                    args.construction_penalty,
                    teacher=teacher,
                ),
                run_policy_on_instance(
                    instance_name, instance, optimum, run_seed,
                    "teacher_guided_q_learning_v3", "q_learning",
                    args.max_steps, args.initial_method, args.construction_starts,
                    args.two_opt_iterations, args.three_opt_samples,
                    args.accept_worse_perturbation, args.reward_scale,
                    args.construction_penalty,
                    agent=agent,
                ),
            ]

            all_rows.extend(rows)

            best_row = min(rows, key=lambda r: r["tour_length"])
            print(
                f"  run={run_idx + 1}/{args.n_runs} | "
                f"best={best_row['method']} | "
                f"gap={best_row['gap_to_optimum_percent']:.3f}%"
            )

    raw = pd.DataFrame(all_rows)
    summary = summarize_results(raw)

    raw_path = out_dir / "tsplib_v3_raw.csv"
    summary_path = out_dir / "tsplib_v3_summary.csv"

    raw.to_csv(raw_path, index=False)
    summary.to_csv(summary_path, index=False)

    print("\nV3 TSPLIB comparison finished.")
    print(f"Raw saved to: {raw_path}")
    print(f"Summary saved to: {summary_path}")
    print("\nSummary:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()