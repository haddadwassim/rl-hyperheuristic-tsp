import argparse
from pathlib import Path

import pandas as pd
from stable_baselines3 import DQN

from src.agents.train_dqn import load_config
from src.tsp.tsplib import load_tsplib_instance
from src.evaluation.benchmark import run_dqn_on_instance, summarize_benchmark
from src.baselines.nearest_neighbor import (
    run_nearest_neighbor,
    run_nearest_neighbor_two_opt,
)
from src.baselines.random_operator import run_random_operator
from src.baselines.fixed_schedule import run_fixed_schedule
from src.tsp.tsplib_optima import get_tsplib_optimum
from src.baselines.two_opt_full import run_full_two_opt
from src.baselines.multistart_two_opt import run_multistart_two_opt


DEFAULT_TSP_FILES = [
    "eil51.tsp",
    "berlin52.tsp",
    "st70.tsp",
    "pr76.tsp",
    "rat99.tsp",
    "kroA100.tsp",
    "kroB100.tsp",
    "eil101.tsp",
    "ch130.tsp",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run benchmark comparison on TSPLIB instances."
    )

    parser.add_argument(
        "--tsplib-dir",
        type=str,
        default="data/tsplib",
        help="Directory containing TSPLIB .tsp files.",
    )

    parser.add_argument(
        "--quality-config",
        type=str,
        default="configs/default.yaml",
        help="Config for DQN-quality.",
    )

    parser.add_argument(
        "--quality-model",
        type=str,
        default="results/run_100k_stop_reward/models/dqn_operator_selector.zip",
        help="Model path for DQN-quality.",
    )

    parser.add_argument(
        "--efficiency-config",
        type=str,
        default="configs/efficiency_stop.yaml",
        help="Config for DQN-efficient.",
    )

    parser.add_argument(
        "--efficiency-model",
        type=str,
        default="results/run_100k_efficiency_stop/models/dqn_operator_selector.zip",
        help="Model path for DQN-efficient.",
    )

    parser.add_argument(
        "--files",
        type=str,
        nargs="*",
        default=DEFAULT_TSP_FILES,
        help="TSPLIB files to evaluate.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for stochastic baselines/operators.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/tsplib",
        help="Directory where TSPLIB results will be saved.",
    )

    parser.add_argument(
        "--multistart-runs",
        type=int,
        default=10,
        help="Number of starts for multi-start 2-opt baseline.",
    )

    parser.add_argument(
        "--two-opt-max-passes",
        type=int,
        default=1000,
        help="Maximum full 2-opt passes.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    tsplib_dir = Path(args.tsplib_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    quality_config = load_config(args.quality_config)
    efficiency_config = load_config(args.efficiency_config)

    quality_model = DQN.load(args.quality_model)
    efficiency_model = DQN.load(args.efficiency_model)

    rows = []

    for filename in args.files:
        path = tsplib_dir / filename

        if not path.exists():
            print(f"Skipping missing file: {path}")
            continue

        print(f"Running TSPLIB instance: {filename}")

        instance = load_tsplib_instance(path)
        num_nodes = instance.num_nodes
        known_optimum = get_tsplib_optimum(filename)

        quality_env_config = quality_config.get("environment", {})
        efficiency_env_config = efficiency_config.get("environment", {})

        quality_operator_config = quality_config.get("operators", {})
        efficiency_operator_config = efficiency_config.get("operators", {})

        quality_initial_method = quality_config.get("initial_solution", {}).get(
            "method", "nearest_neighbor"
        )
        efficiency_initial_method = efficiency_config.get("initial_solution", {}).get(
            "method", "nearest_neighbor"
        )

        max_steps_quality = quality_env_config.get("max_steps", 100)
        max_steps_efficiency = efficiency_env_config.get("max_steps", 100)

        baseline_results = [
            run_nearest_neighbor(instance),
            run_nearest_neighbor_two_opt(
                instance=instance,
                max_steps=max_steps_quality,
                max_trials=quality_operator_config.get("two_opt", {}).get(
                    "max_trials", 200
                ),
                seed=args.seed,
            ),
            run_random_operator(
                instance=instance,
                operator_config=quality_operator_config,
                max_steps=max_steps_quality,
                initial_solution_method=quality_initial_method,
                seed=args.seed,
            ),
            run_fixed_schedule(
                instance=instance,
                operator_config=quality_operator_config,
                max_steps=max_steps_quality,
                initial_solution_method=quality_initial_method,
                seed=args.seed,
            ),
            run_full_two_opt(
                instance=instance,
                initial_solution_method="nearest_neighbor",
                max_passes=args.two_opt_max_passes,
                seed=args.seed,
            ),
            run_multistart_two_opt(
                instance=instance,
                num_starts=args.multistart_runs,
                max_passes_per_start=args.two_opt_max_passes,
                seed=args.seed,
            ),
        ]

        dqn_efficiency = run_dqn_on_instance(
            model=efficiency_model,
            instance=instance,
            config=efficiency_config,
            max_steps=max_steps_efficiency,
            deterministic=True,
            seed=args.seed,
        )
        dqn_efficiency["method"] = "dqn_efficiency"

        dqn_quality = run_dqn_on_instance(
            model=quality_model,
            instance=instance,
            config=quality_config,
            max_steps=max_steps_quality,
            deterministic=True,
            seed=args.seed,
        )
        dqn_quality["method"] = "dqn_quality"

        instance_results = baseline_results + [dqn_efficiency, dqn_quality]

        for result in instance_results:
            result = result.copy()
            result.pop("tour", None)
            result["instance"] = filename.replace(".tsp", "")
            result["num_nodes"] = num_nodes
            result["known_optimum"] = known_optimum
            if known_optimum is not None:
                result["gap_to_optimum_percent"] = (
                    (result["final_length"] - known_optimum) / known_optimum
                ) * 100
            else:
                result["gap_to_optimum_percent"] = None
            rows.append(result)

    raw_df = pd.DataFrame(rows)

    raw_output = output_dir / "tsplib_raw.csv"
    summary_output = output_dir / "tsplib_summary.csv"

    raw_df.to_csv(raw_output, index=False)

    summary = (
        raw_df.groupby("method")
        .agg(
            mean_final_length=("final_length", "mean"),
            mean_gap_to_optimum_percent=("gap_to_optimum_percent", "mean"),
            std_gap_to_optimum_percent=("gap_to_optimum_percent", "std"),
            mean_relative_improvement=("relative_improvement", "mean"),
            mean_num_steps=("num_steps", "mean"),
            mean_runtime_sec=("runtime_sec", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(summary_output, index=False)

    print("\nTSPLIB aggregate summary")
    print("------------------------")
    print(summary.to_string(index=False))

    print("\nPer-instance results")
    print("--------------------")
    selected_cols = [
        "instance",
        "num_nodes",
        "method",
        "known_optimum",
        "final_length",
        "gap_to_optimum_percent",
        "relative_improvement",
        "num_steps",
        "runtime_sec",
    ]
    print(raw_df[selected_cols].to_string(index=False))

    print(f"\nSaved raw results to: {raw_output}")
    print(f"Saved summary to: {summary_output}")


if __name__ == "__main__":
    main()