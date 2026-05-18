import argparse
from pathlib import Path

import pandas as pd

from src.agents.train_dqn import load_config
from src.evaluation.benchmark import run_benchmark, summarize_benchmark


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run generalization benchmarks across TSP sizes and distributions."
    )

    parser.add_argument(
        "--quality-config",
        type=str,
        default="configs/default.yaml",
        help="Config file for the quality-focused DQN model.",
    )

    parser.add_argument(
        "--quality-model",
        type=str,
        default="results/run_100k_stop_reward/models/dqn_operator_selector.zip",
        help="Path to the quality-focused DQN model.",
    )

    parser.add_argument(
        "--efficiency-config",
        type=str,
        default="configs/efficiency_stop.yaml",
        help="Config file for the efficiency-focused DQN model.",
    )

    parser.add_argument(
        "--efficiency-model",
        type=str,
        default="results/run_100k_efficiency_stop/models/dqn_operator_selector.zip",
        help="Path to the efficiency-focused DQN model.",
    )

    parser.add_argument(
        "--num-instances",
        type=int,
        default=50,
        help="Number of instances per setting.",
    )

    parser.add_argument(
        "--node-sizes",
        type=int,
        nargs="+",
        default=[50, 100],
        help="TSP sizes to evaluate.",
    )

    parser.add_argument(
        "--distributions",
        type=str,
        nargs="+",
        default=["uniform", "clustered", "grid"],
        choices=["uniform", "clustered", "grid"],
        help="TSP distributions to evaluate.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/generalization",
        help="Directory where results will be saved.",
    )

    return parser.parse_args()


def rename_dqn_method(df: pd.DataFrame, new_name: str) -> pd.DataFrame:
    df = df.copy()
    df.loc[df["method"] == "dqn_operator_selector", "method"] = new_name
    return df


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    quality_config = load_config(args.quality_config)
    efficiency_config = load_config(args.efficiency_config)

    all_rows = []

    baseline_methods = [
        "nearest_neighbor",
        "nearest_neighbor_two_opt",
        "random_operator",
        "fixed_schedule",
    ]

    for distribution in args.distributions:
        for num_nodes in args.node_sizes:
            print(f"\nRunning setting: distribution={distribution}, n={num_nodes}")

            setting_seed = args.seed + num_nodes + 1000 * args.distributions.index(distribution)

            print("  Quality-focused DQN...")
            quality_df = run_benchmark(
                config=quality_config,
                model_path=args.quality_model,
                num_instances=args.num_instances,
                num_nodes=num_nodes,
                distribution=distribution,
                seed=setting_seed,
                deterministic=True,
            )
            quality_df = rename_dqn_method(quality_df, "dqn_quality")

            print("  Efficiency-focused DQN...")
            efficiency_df = run_benchmark(
                config=efficiency_config,
                model_path=args.efficiency_model,
                num_instances=args.num_instances,
                num_nodes=num_nodes,
                distribution=distribution,
                seed=setting_seed,
                deterministic=True,
            )
            efficiency_df = rename_dqn_method(efficiency_df, "dqn_efficiency")

            baselines_df = quality_df[quality_df["method"].isin(baseline_methods)]
            quality_dqn_df = quality_df[quality_df["method"] == "dqn_quality"]
            efficiency_dqn_df = efficiency_df[efficiency_df["method"] == "dqn_efficiency"]

            setting_df = pd.concat(
                [baselines_df, quality_dqn_df, efficiency_dqn_df],
                ignore_index=True,
            )

            setting_df["eval_distribution"] = distribution
            setting_df["eval_num_nodes"] = num_nodes

            all_rows.append(setting_df)

    raw_df = pd.concat(all_rows, ignore_index=True)

    raw_output = output_dir / "generalization_raw.csv"
    summary_output = output_dir / "generalization_summary.csv"

    raw_df.to_csv(raw_output, index=False)

    summary = (
        raw_df.groupby(["eval_distribution", "eval_num_nodes", "method"])
        .agg(
            mean_initial_length=("initial_length", "mean"),
            mean_final_length=("final_length", "mean"),
            mean_relative_improvement=("relative_improvement", "mean"),
            std_relative_improvement=("relative_improvement", "std"),
            mean_num_steps=("num_steps", "mean"),
            std_num_steps=("num_steps", "std"),
            mean_runtime_sec=("runtime_sec", "mean"),
            std_runtime_sec=("runtime_sec", "std"),
        )
        .reset_index()
    )

    method_order = {
        "nearest_neighbor": 0,
        "nearest_neighbor_two_opt": 1,
        "random_operator": 2,
        "fixed_schedule": 3,
        "dqn_efficiency": 4,
        "dqn_quality": 5,
    }

    summary["method_order"] = summary["method"].map(method_order)
    summary = summary.sort_values(
        ["eval_distribution", "eval_num_nodes", "method_order"]
    ).drop(columns=["method_order"])

    summary.to_csv(summary_output, index=False)

    print("\nGeneralization summary")
    print("----------------------")
    print(summary.to_string(index=False))

    print(f"\nSaved raw results to: {raw_output}")
    print(f"Saved summary to: {summary_output}")


if __name__ == "__main__":
    main()