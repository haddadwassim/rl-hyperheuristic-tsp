import argparse
from pathlib import Path

import pandas as pd

from src.agents.train_dqn import load_config
from src.evaluation.benchmark import run_benchmark, summarize_benchmark


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare multiple DQN variants against common TSP baselines."
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
        help="Number of benchmark instances.",
    )

    parser.add_argument(
        "--num-nodes",
        type=int,
        default=50,
        help="Number of TSP nodes.",
    )

    parser.add_argument(
        "--distribution",
        type=str,
        default="uniform",
        choices=["uniform", "clustered", "grid"],
        help="TSP instance distribution.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Benchmark seed.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/comparison",
        help="Directory where comparison results will be saved.",
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

    print("Running quality-focused DQN benchmark...")
    quality_df = run_benchmark(
        config=quality_config,
        model_path=args.quality_model,
        num_instances=args.num_instances,
        num_nodes=args.num_nodes,
        distribution=args.distribution,
        seed=args.seed,
        deterministic=True,
    )
    quality_df = rename_dqn_method(quality_df, "dqn_quality")

    print("Running efficiency-focused DQN benchmark...")
    efficiency_df = run_benchmark(
        config=efficiency_config,
        model_path=args.efficiency_model,
        num_instances=args.num_instances,
        num_nodes=args.num_nodes,
        distribution=args.distribution,
        seed=args.seed,
        deterministic=True,
    )
    efficiency_df = rename_dqn_method(efficiency_df, "dqn_efficiency")

    # Keep baselines from only one run to avoid duplicate baseline rows.
    baseline_methods = [
        "nearest_neighbor",
        "nearest_neighbor_two_opt",
        "random_operator",
        "fixed_schedule",
    ]

    baselines_df = quality_df[quality_df["method"].isin(baseline_methods)]
    quality_dqn_df = quality_df[quality_df["method"] == "dqn_quality"]
    efficiency_dqn_df = efficiency_df[efficiency_df["method"] == "dqn_efficiency"]

    combined_df = pd.concat(
        [baselines_df, quality_dqn_df, efficiency_dqn_df],
        ignore_index=True,
    )

    raw_output = output_dir / (
        f"comparison_{args.distribution}_n{args.num_nodes}_raw.csv"
    )
    summary_output = output_dir / (
        f"comparison_{args.distribution}_n{args.num_nodes}_summary.csv"
    )

    combined_df.to_csv(raw_output, index=False)

    summary = summarize_benchmark(combined_df)

    method_order = {
        "nearest_neighbor": 0,
        "nearest_neighbor_two_opt": 1,
        "random_operator": 2,
        "fixed_schedule": 3,
        "dqn_efficiency": 4,
        "dqn_quality": 5,
    }

    summary["order"] = summary["method"].map(method_order)
    summary = summary.sort_values("order").drop(columns=["order"])
    summary.to_csv(summary_output, index=False)

    print("\nComparison summary")
    print("------------------")
    print(summary.to_string(index=False))

    print(f"\nSaved raw results to: {raw_output}")
    print(f"Saved summary to: {summary_output}")


if __name__ == "__main__":
    main()