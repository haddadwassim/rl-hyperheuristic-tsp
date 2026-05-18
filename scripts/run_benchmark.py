import argparse

from src.agents.train_dqn import load_config
from src.evaluation.benchmark import (
    run_benchmark,
    summarize_benchmark,
    save_benchmark_results,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run benchmark comparison for TSP operator-selection methods."
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to config file.",
    )

    parser.add_argument(
        "--model-path",
        type=str,
        default="results/models/dqn_operator_selector.zip",
        help="Path to trained DQN model.",
    )

    parser.add_argument(
        "--no-dqn",
        action="store_true",
        help="Run only classical/non-learning baselines.",
    )

    parser.add_argument(
        "--num-instances",
        type=int,
        default=20,
        help="Number of benchmark instances.",
    )

    parser.add_argument(
        "--num-nodes",
        type=int,
        default=None,
        help="Number of nodes. If omitted, config value is used.",
    )

    parser.add_argument(
        "--distribution",
        type=str,
        default=None,
        choices=["uniform", "clustered", "grid"],
        help="Instance distribution.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Benchmark seed.",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="benchmark.csv",
        help="Output CSV filename inside results/logs.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    config = load_config(args.config)

    model_path = None if args.no_dqn else args.model_path

    df = run_benchmark(
        config=config,
        model_path=model_path,
        num_instances=args.num_instances,
        num_nodes=args.num_nodes,
        distribution=args.distribution,
        seed=args.seed,
        deterministic=True,
    )

    output_path = save_benchmark_results(
        df=df,
        output_dir=config.get("project", {}).get("output_dir", "results"),
        filename=args.output,
    )

    summary = summarize_benchmark(df)

    print("\nBenchmark summary")
    print("-----------------")
    print(summary.to_string(index=False))

    print(f"\nSaved benchmark results to: {output_path}")


if __name__ == "__main__":
    main()