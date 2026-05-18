import argparse

from src.agents.train_dqn import load_config
from src.evaluation.evaluate import (
    evaluate_model,
    summarize_results,
    save_evaluation_results,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained DQN operator-selection agent for TSP."
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to the YAML configuration file.",
    )

    parser.add_argument(
        "--model-path",
        type=str,
        default="results/models/dqn_operator_selector.zip",
        help="Path to the trained DQN model.",
    )

    parser.add_argument(
        "--num-episodes",
        type=int,
        default=100,
        help="Number of evaluation episodes.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Evaluation seed.",
    )

    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Use stochastic model predictions instead of deterministic predictions.",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="dqn_evaluation.csv",
        help="Output CSV filename inside results/logs.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    config = load_config(args.config)

    df = evaluate_model(
        model_path=args.model_path,
        config=config,
        num_episodes=args.num_episodes,
        deterministic=not args.stochastic,
        seed=args.seed,
    )

    output_path = save_evaluation_results(
        df=df,
        output_dir=config.get("project", {}).get("output_dir", "results"),
        filename=args.output,
    )

    summary = summarize_results(df)

    print("\nEvaluation summary")
    print("------------------")
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"{key}: {value:.6f}")
        else:
            print(f"{key}: {value}")

    print(f"\nSaved evaluation results to: {output_path}")


if __name__ == "__main__":
    main()