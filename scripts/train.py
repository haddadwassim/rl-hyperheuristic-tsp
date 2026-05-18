import argparse

from src.agents.train_dqn import load_config, train_dqn


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a DQN operator-selection agent for TSP."
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to the YAML configuration file.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    config = load_config(args.config)

    train_dqn(config)


if __name__ == "__main__":
    main()