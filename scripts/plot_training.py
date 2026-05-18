import argparse

from src.utils.plotting import (
    plot_episode_reward,
    plot_episode_length,
    plot_stop_rate_proxy,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot DQN training curves from Stable-Baselines3 Monitor logs."
    )

    parser.add_argument(
        "--monitor-path",
        type=str,
        default="results/logs/train_monitor.csv",
        help="Path to SB3 Monitor CSV file.",
    )

    parser.add_argument(
        "--window",
        type=int,
        default=20,
        help="Moving average window.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/plots",
        help="Directory where plots will be saved.",
    )

    parser.add_argument(
        "--max-steps",
        type=int,
        default=100,
        help="Maximum episode length used to infer STOP behavior.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    reward_plot = f"{args.output_dir}/training_reward.png"
    length_plot = f"{args.output_dir}/training_episode_length.png"
    stop_plot = f"{args.output_dir}/training_stop_rate.png"

    plot_episode_reward(
        monitor_path=args.monitor_path,
        output_path=reward_plot,
        window=args.window,
    )

    plot_episode_length(
        monitor_path=args.monitor_path,
        output_path=length_plot,
        window=args.window,
    )

    plot_stop_rate_proxy(
        monitor_path=args.monitor_path,
        output_path=stop_plot,
        window=args.window,
        max_steps=args.max_steps,
    )

    print(f"Saved reward plot to: {reward_plot}")
    print(f"Saved episode length plot to: {length_plot}")
    print(f"Saved STOP-rate plot to: {stop_plot}")


if __name__ == "__main__":
    main()