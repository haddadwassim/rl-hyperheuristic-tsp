import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def load_training_log(log_path: Path) -> pd.DataFrame:
    if not log_path.exists():
        raise FileNotFoundError(f"Training log not found: {log_path}")

    df = pd.read_csv(log_path)

    required_columns = {
        "episode",
        "n_cities",
        "best_length",
        "improvement",
        "total_reward",
        "epsilon",
        "q_table_size",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(f"Missing columns in training log: {missing}")

    return df


def add_rolling_columns(df: pd.DataFrame, window: int) -> pd.DataFrame:
    df = df.copy()

    df["improvement_ma"] = df["improvement"].rolling(window=window, min_periods=1).mean()
    df["best_length_ma"] = df["best_length"].rolling(window=window, min_periods=1).mean()
    df["reward_ma"] = df["total_reward"].rolling(window=window, min_periods=1).mean()

    return df


def plot_improvement(df: pd.DataFrame, out_path: Path) -> None:
    plt.figure(figsize=(9, 5))
    plt.plot(df["episode"], df["improvement"], alpha=0.25, label="Improvement")
    plt.plot(df["episode"], df["improvement_ma"], label="Moving average")
    plt.title("Q-learning Training Improvement")
    plt.xlabel("Episode")
    plt.ylabel("Improvement")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_reward(df: pd.DataFrame, out_path: Path) -> None:
    plt.figure(figsize=(9, 5))
    plt.plot(df["episode"], df["total_reward"], alpha=0.25, label="Total reward")
    plt.plot(df["episode"], df["reward_ma"], label="Moving average")
    plt.title("Q-learning Training Reward")
    plt.xlabel("Episode")
    plt.ylabel("Total reward")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_best_length(df: pd.DataFrame, out_path: Path) -> None:
    plt.figure(figsize=(9, 5))
    plt.plot(df["episode"], df["best_length"], alpha=0.25, label="Best length")
    plt.plot(df["episode"], df["best_length_ma"], label="Moving average")
    plt.title("Q-learning Training Best Tour Length")
    plt.xlabel("Episode")
    plt.ylabel("Best tour length")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_epsilon(df: pd.DataFrame, out_path: Path) -> None:
    plt.figure(figsize=(9, 5))
    plt.plot(df["episode"], df["epsilon"])
    plt.title("Epsilon Decay")
    plt.xlabel("Episode")
    plt.ylabel("Epsilon")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_q_table_size(df: pd.DataFrame, out_path: Path) -> None:
    plt.figure(figsize=(9, 5))
    plt.plot(df["episode"], df["q_table_size"])
    plt.title("Q-table Growth")
    plt.xlabel("Episode")
    plt.ylabel("Number of visited discrete states")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--log-path",
        type=str,
        default="results/q_learning/q_learning_train_log.csv",
        help="Path to q_learning_train_log.csv.",
    )

    parser.add_argument(
        "--out-dir",
        type=str,
        default="results/q_learning/plots",
        help="Directory where training plots are saved.",
    )

    parser.add_argument(
        "--window",
        type=int,
        default=25,
        help="Moving-average window.",
    )

    args = parser.parse_args()

    log_path = Path(args.log_path)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_training_log(log_path)
    df = add_rolling_columns(df, window=args.window)

    plot_improvement(df, out_dir / "training_improvement.png")
    plot_reward(df, out_dir / "training_reward.png")
    plot_best_length(df, out_dir / "training_best_length.png")
    plot_epsilon(df, out_dir / "epsilon_decay.png")
    plot_q_table_size(df, out_dir / "q_table_growth.png")

    print("Training plots saved to:")
    print(out_dir)


if __name__ == "__main__":
    main()