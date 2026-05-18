import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


METHOD_LABELS = {
    "nearest_neighbor": "NN",
    "nearest_neighbor_two_opt": "NN + 2-opt",
    "random_operator": "Random",
    "fixed_schedule": "Fixed schedule",
    "dqn_efficiency": "DQN-efficient",
    "dqn_quality": "DQN-quality",
}


def load_summary(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["label"] = df["method"].map(METHOD_LABELS).fillna(df["method"])
    return df


def plot_bar(
    df: pd.DataFrame,
    y_col: str,
    ylabel: str,
    title: str,
    output_path: str,
):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 5))
    plt.bar(df["label"], df[y_col])
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(rotation=25, ha="right")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_quality_runtime_scatter(
    df: pd.DataFrame,
    output_path: str,
):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))

    for _, row in df.iterrows():
        x = row["mean_runtime_sec"]
        y = row["mean_relative_improvement"]
        label = row["label"]

        plt.scatter(x, y, s=80)
        plt.annotate(
            label,
            (x, y),
            textcoords="offset points",
            xytext=(6, 6),
            ha="left",
        )

    plt.xlabel("Mean runtime (seconds)")
    plt.ylabel("Mean relative improvement")
    plt.title("Quality-runtime trade-off")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_quality_steps_scatter(
    df: pd.DataFrame,
    output_path: str,
):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))

    for _, row in df.iterrows():
        x = row["mean_num_steps"]
        y = row["mean_relative_improvement"]
        label = row["label"]

        plt.scatter(x, y, s=80)
        plt.annotate(
            label,
            (x, y),
            textcoords="offset points",
            xytext=(6, 6),
            ha="left",
        )

    plt.xlabel("Mean number of operator steps")
    plt.ylabel("Mean relative improvement")
    plt.title("Quality-search effort trade-off")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot paper-style comparison figures from benchmark summary."
    )

    parser.add_argument(
        "--summary",
        type=str,
        default="results/comparison/comparison_uniform_n50_summary.csv",
        help="Path to comparison summary CSV.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/comparison/plots",
        help="Directory where plots will be saved.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    df = load_summary(args.summary)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_bar(
        df=df,
        y_col="mean_relative_improvement",
        ylabel="Mean relative improvement",
        title="Solution improvement by method",
        output_path=output_dir / "mean_relative_improvement.png",
    )

    plot_bar(
        df=df,
        y_col="mean_num_steps",
        ylabel="Mean number of operator steps",
        title="Search effort by method",
        output_path=output_dir / "mean_num_steps.png",
    )

    plot_bar(
        df=df,
        y_col="mean_runtime_sec",
        ylabel="Mean runtime (seconds)",
        title="Runtime by method",
        output_path=output_dir / "mean_runtime.png",
    )

    plot_quality_runtime_scatter(
        df=df,
        output_path=output_dir / "quality_runtime_tradeoff.png",
    )

    plot_quality_steps_scatter(
        df=df,
        output_path=output_dir / "quality_steps_tradeoff.png",
    )

    print(f"Saved plots to: {output_dir}")


if __name__ == "__main__":
    main()