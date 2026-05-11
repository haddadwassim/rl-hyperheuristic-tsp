import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


METHOD_LABELS = {
    "random": "Random",
    "nearest_neighbor": "Nearest Neighbor",
    "random_2opt": "Random + 2-opt",
    "nearest_neighbor_2opt": "NN + 2-opt",
    "random_2opt_long": "Random + 2-opt Long",
    "nearest_neighbor_2opt_long": "NN + 2-opt Long",
    "random_hh": "Random HH",
    "always_first_2opt": "Always First 2-opt",
    "always_best_2opt": "Always Best 2-opt",
    "always_random_swap": "Always Swap",
    "always_random_insertion": "Always Insertion",
    "always_perturbation": "Always Perturbation",
    "cycle_hh": "Cyclic HH",
    "q_learning_hh": "Q-learning HH",
}


def load_summary(summary_path: Path) -> pd.DataFrame:
    if not summary_path.exists():
        raise FileNotFoundError(f"Summary file not found: {summary_path}")

    df = pd.read_csv(summary_path)

    required_columns = {
        "n_cities",
        "method",
        "mean_length",
        "mean_runtime_sec",
        "gap_to_best_observed_percent",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(f"Missing columns in summary file: {missing}")

    df["method_label"] = df["method"].map(METHOD_LABELS).fillna(df["method"])

    return df


def plot_mean_length(summary: pd.DataFrame, out_path: Path) -> None:
    pivot = summary.pivot(
        index="n_cities",
        columns="method_label",
        values="mean_length",
    )

    ax = pivot.plot(marker="o", figsize=(9, 5))

    ax.set_title("Mean Tour Length by Method")
    ax.set_xlabel("Number of cities")
    ax.set_ylabel("Mean tour length")
    ax.grid(True, alpha=0.3)
    ax.legend(title="Method", fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_gap_to_best(summary: pd.DataFrame, out_path: Path) -> None:
    pivot = summary.pivot(
        index="n_cities",
        columns="method_label",
        values="gap_to_best_observed_percent",
    )

    ax = pivot.plot(kind="bar", figsize=(10, 5))

    ax.set_title("Gap to Best Observed Method")
    ax.set_xlabel("Number of cities")
    ax.set_ylabel("Gap to best observed (%)")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(title="Method", fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_runtime(summary: pd.DataFrame, out_path: Path) -> None:
    pivot = summary.pivot(
        index="n_cities",
        columns="method_label",
        values="mean_runtime_sec",
    )

    ax = pivot.plot(marker="o", figsize=(9, 5))

    ax.set_title("Mean Runtime by Method")
    ax.set_xlabel("Number of cities")
    ax.set_ylabel("Mean runtime (seconds)")
    ax.grid(True, alpha=0.3)
    ax.legend(title="Method", fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_q_learning_action_usage(raw: pd.DataFrame, out_path: Path) -> None:
    """
    Plot average action usage for q_learning_hh.

    This plot is only generated if q_learning_hh exists in the raw results.
    """
    q_df = raw[raw["method"] == "q_learning_hh"].copy()

    if q_df.empty:
        print("No q_learning_hh rows found. Skipping action-usage plot.")
        return

    action_columns = [
        col for col in q_df.columns
        if col.startswith("count_")
    ]

    if not action_columns:
        print("No action count columns found. Skipping action-usage plot.")
        return

    grouped = q_df.groupby("n_cities")[action_columns].mean()

    grouped.columns = [
        col.replace("count_", "").replace("_", " ")
        for col in grouped.columns
    ]

    ax = grouped.plot(kind="bar", figsize=(10, 5))

    ax.set_title("Average Q-learning Hyper-Heuristic Action Usage")
    ax.set_xlabel("Number of cities")
    ax.set_ylabel("Average action count")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(title="Action", fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--summary-path",
        type=str,
        default="results/comparison_with_q/comparison_summary.csv",
        help="Path to comparison_summary.csv.",
    )

    parser.add_argument(
        "--raw-path",
        type=str,
        default="results/comparison_with_q/comparison_raw.csv",
        help="Path to comparison_raw.csv.",
    )

    parser.add_argument(
        "--out-dir",
        type=str,
        default="results/comparison_with_q/plots",
        help="Directory where plots are saved.",
    )

    args = parser.parse_args()

    summary_path = Path(args.summary_path)
    raw_path = Path(args.raw_path)
    out_dir = Path(args.out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)

    summary = load_summary(summary_path)

    plot_mean_length(
        summary,
        out_dir / "mean_tour_length.png",
    )

    plot_gap_to_best(
        summary,
        out_dir / "gap_to_best_observed.png",
    )

    plot_runtime(
        summary,
        out_dir / "mean_runtime.png",
    )

    if raw_path.exists():
        raw = pd.read_csv(raw_path)

        plot_q_learning_action_usage(
            raw,
            out_dir / "q_learning_action_usage.png",
        )
    else:
        print(f"Raw file not found: {raw_path}. Skipping action-usage plot.")

    print("Plots saved to:")
    print(out_dir)


if __name__ == "__main__":
    main()