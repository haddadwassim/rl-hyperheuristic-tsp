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


METHOD_ORDER = [
    "nearest_neighbor",
    "nearest_neighbor_two_opt",
    "random_operator",
    "fixed_schedule",
    "dqn_efficiency",
    "dqn_quality",
]


def load_generalization_summary(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    df["method"] = pd.Categorical(
        df["method"],
        categories=METHOD_ORDER,
        ordered=True,
    )

    df = df.sort_values(["eval_distribution", "eval_num_nodes", "method"])
    df["label"] = df["method"].astype(str).map(METHOD_LABELS)

    return df


def plot_grouped_bar_for_setting(
    df: pd.DataFrame,
    distribution: str,
    num_nodes: int,
    y_col: str,
    yerr_col: str | None,
    ylabel: str,
    title: str,
    output_path: Path,
):
    setting_df = df[
        (df["eval_distribution"] == distribution)
        & (df["eval_num_nodes"] == num_nodes)
    ].copy()

    output_path.parent.mkdir(parents=True, exist_ok=True)

    yerr = None
    if yerr_col is not None and yerr_col in setting_df.columns:
        yerr = setting_df[yerr_col]

    plt.figure(figsize=(10, 5))
    plt.bar(setting_df["label"], setting_df[y_col], yerr=yerr, capsize=4)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(rotation=25, ha="right")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_all_settings_metric(
    df: pd.DataFrame,
    y_col: str,
    ylabel: str,
    title: str,
    output_path: Path,
):
    """
    Plot all generalization settings in one grouped figure.

    X-axis groups are distribution-size pairs.
    Each method is shown as a separate line.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = df.copy()
    df["setting"] = (
        df["eval_distribution"].astype(str)
        + "-n"
        + df["eval_num_nodes"].astype(str)
    )

    settings = df["setting"].drop_duplicates().tolist()

    plt.figure(figsize=(11, 5))

    for method in METHOD_ORDER:
        method_df = df[df["method"] == method]

        if method_df.empty:
            continue

        method_df = method_df.set_index("setting").reindex(settings)

        plt.plot(
            settings,
            method_df[y_col],
            marker="o",
            label=METHOD_LABELS.get(method, method),
        )

    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(rotation=25, ha="right")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_quality_effort_by_setting(
    df: pd.DataFrame,
    distribution: str,
    num_nodes: int,
    output_path: Path,
):
    setting_df = df[
        (df["eval_distribution"] == distribution)
        & (df["eval_num_nodes"] == num_nodes)
    ].copy()

    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))

    for _, row in setting_df.iterrows():
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
    plt.title(f"Quality-effort trade-off: {distribution}, n={num_nodes}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot generalization benchmark results."
    )

    parser.add_argument(
        "--summary",
        type=str,
        default="results/generalization/generalization_summary.csv",
        help="Path to generalization summary CSV.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/generalization/plots",
        help="Directory where plots will be saved.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    df = load_generalization_summary(args.summary)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Global plots across all distribution-size settings.
    plot_all_settings_metric(
        df=df,
        y_col="mean_relative_improvement",
        ylabel="Mean relative improvement",
        title="Generalization: solution improvement",
        output_path=output_dir / "generalization_relative_improvement.png",
    )

    plot_all_settings_metric(
        df=df,
        y_col="mean_num_steps",
        ylabel="Mean number of operator steps",
        title="Generalization: search effort",
        output_path=output_dir / "generalization_num_steps.png",
    )

    plot_all_settings_metric(
        df=df,
        y_col="mean_runtime_sec",
        ylabel="Mean runtime (seconds)",
        title="Generalization: runtime",
        output_path=output_dir / "generalization_runtime.png",
    )

    # Per-setting bar plots and scatter plots.
    for distribution in df["eval_distribution"].unique():
        for num_nodes in sorted(df["eval_num_nodes"].unique()):
            setting_name = f"{distribution}_n{num_nodes}"

            plot_grouped_bar_for_setting(
                df=df,
                distribution=distribution,
                num_nodes=num_nodes,
                y_col="mean_relative_improvement",
                yerr_col="std_relative_improvement",
                ylabel="Mean relative improvement",
                title=f"Solution improvement: {distribution}, n={num_nodes}",
                output_path=output_dir / f"{setting_name}_improvement.png",
            )

            plot_grouped_bar_for_setting(
                df=df,
                distribution=distribution,
                num_nodes=num_nodes,
                y_col="mean_num_steps",
                yerr_col="std_num_steps",
                ylabel="Mean number of operator steps",
                title=f"Search effort: {distribution}, n={num_nodes}",
                output_path=output_dir / f"{setting_name}_steps.png",
            )

            plot_quality_effort_by_setting(
                df=df,
                distribution=distribution,
                num_nodes=num_nodes,
                output_path=output_dir / f"{setting_name}_quality_effort.png",
            )

    print(f"Saved generalization plots to: {output_dir}")


if __name__ == "__main__":
    main()