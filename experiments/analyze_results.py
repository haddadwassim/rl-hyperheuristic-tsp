import argparse
from pathlib import Path

import pandas as pd


def print_section(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def analyze_training(log_path: Path) -> None:
    if not log_path.exists():
        print(f"Training log not found: {log_path}")
        return

    df = pd.read_csv(log_path)

    print_section("Q-learning training overview")

    print(f"Number of episodes: {len(df)}")
    print(f"Final epsilon: {df['epsilon'].iloc[-1]:.4f}")
    print(f"Final Q-table size: {df['q_table_size'].iloc[-1]}")

    first_part = df.head(max(1, len(df) // 5))
    last_part = df.tail(max(1, len(df) // 5))

    print("\nAverage improvement:")
    print(f"  First 20% episodes: {first_part['improvement'].mean():.4f}")
    print(f"  Last 20% episodes:  {last_part['improvement'].mean():.4f}")

    print("\nAverage total reward:")
    print(f"  First 20% episodes: {first_part['total_reward'].mean():.4f}")
    print(f"  Last 20% episodes:  {last_part['total_reward'].mean():.4f}")

    action_cols = [c for c in df.columns if c.startswith("count_")]

    if action_cols:
        print("\nAverage action usage in last 20% of training:")
        usage = last_part[action_cols].mean().sort_values(ascending=False)

        for action, value in usage.items():
            clean_name = action.replace("count_", "")
            print(f"  {clean_name:25s}: {value:.2f}")


def analyze_comparison(summary_path: Path, raw_path: Path | None = None) -> None:
    if not summary_path.exists():
        print(f"Comparison summary not found: {summary_path}")
        return

    summary = pd.read_csv(summary_path)

    print_section("Final comparison summary")

    cols = [
        "n_cities",
        "method",
        "mean_length",
        "std_length",
        "mean_improvement",
        "mean_runtime_sec",
        "gap_to_best_observed_percent",
    ]

    available_cols = [c for c in cols if c in summary.columns]
    print(summary[available_cols].to_string(index=False))

    print_section("Best method by problem size")

    for n_cities, group in summary.groupby("n_cities"):
        best = group.sort_values("mean_length").iloc[0]
        print(
            f"n={n_cities}: best={best['method']} | "
            f"mean_length={best['mean_length']:.4f}"
        )

    if raw_path is not None and raw_path.exists():
        raw = pd.read_csv(raw_path)

        if "q_learning_hh" in raw["method"].unique() and "random_hh" in raw["method"].unique():
            print_section("Q-learning HH vs Random HH win rate")

            pivot = raw.pivot_table(
                index=["n_cities", "seed"],
                columns="method",
                values="tour_length",
                aggfunc="first",
            ).reset_index()

            if {"q_learning_hh", "random_hh"}.issubset(pivot.columns):
                for n_cities, group in pivot.groupby("n_cities"):
                    wins = (group["q_learning_hh"] < group["random_hh"]).sum()
                    ties = (group["q_learning_hh"] == group["random_hh"]).sum()
                    losses = (group["q_learning_hh"] > group["random_hh"]).sum()
                    total = len(group)

                    print(
                        f"n={n_cities}: "
                        f"wins={wins}/{total}, ties={ties}/{total}, losses={losses}/{total}"
                    )

            q_df = raw[raw["method"] == "q_learning_hh"]
            action_cols = [c for c in q_df.columns if c.startswith("count_")]

            if action_cols:
                print_section("Q-learning HH average action usage during evaluation")

                usage = q_df.groupby("n_cities")[action_cols].mean()

                for n_cities, row in usage.iterrows():
                    print(f"\nn={n_cities}")
                    sorted_row = row.sort_values(ascending=False)

                    for action, value in sorted_row.items():
                        clean_name = action.replace("count_", "")
                        print(f"  {clean_name:25s}: {value:.2f}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train-log",
        type=str,
        default="results/q_learning/q_learning_train_log.csv",
        help="Path to Q-learning training log.",
    )

    parser.add_argument(
        "--summary",
        type=str,
        default="results/comparison_with_q/comparison_summary.csv",
        help="Path to comparison summary CSV.",
    )

    parser.add_argument(
        "--raw",
        type=str,
        default="results/comparison_with_q/comparison_raw.csv",
        help="Path to comparison raw CSV.",
    )

    args = parser.parse_args()

    analyze_training(Path(args.train_log))
    analyze_comparison(Path(args.summary), Path(args.raw))


if __name__ == "__main__":
    main()