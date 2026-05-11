import argparse
from pathlib import Path

import numpy as np
import pandas as pd


try:
    from scipy.stats import wilcoxon
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


DEFAULT_COMPARISONS = [
    ("q_learning_hh", "random_hh"),
    ("q_learning_hh", "always_first_2opt"),
    ("q_learning_hh", "always_best_2opt"),
    ("q_learning_hh", "always_random_swap"),
    ("q_learning_hh", "always_random_insertion"),
    ("q_learning_hh", "always_perturbation"),
    ("q_learning_hh", "cycle_hh"),
    ("q_learning_hh", "nearest_neighbor_2opt"),
    ("q_learning_hh", "nearest_neighbor_2opt_long"),
    ("q_learning_hh", "random_2opt_long"),
]


def print_section(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def paired_method_data(
    df: pd.DataFrame,
    method_a: str,
    method_b: str,
    n_cities: int,
) -> pd.DataFrame:
    """
    Return paired results for method_a and method_b on the same seeds.

    Lower tour length is better.
    """
    subset = df[df["n_cities"] == n_cities].copy()

    pivot = subset.pivot_table(
        index="seed",
        columns="method",
        values="tour_length",
        aggfunc="first",
    )

    if method_a not in pivot.columns or method_b not in pivot.columns:
        return pd.DataFrame()

    paired = pivot[[method_a, method_b]].dropna().copy()

    paired["diff"] = paired[method_a] - paired[method_b]
    paired["percent_improvement"] = (
        (paired[method_b] - paired[method_a]) / paired[method_b] * 100.0
    )

    return paired


def summarize_pair(
    paired: pd.DataFrame,
    method_a: str,
    method_b: str,
) -> dict:
    """
    Summarize paired comparison.

    diff = method_a - method_b

    Since lower tour length is better:
    - diff < 0 means method_a wins
    - diff > 0 means method_b wins
    """
    if paired.empty:
        return {
            "n_pairs": 0,
            "wins": 0,
            "ties": 0,
            "losses": 0,
            "mean_a": np.nan,
            "mean_b": np.nan,
            "mean_diff": np.nan,
            "median_diff": np.nan,
            "mean_percent_improvement": np.nan,
            "wilcoxon_p_value": np.nan,
        }

    diff = paired["diff"]

    wins = int((diff < 0).sum())
    ties = int((diff == 0).sum())
    losses = int((diff > 0).sum())

    p_value = np.nan

    if SCIPY_AVAILABLE:
        nonzero_diff = diff[diff != 0]

        if len(nonzero_diff) > 0:
            try:
                _, p_value = wilcoxon(
                    paired[method_a],
                    paired[method_b],
                    alternative="two-sided",
                )
            except ValueError:
                p_value = np.nan

    return {
        "n_pairs": len(paired),
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "mean_a": paired[method_a].mean(),
        "mean_b": paired[method_b].mean(),
        "mean_diff": diff.mean(),
        "median_diff": diff.median(),
        "mean_percent_improvement": paired["percent_improvement"].mean(),
        "wilcoxon_p_value": p_value,
    }


def run_statistical_tests(
    raw_path: Path,
    out_dir: Path,
    comparisons: list[tuple[str, str]],
) -> pd.DataFrame:
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw comparison file not found: {raw_path}")

    df = pd.read_csv(raw_path)

    required_columns = {"n_cities", "seed", "method", "tour_length"}
    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(f"Missing columns in raw file: {missing}")

    rows = []

    for n_cities in sorted(df["n_cities"].unique()):
        for method_a, method_b in comparisons:
            paired = paired_method_data(
                df=df,
                method_a=method_a,
                method_b=method_b,
                n_cities=n_cities,
            )

            summary = summarize_pair(
                paired=paired,
                method_a=method_a,
                method_b=method_b,
            )

            row = {
                "n_cities": n_cities,
                "method_a": method_a,
                "method_b": method_b,
                **summary,
            }

            rows.append(row)

    results = pd.DataFrame(rows)

    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / "statistical_tests.csv"
    results.to_csv(out_path, index=False)

    return results


def print_results(results: pd.DataFrame) -> None:
    print_section("Statistical comparison results")

    if not SCIPY_AVAILABLE:
        print("SciPy is not available. Wilcoxon p-values were skipped.")

    display_cols = [
        "n_cities",
        "method_a",
        "method_b",
        "n_pairs",
        "wins",
        "ties",
        "losses",
        "mean_a",
        "mean_b",
        "mean_diff",
        "mean_percent_improvement",
        "wilcoxon_p_value",
    ]

    available_cols = [col for col in display_cols if col in results.columns]

    with pd.option_context(
        "display.max_rows",
        None,
        "display.max_columns",
        None,
        "display.width",
        180,
    ):
        print(results[available_cols].to_string(index=False))

    print_section("How to read this table")

    print("method_a is usually q_learning_hh.")
    print("Lower tour length is better.")
    print("wins means method_a produced a shorter tour than method_b on the same seed.")
    print("mean_diff = mean(method_a - method_b).")
    print("A negative mean_diff means method_a is better.")
    print("mean_percent_improvement > 0 means method_a improved over method_b.")
    print("Wilcoxon p-value < 0.05 suggests the paired difference is statistically significant.")


def parse_comparisons(values: list[str] | None) -> list[tuple[str, str]]:
    """
    Parse comparisons passed like:

    --compare q_learning_hh random_hh
    --compare q_learning_hh nearest_neighbor_2opt
    """
    if values is None:
        return DEFAULT_COMPARISONS

    if len(values) % 2 != 0:
        raise ValueError("--compare must receive pairs of methods")

    comparisons = []

    for i in range(0, len(values), 2):
        comparisons.append((values[i], values[i + 1]))

    return comparisons


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--raw-path",
        type=str,
        default="results/comparison_with_q/comparison_raw.csv",
        help="Path to comparison_raw.csv.",
    )

    parser.add_argument(
        "--out-dir",
        type=str,
        default="results/comparison_with_q/stats",
        help="Directory where statistical test results are saved.",
    )

    parser.add_argument(
        "--compare",
        type=str,
        nargs="+",
        default=None,
        help=(
            "Optional method pairs. Example: "
            "--compare q_learning_hh random_hh q_learning_hh nearest_neighbor_2opt"
        ),
    )

    args = parser.parse_args()

    comparisons = parse_comparisons(args.compare)

    results = run_statistical_tests(
        raw_path=Path(args.raw_path),
        out_dir=Path(args.out_dir),
        comparisons=comparisons,
    )

    print_results(results)

    print("\nSaved statistical results to:")
    print(Path(args.out_dir) / "statistical_tests.csv")


if __name__ == "__main__":
    main()