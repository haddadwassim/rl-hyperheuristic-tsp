import argparse
import numpy as np
import pandas as pd

from src.tsp.generators import generate_instance
from src.tsp.tour import nearest_neighbor_tour, tour_length
from src.operators.full_two_opt import full_two_opt, perturb_then_full_two_opt
from src.operators.two_opt import two_opt_best_improvement
from src.operators.relocate import relocate_best_improvement
from src.operators.swap import swap_best_improvement


def parse_args():
    parser = argparse.ArgumentParser(
        description="Diagnose empirical utility of macro-actions."
    )

    parser.add_argument("--num-instances", type=int, default=100)
    parser.add_argument("--num-nodes", type=int, default=50)
    parser.add_argument("--distribution", type=str, default="uniform")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--perturb-trials", type=int, default=10)
    parser.add_argument("--output", type=str, default="results/logs/macro_action_diagnostics.csv")

    return parser.parse_args()


def evaluate_candidate_action(name, old_best, new_length, initial_length):
    improvement = max(0.0, (old_best - new_length) / initial_length)

    return {
        "action": name,
        "success": float(new_length < old_best),
        "best_improvement": improvement,
        "new_length": new_length,
    }


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    rows = []

    for i in range(args.num_instances):
        instance = generate_instance(
            distribution=args.distribution,
            num_nodes=args.num_nodes,
            seed=args.seed + i,
        )

        initial_tour = nearest_neighbor_tour(instance)
        initial_length = tour_length(initial_tour, instance)

        local_tour, local_length, _ = full_two_opt(
            tour=initial_tour,
            instance=instance,
            max_passes=1000,
            rng=rng,
        )

        old_best = local_length

        # Try sampled 2-opt from local optimum.
        t, l, _ = two_opt_best_improvement(
            tour=local_tour,
            instance=instance,
            max_trials=200,
            rng=rng,
        )
        row = evaluate_candidate_action(
            "sampled_two_opt_after_full",
            old_best,
            l,
            initial_length,
        )
        row["instance_id"] = i
        rows.append(row)

        # Try relocate from local optimum.
        t, l, _ = relocate_best_improvement(
            tour=local_tour,
            instance=instance,
            max_trials=200,
            rng=rng,
        )
        row = evaluate_candidate_action(
            "relocate_after_full",
            old_best,
            l,
            initial_length,
        )
        row["instance_id"] = i
        rows.append(row)

        # Try swap from local optimum.
        t, l, _ = swap_best_improvement(
            tour=local_tour,
            instance=instance,
            max_trials=200,
            rng=rng,
        )
        row = evaluate_candidate_action(
            "swap_after_full",
            old_best,
            l,
            initial_length,
        )
        row["instance_id"] = i
        rows.append(row)

        # Try perturb + full 2-opt several times.
        for trial in range(args.perturb_trials):
            t, l, _ = perturb_then_full_two_opt(
                tour=local_tour,
                instance=instance,
                perturb_strength=1,
                max_passes=1000,
                rng=rng,
            )
            row = evaluate_candidate_action(
                "perturb_full_two_opt_after_full",
                old_best,
                l,
                initial_length,
            )
            row["instance_id"] = i
            row["trial"] = trial
            rows.append(row)

    df = pd.DataFrame(rows)

    summary = (
        df.groupby("action")
        .agg(
            success_rate=("success", "mean"),
            mean_best_improvement=("best_improvement", "mean"),
            max_best_improvement=("best_improvement", "max"),
            mean_new_length=("new_length", "mean"),
        )
        .reset_index()
    )

    print("\nMacro-action utility after full 2-opt")
    print("------------------------------------")
    print(summary.to_string(index=False))

    df.to_csv(args.output, index=False)
    print(f"\nSaved diagnostics to: {args.output}")


if __name__ == "__main__":
    main()