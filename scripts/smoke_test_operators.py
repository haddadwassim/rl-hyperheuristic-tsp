import numpy as np

from src.tsp.generators import generate_uniform_instance
from src.tsp.tour import nearest_neighbor_tour, random_tour, tour_length
from src.operators.registry import apply_operator, ACTION_NAMES


def main():
    rng = np.random.default_rng(42)

    instance = generate_uniform_instance(num_nodes=30, seed=42)

    tour = random_tour(instance, rng=rng)
    initial_length = tour_length(tour, instance)

    print("Initial random tour length:", initial_length)
    print()

    operator_config = {
        "two_opt": {
            "mode": "best_improvement",
            "max_trials": 300,
        },
        "swap": {
            "max_trials": 300,
        },
        "relocate": {
            "max_trials": 300,
        },
        "perturb_two_opt": {
            "perturb_strength": 1,
            "two_opt_trials": 100,
        },
    }

    for action, name in ACTION_NAMES.items():
        new_tour, new_length, improved, info = apply_operator(
            action=action,
            tour=tour,
            instance=instance,
            config=operator_config,
            rng=rng,
        )

        print(f"Action {action}: {name}")
        print(f"  operator: {info['operator']}")
        print(f"  new length: {new_length:.6f}")
        print(f"  delta: {initial_length - new_length:.6f}")
        print(f"  improved: {improved}")
        print(f"  valid permutation: {sorted(new_tour.tolist()) == list(range(instance.num_nodes))}")
        print()


if __name__ == "__main__":
    main()