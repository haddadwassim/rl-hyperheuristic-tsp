import numpy as np

from tsp_hh.instances import generate_euclidean_instance
from tsp_hh.heuristics_v3 import construct_tour_v3
from tsp_hh.tour import tour_length
from tsp_hh.operators import apply_operator, ACTION_NAMES


def compute_distance_matrix(coords: np.ndarray) -> np.ndarray:
    diff = coords[:, None, :] - coords[None, :, :]
    return np.linalg.norm(diff, axis=-1)


def main():
    instance = generate_euclidean_instance(n_cities=50, seed=42)
    distance_matrix = compute_distance_matrix(instance.coords)

    tour = construct_tour_v3(
        distance_matrix=distance_matrix,
        method="nearest_neighbor",
        seed=42,
        n_starts=1,
    )

    tour = list(map(int, tour))
    initial_length = tour_length(np.asarray(tour), distance_matrix)

    print(f"Initial NN length: {initial_length:.4f}")

    for action, name in ACTION_NAMES.items():
        result = apply_operator(
            action=action,
            tour=tour,
            distance_matrix=distance_matrix,
            seed=42,
            two_opt_iterations=50,
            max_trials=100,
            perturb_swaps=3,
        )

        print(
            f"{action} | {name:24s} | "
            f"length={result.length:.4f} | "
            f"improvement={result.improvement:.4f} | "
            f"improved={result.improved}"
        )


if __name__ == "__main__":
    main()