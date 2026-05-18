import numpy as np

from src.tsp.generators import generate_uniform_instance
from src.tsp.tour import nearest_neighbor_tour, tour_length
from src.operators.full_two_opt import full_two_opt, perturb_then_full_two_opt


def main():
    rng = np.random.default_rng(42)

    instance = generate_uniform_instance(num_nodes=50, seed=42)
    tour = nearest_neighbor_tour(instance)

    initial_length = tour_length(tour, instance)

    full_tour, full_length, full_improved = full_two_opt(
        tour=tour,
        instance=instance,
        max_passes=1000,
        rng=rng,
    )

    perturb_tour, perturb_length, perturb_improved = perturb_then_full_two_opt(
        tour=full_tour,
        instance=instance,
        perturb_strength=1,
        max_passes=1000,
        rng=rng,
    )

    print("Initial NN length:", initial_length)
    print("After full 2-opt:", full_length, "improved:", full_improved)
    print("After perturb + full 2-opt:", perturb_length, "improved:", perturb_improved)


if __name__ == "__main__":
    main()