import numpy as np

from tsp_hh.instances import generate_euclidean_instance
from tsp_hh.exact_tsp import brute_force_optimal_tour, held_karp_optimal_tour


def test_brute_force_optimal_tour_valid():
    instance = generate_euclidean_instance(n_cities=7, seed=42)

    tour, length = brute_force_optimal_tour(
        instance.distance_matrix,
        start_city=0,
    )

    assert sorted(tour) == list(range(7))
    assert tour[0] == 0
    assert length > 0


def test_held_karp_matches_brute_force():
    instance = generate_euclidean_instance(n_cities=8, seed=42)

    brute_tour, brute_length = brute_force_optimal_tour(
        instance.distance_matrix,
        start_city=0,
    )

    hk_tour, hk_length = held_karp_optimal_tour(
        instance.distance_matrix,
        start_city=0,
    )

    assert sorted(hk_tour) == list(range(8))
    assert hk_tour[0] == 0
    assert np.isclose(hk_length, brute_length)