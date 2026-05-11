import numpy as np

from tsp_hh.instances import generate_euclidean_instance
from tsp_hh.tour import validate_tour, tour_length, random_tour
from tsp_hh.bounded_heuristics import (
    random_two_opt_move,
    best_of_k_random_two_opt,
    best_of_k_random_swaps,
    best_of_k_random_insertions,
    perturb_then_best_of_k_two_opt,
)


def test_random_two_opt_move_returns_valid_tour():
    tour = np.arange(20)

    new_tour = random_two_opt_move(tour, seed=42)

    assert validate_tour(new_tour, n_cities=20)


def test_best_of_k_random_two_opt_returns_valid_non_worse_tour():
    instance = generate_euclidean_instance(n_cities=30, seed=42)
    tour = random_tour(30, seed=1)

    old_length = tour_length(tour, instance.distance_matrix)

    new_tour, improvement = best_of_k_random_two_opt(
        tour,
        instance.distance_matrix,
        k=10,
        seed=42,
    )

    new_length = tour_length(new_tour, instance.distance_matrix)

    assert validate_tour(new_tour, n_cities=30)
    assert new_length <= old_length
    assert improvement >= 0.0


def test_best_of_k_random_swaps_returns_valid_non_worse_tour():
    instance = generate_euclidean_instance(n_cities=30, seed=42)
    tour = random_tour(30, seed=1)

    old_length = tour_length(tour, instance.distance_matrix)

    new_tour, improvement = best_of_k_random_swaps(
        tour,
        instance.distance_matrix,
        k=10,
        seed=42,
    )

    new_length = tour_length(new_tour, instance.distance_matrix)

    assert validate_tour(new_tour, n_cities=30)
    assert new_length <= old_length
    assert improvement >= 0.0


def test_best_of_k_random_insertions_returns_valid_non_worse_tour():
    instance = generate_euclidean_instance(n_cities=30, seed=42)
    tour = random_tour(30, seed=1)

    old_length = tour_length(tour, instance.distance_matrix)

    new_tour, improvement = best_of_k_random_insertions(
        tour,
        instance.distance_matrix,
        k=10,
        seed=42,
    )

    new_length = tour_length(new_tour, instance.distance_matrix)

    assert validate_tour(new_tour, n_cities=30)
    assert new_length <= old_length
    assert improvement >= 0.0


def test_perturb_then_best_of_k_two_opt_returns_valid_non_worse_tour():
    instance = generate_euclidean_instance(n_cities=30, seed=42)
    tour = random_tour(30, seed=1)

    old_length = tour_length(tour, instance.distance_matrix)

    new_tour, improvement = perturb_then_best_of_k_two_opt(
        tour,
        instance.distance_matrix,
        perturbation_moves=3,
        k=10,
        seed=42,
    )

    new_length = tour_length(new_tour, instance.distance_matrix)

    assert validate_tour(new_tour, n_cities=30)
    assert new_length <= old_length
    assert improvement >= 0.0