import numpy as np

from tsp_hh.instances import generate_euclidean_instance
from tsp_hh.tour import validate_tour, tour_length, random_tour
from tsp_hh.heuristics import (
    nearest_neighbor_tour,
    two_opt_move,
    first_improvement_two_opt,
    best_improvement_two_opt,
    random_swap_move,
    random_insertion_move,
    perturb_tour,
    repeated_two_opt,
    create_initial_tour,
)


def test_nearest_neighbor_returns_valid_tour():
    instance = generate_euclidean_instance(n_cities=20, seed=42)

    tour = nearest_neighbor_tour(instance.distance_matrix, start_city=0)

    assert validate_tour(tour, n_cities=20)


def test_two_opt_move_returns_valid_tour():
    tour = np.array([0, 1, 2, 3, 4])

    new_tour = two_opt_move(tour, i=1, k=3)

    assert validate_tour(new_tour, n_cities=5)
    assert new_tour.tolist() == [0, 3, 2, 1, 4]


def test_first_improvement_two_opt_does_not_worsen_solution():
    instance = generate_euclidean_instance(n_cities=30, seed=1)
    tour = random_tour(n_cities=30, seed=2)

    old_length = tour_length(tour, instance.distance_matrix)
    new_tour, improvement = first_improvement_two_opt(tour, instance.distance_matrix)
    new_length = tour_length(new_tour, instance.distance_matrix)

    assert validate_tour(new_tour, n_cities=30)
    assert new_length <= old_length
    assert improvement >= 0


def test_best_improvement_two_opt_does_not_worsen_solution():
    instance = generate_euclidean_instance(n_cities=30, seed=1)
    tour = random_tour(n_cities=30, seed=2)

    old_length = tour_length(tour, instance.distance_matrix)
    new_tour, improvement = best_improvement_two_opt(tour, instance.distance_matrix)
    new_length = tour_length(new_tour, instance.distance_matrix)

    assert validate_tour(new_tour, n_cities=30)
    assert new_length <= old_length
    assert improvement >= 0


def test_random_swap_move_returns_valid_tour():
    tour = np.arange(10)

    new_tour = random_swap_move(tour, seed=42)

    assert validate_tour(new_tour, n_cities=10)


def test_random_insertion_move_returns_valid_tour():
    tour = np.arange(10)

    new_tour = random_insertion_move(tour, seed=42)

    assert validate_tour(new_tour, n_cities=10)


def test_perturb_tour_returns_valid_tour():
    tour = np.arange(15)

    new_tour = perturb_tour(tour, n_moves=5, seed=42)

    assert validate_tour(new_tour, n_cities=15)


def test_repeated_two_opt_does_not_worsen_solution():
    instance = generate_euclidean_instance(n_cities=40, seed=3)
    tour = random_tour(n_cities=40, seed=4)

    old_length = tour_length(tour, instance.distance_matrix)

    new_tour, total_improvement, n_iterations = repeated_two_opt(
        tour,
        instance.distance_matrix,
        max_iterations=50,
    )

    new_length = tour_length(new_tour, instance.distance_matrix)

    assert validate_tour(new_tour, n_cities=40)
    assert new_length <= old_length
    assert total_improvement >= 0
    assert n_iterations >= 0


def test_create_initial_tour_nearest_neighbor():
    instance = generate_euclidean_instance(n_cities=20, seed=42)

    tour = create_initial_tour(instance.distance_matrix, method="nearest_neighbor")

    assert validate_tour(tour, n_cities=20)


def test_create_initial_tour_random():
    instance = generate_euclidean_instance(n_cities=20, seed=42)

    tour = create_initial_tour(
        instance.distance_matrix,
        method="random",
        seed=123,
    )

    assert validate_tour(tour, n_cities=20)