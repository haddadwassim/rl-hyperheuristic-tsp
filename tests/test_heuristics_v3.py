import numpy as np

from tsp_hh.instances import generate_euclidean_instance
from tsp_hh.tour import validate_tour, tour_length, random_tour
from tsp_hh.heuristics_v3 import (
    greedy_multi_start_nearest_neighbor,
    construct_tour_v3,
    no_improvement,
    two_opt_local_search_limited,
    sampled_three_opt_move,
    three_opt_local_search_sampled,
    apply_improvement_v3,
    random_two_opt_perturbation,
    city_swap_perturbation,
    insertion_perturbation,
    apply_perturbation_v3,
)


def test_greedy_multi_start_nearest_neighbor_returns_valid_tour():
    instance = generate_euclidean_instance(n_cities=30, seed=42)

    tour = greedy_multi_start_nearest_neighbor(
        instance.distance_matrix,
        n_starts=5,
        seed=123,
    )

    assert validate_tour(tour, n_cities=30)


def test_construct_tour_v3_all_methods():
    instance = generate_euclidean_instance(n_cities=30, seed=42)

    for method in ["random", "nearest_neighbor", "greedy"]:
        tour = construct_tour_v3(
            instance.distance_matrix,
            method=method,
            seed=123,
            n_starts=5,
        )

        assert validate_tour(tour, n_cities=30)


def test_no_improvement_returns_same_length():
    instance = generate_euclidean_instance(n_cities=20, seed=42)
    tour = random_tour(20, seed=1)

    old_length = tour_length(tour, instance.distance_matrix)

    new_tour, improvement, n_success = no_improvement(
        tour,
        instance.distance_matrix,
    )

    new_length = tour_length(new_tour, instance.distance_matrix)

    assert validate_tour(new_tour, n_cities=20)
    assert np.isclose(old_length, new_length)
    assert improvement == 0.0
    assert n_success == 0


def test_two_opt_local_search_limited_does_not_worsen():
    instance = generate_euclidean_instance(n_cities=40, seed=42)
    tour = random_tour(40, seed=1)

    old_length = tour_length(tour, instance.distance_matrix)

    new_tour, improvement, n_success = two_opt_local_search_limited(
        tour,
        instance.distance_matrix,
        max_iterations=10,
    )

    new_length = tour_length(new_tour, instance.distance_matrix)

    assert validate_tour(new_tour, n_cities=40)
    assert new_length <= old_length
    assert improvement >= 0.0
    assert n_success >= 0


def test_sampled_three_opt_move_returns_valid_tour():
    tour = np.arange(20)

    new_tour = sampled_three_opt_move(tour, seed=42)

    assert validate_tour(new_tour, n_cities=20)


def test_three_opt_local_search_sampled_does_not_worsen():
    instance = generate_euclidean_instance(n_cities=40, seed=42)
    tour = random_tour(40, seed=1)

    old_length = tour_length(tour, instance.distance_matrix)

    new_tour, improvement, n_success = three_opt_local_search_sampled(
        tour,
        instance.distance_matrix,
        samples=20,
        seed=123,
    )

    new_length = tour_length(new_tour, instance.distance_matrix)

    assert validate_tour(new_tour, n_cities=40)
    assert new_length <= old_length
    assert improvement >= 0.0
    assert n_success in [0, 1]


def test_apply_improvement_v3_all_methods():
    instance = generate_euclidean_instance(n_cities=40, seed=42)
    tour = random_tour(40, seed=1)

    for method in ["none", "two_opt", "three_opt"]:
        new_tour, improvement, n_success = apply_improvement_v3(
            tour,
            instance.distance_matrix,
            method=method,
            seed=123,
            two_opt_iterations=5,
            three_opt_samples=10,
        )

        assert validate_tour(new_tour, n_cities=40)
        assert improvement >= 0.0
        assert n_success >= 0


def test_perturbation_methods_return_valid_tours():
    tour = np.arange(30)

    methods = [
        random_two_opt_perturbation,
        city_swap_perturbation,
        insertion_perturbation,
    ]

    for fn in methods:
        new_tour = fn(tour, seed=42)
        assert validate_tour(new_tour, n_cities=30)


def test_apply_perturbation_v3_all_methods():
    tour = np.arange(30)

    for method in ["random_2opt", "city_swap", "insertion"]:
        new_tour = apply_perturbation_v3(
            tour,
            method=method,
            seed=42,
        )

        assert validate_tour(new_tour, n_cities=30)