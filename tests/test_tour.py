import numpy as np

from tsp_hh.instances import generate_euclidean_instance
from tsp_hh.tour import validate_tour, tour_length, random_tour


def test_random_tour_is_valid():
    n_cities = 10
    tour = random_tour(n_cities, seed=42)

    assert validate_tour(tour, n_cities)


def test_invalid_tour_with_duplicate_city():
    tour = np.array([0, 1, 1, 3])

    assert not validate_tour(tour, n_cities=4)


def test_tour_length_square():
    coords = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0],
    ])

    from tsp_hh.instances import compute_distance_matrix

    distance_matrix = compute_distance_matrix(coords)
    tour = np.array([0, 1, 2, 3])

    length = tour_length(tour, distance_matrix)

    assert np.isclose(length, 4.0)


def test_generated_instance_shapes():
    instance = generate_euclidean_instance(n_cities=20, seed=42)

    assert instance.coords.shape == (20, 2)
    assert instance.distance_matrix.shape == (20, 20)