import numpy as np


def validate_tour(tour: np.ndarray, n_cities: int) -> bool:
    """
    Check whether a tour is a valid permutation of all city indices.

    A valid tour must:
    - contain exactly n_cities elements
    - contain each city index from 0 to n_cities - 1 exactly once
    """
    tour = np.asarray(tour)

    if len(tour) != n_cities:
        return False

    return set(tour.tolist()) == set(range(n_cities))


def tour_length(tour: np.ndarray, distance_matrix: np.ndarray) -> float:
    """
    Compute the total length of a closed TSP tour.

    Example:
    If tour = [0, 2, 1], the cost is:

        d(0, 2) + d(2, 1) + d(1, 0)
    """
    tour = np.asarray(tour, dtype=int)
    n_cities = distance_matrix.shape[0]

    if distance_matrix.shape != (n_cities, n_cities):
        raise ValueError("distance_matrix must be square")

    if not validate_tour(tour, n_cities):
        raise ValueError("Invalid tour")

    next_cities = np.roll(tour, -1)
    return float(distance_matrix[tour, next_cities].sum())


def random_tour(n_cities: int, seed: int | None = None) -> np.ndarray:
    """
    Generate a random valid TSP tour.
    """
    if n_cities < 3:
        raise ValueError("n_cities must be at least 3")

    rng = np.random.default_rng(seed)
    return rng.permutation(n_cities)