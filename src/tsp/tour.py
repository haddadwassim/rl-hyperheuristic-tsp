import numpy as np
from src.tsp.instance import TSPInstance


def validate_tour(tour: np.ndarray, instance: TSPInstance) -> None:
    """
    Validate that a tour is a permutation of all city indices.

    Parameters
    ----------
    tour:
        Array containing a permutation of city indices.

    instance:
        TSP instance.

    Raises
    ------
    ValueError
        If the tour is invalid.
    """
    tour = np.asarray(tour, dtype=np.int64)

    if tour.ndim != 1:
        raise ValueError("tour must be a one-dimensional array.")

    if len(tour) != instance.num_nodes:
        raise ValueError(
            f"tour length must be {instance.num_nodes}, got {len(tour)}."
        )

    expected = set(range(instance.num_nodes))
    actual = set(tour.tolist())

    if actual != expected:
        raise ValueError("tour must contain each city exactly once.")


def tour_length(tour: np.ndarray, instance: TSPInstance) -> float:
    """
    Compute the total length of a Hamiltonian cycle.

    The tour is interpreted as cyclic, meaning that the final city is connected
    back to the first city.
    """
    tour = np.asarray(tour, dtype=np.int64)
    validate_tour(tour, instance)

    distances = instance.distance_matrix
    next_tour = np.roll(tour, -1)

    return float(np.sum(distances[tour, next_tour]))


def nearest_neighbor_tour(
    instance: TSPInstance,
    start_city: int = 0,
) -> np.ndarray:
    """
    Construct an initial tour using the nearest-neighbor heuristic.

    Parameters
    ----------
    instance:
        TSP instance.

    start_city:
        City from which the greedy construction starts.

    Returns
    -------
    np.ndarray
        A valid TSP tour.
    """
    n = instance.num_nodes

    if start_city < 0 or start_city >= n:
        raise ValueError("start_city must be a valid city index.")

    unvisited = set(range(n))
    tour = [start_city]
    unvisited.remove(start_city)

    current = start_city

    while unvisited:
        next_city = min(
            unvisited,
            key=lambda city: instance.distance_matrix[current, city],
        )
        tour.append(next_city)
        unvisited.remove(next_city)
        current = next_city

    return np.asarray(tour, dtype=np.int64)


def random_tour(
    instance: TSPInstance,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Generate a random valid tour.
    """
    if rng is None:
        rng = np.random.default_rng()

    return rng.permutation(instance.num_nodes).astype(np.int64)


def copy_tour(tour: np.ndarray) -> np.ndarray:
    """
    Return a safe copy of a tour.
    """
    return np.asarray(tour, dtype=np.int64).copy()