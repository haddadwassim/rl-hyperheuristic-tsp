import numpy as np
from src.tsp.instance import TSPInstance
from src.tsp.tour import tour_length


def relative_improvement(
    initial_length: float,
    final_length: float,
) -> float:
    """
    Compute relative improvement from an initial solution to a final solution.
    """
    if initial_length <= 0:
        raise ValueError("initial_length must be positive.")

    return float((initial_length - final_length) / initial_length)


def optimality_gap(
    method_length: float,
    reference_length: float,
) -> float:
    """
    Compute percentage gap relative to a reference or best-known solution.

    Gap = ((method - reference) / reference) * 100
    """
    if reference_length <= 0:
        raise ValueError("reference_length must be positive.")

    return float(((method_length - reference_length) / reference_length) * 100.0)


def edge_lengths(
    tour: np.ndarray,
    instance: TSPInstance,
) -> np.ndarray:
    """
    Return the list of edge lengths appearing in the cyclic tour.
    """
    tour = np.asarray(tour, dtype=np.int64)
    next_tour = np.roll(tour, -1)

    return instance.distance_matrix[tour, next_tour]


def tour_edge_statistics(
    tour: np.ndarray,
    instance: TSPInstance,
) -> dict:
    """
    Compute simple statistics of the edge lengths in a tour.

    These features will later be useful for the DRL state representation.
    """
    lengths = edge_lengths(tour, instance)

    return {
        "tour_length": tour_length(tour, instance),
        "mean_edge_length": float(np.mean(lengths)),
        "std_edge_length": float(np.std(lengths)),
        "min_edge_length": float(np.min(lengths)),
        "max_edge_length": float(np.max(lengths)),
        "median_edge_length": float(np.median(lengths)),
    }


def count_crossing_edges(
    tour: np.ndarray,
    instance: TSPInstance,
) -> int:
    """
    Count the number of crossing edge pairs in a 2D Euclidean tour.

    This is mainly used as an interpretable feature. It is O(n^2), so it should
    be used carefully for large instances.
    """
    coordinates = instance.coordinates
    tour = np.asarray(tour, dtype=np.int64)
    n = len(tour)

    count = 0

    for i in range(n):
        a = coordinates[tour[i]]
        b = coordinates[tour[(i + 1) % n]]

        for j in range(i + 2, n):
            # Adjacent edges share a city and should not be counted.
            if i == 0 and j == n - 1:
                continue

            c = coordinates[tour[j]]
            d = coordinates[tour[(j + 1) % n]]

            if segments_intersect(a, b, c, d):
                count += 1

    return count


def segments_intersect(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    d: np.ndarray,
) -> bool:
    """
    Check whether line segments AB and CD intersect.
    """

    def orientation(p: np.ndarray, q: np.ndarray, r: np.ndarray) -> float:
        return float((q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1]))

    o1 = orientation(a, b, c)
    o2 = orientation(a, b, d)
    o3 = orientation(c, d, a)
    o4 = orientation(c, d, b)

    return (o1 * o2 < 0) and (o3 * o4 < 0)