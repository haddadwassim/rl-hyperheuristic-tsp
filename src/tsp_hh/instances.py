from dataclasses import dataclass
import numpy as np


@dataclass
class TSPInstance:
    """
    Represents a Euclidean TSP instance.

    Attributes
    ----------
    coords:
        Array of shape (n_cities, 2), where each row is the (x, y)
        coordinate of one city.
    distance_matrix:
        Array of shape (n_cities, n_cities), where entry [i, j]
        is the Euclidean distance between city i and city j.
    """
    coords: np.ndarray
    distance_matrix: np.ndarray


def compute_distance_matrix(coords: np.ndarray) -> np.ndarray:
    """
    Compute the Euclidean distance matrix for a set of 2D coordinates.
    """
    coords = np.asarray(coords, dtype=float)

    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError("coords must have shape (n_cities, 2)")

    diff = coords[:, None, :] - coords[None, :, :]
    distance_matrix = np.sqrt(np.sum(diff ** 2, axis=-1))

    return distance_matrix


def generate_euclidean_instance(
    n_cities: int,
    seed: int | None = None,
    scale: float = 100.0,
) -> TSPInstance:
    """
    Generate a random Euclidean TSP instance.

    Cities are sampled uniformly in a square [0, scale] x [0, scale].
    """
    if n_cities < 3:
        raise ValueError("n_cities must be at least 3")

    rng = np.random.default_rng(seed)
    coords = rng.uniform(0, scale, size=(n_cities, 2))
    distance_matrix = compute_distance_matrix(coords)

    return TSPInstance(coords=coords, distance_matrix=distance_matrix)