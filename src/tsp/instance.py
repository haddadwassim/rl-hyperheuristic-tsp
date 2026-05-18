from dataclasses import dataclass
import numpy as np


@dataclass
class TSPInstance:
    """
    Representation of a Euclidean Traveling Salesman Problem instance.

    Attributes
    ----------
    coordinates:
        Array of shape (n_nodes, 2), where each row represents the x and y
        coordinates of one city.

    distance_matrix:
        Array of shape (n_nodes, n_nodes), where entry (i, j) represents the
        Euclidean distance between city i and city j.
    """

    coordinates: np.ndarray
    distance_matrix: np.ndarray

    @property
    def num_nodes(self) -> int:
        return self.coordinates.shape[0]

    @classmethod
    def from_coordinates(cls, coordinates: np.ndarray) -> "TSPInstance":
        """
        Build a TSP instance from city coordinates.
        """
        coordinates = np.asarray(coordinates, dtype=np.float64)

        if coordinates.ndim != 2 or coordinates.shape[1] != 2:
            raise ValueError(
                "coordinates must be a NumPy array of shape (n_nodes, 2)."
            )

        distance_matrix = compute_euclidean_distance_matrix(coordinates)

        return cls(
            coordinates=coordinates,
            distance_matrix=distance_matrix,
        )


def compute_euclidean_distance_matrix(coordinates: np.ndarray) -> np.ndarray:
    """
    Compute the full pairwise Euclidean distance matrix.

    Parameters
    ----------
    coordinates:
        Array of shape (n_nodes, 2).

    Returns
    -------
    np.ndarray
        Pairwise distance matrix of shape (n_nodes, n_nodes).
    """
    diff = coordinates[:, None, :] - coordinates[None, :, :]
    distance_matrix = np.sqrt(np.sum(diff ** 2, axis=-1))
    return distance_matrix