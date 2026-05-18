import numpy as np
from src.tsp.instance import TSPInstance


def generate_uniform_instance(
    num_nodes: int,
    low: float = 0.0,
    high: float = 1.0,
    seed: int | None = None,
) -> TSPInstance:
    """
    Generate a random Euclidean TSP instance with uniformly distributed cities.
    """
    if num_nodes <= 2:
        raise ValueError("num_nodes must be greater than 2.")

    rng = np.random.default_rng(seed)
    coordinates = rng.uniform(low=low, high=high, size=(num_nodes, 2))

    return TSPInstance.from_coordinates(coordinates)


def generate_clustered_instance(
    num_nodes: int,
    num_clusters: int = 3,
    cluster_std: float = 0.05,
    seed: int | None = None,
) -> TSPInstance:
    """
    Generate a clustered Euclidean TSP instance.

    This distribution is useful for testing generalization because it differs
    from the standard uniform random distribution.
    """
    if num_nodes <= 2:
        raise ValueError("num_nodes must be greater than 2.")

    if num_clusters <= 0:
        raise ValueError("num_clusters must be positive.")

    rng = np.random.default_rng(seed)

    centers = rng.uniform(0.1, 0.9, size=(num_clusters, 2))

    cluster_ids = rng.integers(0, num_clusters, size=num_nodes)

    coordinates = centers[cluster_ids] + rng.normal(
        loc=0.0,
        scale=cluster_std,
        size=(num_nodes, 2),
    )

    coordinates = np.clip(coordinates, 0.0, 1.0)

    return TSPInstance.from_coordinates(coordinates)


def generate_grid_instance(
    num_nodes: int,
    noise: float = 0.01,
    seed: int | None = None,
) -> TSPInstance:
    """
    Generate a grid-like TSP instance with small coordinate noise.
    """
    if num_nodes <= 2:
        raise ValueError("num_nodes must be greater than 2.")

    rng = np.random.default_rng(seed)

    grid_size = int(np.ceil(np.sqrt(num_nodes)))

    xs, ys = np.meshgrid(
        np.linspace(0.0, 1.0, grid_size),
        np.linspace(0.0, 1.0, grid_size),
    )

    coordinates = np.column_stack([xs.ravel(), ys.ravel()])
    coordinates = coordinates[:num_nodes]

    coordinates += rng.normal(0.0, noise, size=coordinates.shape)
    coordinates = np.clip(coordinates, 0.0, 1.0)

    rng.shuffle(coordinates)

    return TSPInstance.from_coordinates(coordinates)


def generate_instance(
    distribution: str,
    num_nodes: int,
    seed: int | None = None,
) -> TSPInstance:
    """
    General instance generator used by training and evaluation scripts.
    """
    distribution = distribution.lower()

    if distribution == "uniform":
        return generate_uniform_instance(num_nodes=num_nodes, seed=seed)

    if distribution == "clustered":
        return generate_clustered_instance(num_nodes=num_nodes, seed=seed)

    if distribution == "grid":
        return generate_grid_instance(num_nodes=num_nodes, seed=seed)

    raise ValueError(f"Unknown TSP distribution: {distribution}")