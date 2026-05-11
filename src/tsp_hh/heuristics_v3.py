import numpy as np

from tsp_hh.tour import tour_length, validate_tour, random_tour
from tsp_hh.heuristics import (
    nearest_neighbor_tour,
    first_improvement_two_opt,
    random_swap_move,
    random_insertion_move,
    two_opt_move,
)


# ============================================================
# Construction heuristics
# ============================================================

def greedy_multi_start_nearest_neighbor(
    distance_matrix: np.ndarray,
    n_starts: int = 10,
    seed: int | None = None,
) -> np.ndarray:
    """
    Greedy construction using multi-start nearest neighbor.

    Several starting cities are sampled, nearest-neighbor tours are built,
    and the best one is returned.
    """
    n_cities = distance_matrix.shape[0]

    if n_starts < 1:
        raise ValueError("n_starts must be at least 1")

    rng = np.random.default_rng(seed)

    n_starts = min(n_starts, n_cities)
    start_cities = rng.choice(n_cities, size=n_starts, replace=False)

    best_tour = None
    best_length = float("inf")

    for start_city in start_cities:
        tour = nearest_neighbor_tour(
            distance_matrix,
            start_city=int(start_city),
        )

        length = tour_length(tour, distance_matrix)

        if length < best_length:
            best_length = length
            best_tour = tour

    return best_tour


def construct_tour_v3(
    distance_matrix: np.ndarray,
    method: str,
    seed: int | None = None,
    n_starts: int = 10,
) -> np.ndarray:
    """
    Construct a tour using one of the V3 construction heuristics.

    Supported methods:
    - random
    - nearest_neighbor
    - greedy
    """
    n_cities = distance_matrix.shape[0]

    if method == "random":
        return random_tour(n_cities, seed=seed)

    if method == "nearest_neighbor":
        return nearest_neighbor_tour(distance_matrix, start_city=0)

    if method == "greedy":
        return greedy_multi_start_nearest_neighbor(
            distance_matrix,
            n_starts=n_starts,
            seed=seed,
        )

    raise ValueError(f"Unknown construction method: {method}")


# ============================================================
# Improvement heuristics
# ============================================================

def no_improvement(
    tour: np.ndarray,
    distance_matrix: np.ndarray,
) -> tuple[np.ndarray, float, int]:
    """
    Do nothing.

    Useful as a valid improvement heuristic when the controller decides
    not to intensify.
    """
    tour = np.asarray(tour, dtype=int)

    return tour.copy(), 0.0, 0


def two_opt_local_search_limited(
    tour: np.ndarray,
    distance_matrix: np.ndarray,
    max_iterations: int = 20,
) -> tuple[np.ndarray, float, int]:
    """
    Limited repeated first-improvement 2-opt local search.

    Returns:
    - improved tour
    - total improvement
    - number of successful improving moves
    """
    if max_iterations < 1:
        raise ValueError("max_iterations must be at least 1")

    current_tour = np.asarray(tour, dtype=int).copy()
    n_cities = len(current_tour)

    if not validate_tour(current_tour, n_cities):
        raise ValueError("Invalid tour")

    initial_length = tour_length(current_tour, distance_matrix)
    n_success = 0

    for _ in range(max_iterations):
        new_tour, improvement = first_improvement_two_opt(
            current_tour,
            distance_matrix,
        )

        if improvement <= 0:
            break

        current_tour = new_tour
        n_success += 1

    final_length = tour_length(current_tour, distance_matrix)

    return current_tour, float(initial_length - final_length), n_success


def sampled_three_opt_move(
    tour: np.ndarray,
    seed: int | None = None,
) -> np.ndarray:
    """
    Apply one sampled 3-opt-like reconnection.

    This is not a full exhaustive 3-opt. It samples three cut points,
    divides the tour into four segments, and reconnects them using one
    of a few simple segment reversals/reorderings.

    It is intentionally bounded and cheap.
    """
    tour = np.asarray(tour, dtype=int)
    n_cities = len(tour)

    if n_cities < 6:
        raise ValueError("tour must contain at least 6 cities")

    rng = np.random.default_rng(seed)

    cuts = sorted(rng.choice(np.arange(1, n_cities), size=3, replace=False))
    i, j, k = map(int, cuts)

    a = tour[:i]
    b = tour[i:j]
    c = tour[j:k]
    d = tour[k:]

    variants = [
        np.concatenate([a, b[::-1], c, d]),
        np.concatenate([a, b, c[::-1], d]),
        np.concatenate([a, c, b, d]),
        np.concatenate([a, c[::-1], b, d]),
        np.concatenate([a, b[::-1], c[::-1], d]),
        np.concatenate([a, c, b[::-1], d]),
    ]

    idx = int(rng.integers(0, len(variants)))
    return variants[idx].astype(int)


def three_opt_local_search_sampled(
    tour: np.ndarray,
    distance_matrix: np.ndarray,
    samples: int = 30,
    seed: int | None = None,
) -> tuple[np.ndarray, float, int]:
    """
    Bounded sampled 3-opt local search.

    Samples several 3-opt-like moves and returns the best improving one.
    This is a simple bounded approximation, not full 3-opt.
    """
    if samples < 1:
        raise ValueError("samples must be at least 1")

    tour = np.asarray(tour, dtype=int)
    n_cities = len(tour)

    if not validate_tour(tour, n_cities):
        raise ValueError("Invalid tour")

    rng = np.random.default_rng(seed)

    current_length = tour_length(tour, distance_matrix)
    best_tour = tour.copy()
    best_length = current_length

    for _ in range(samples):
        move_seed = int(rng.integers(0, 1_000_000_000))
        candidate = sampled_three_opt_move(tour, seed=move_seed)
        candidate_length = tour_length(candidate, distance_matrix)

        if candidate_length < best_length:
            best_tour = candidate
            best_length = candidate_length

    improvement = current_length - best_length

    if improvement > 0:
        return best_tour, float(improvement), 1

    return tour.copy(), 0.0, 0


def apply_improvement_v3(
    tour: np.ndarray,
    distance_matrix: np.ndarray,
    method: str,
    seed: int | None = None,
    two_opt_iterations: int = 20,
    three_opt_samples: int = 30,
) -> tuple[np.ndarray, float, int]:
    """
    Apply one V3 improvement heuristic.

    Supported methods:
    - none
    - two_opt
    - three_opt
    """
    if method == "none":
        return no_improvement(tour, distance_matrix)

    if method == "two_opt":
        return two_opt_local_search_limited(
            tour,
            distance_matrix,
            max_iterations=two_opt_iterations,
        )

    if method == "three_opt":
        return three_opt_local_search_sampled(
            tour,
            distance_matrix,
            samples=three_opt_samples,
            seed=seed,
        )

    raise ValueError(f"Unknown improvement method: {method}")


# ============================================================
# Perturbation heuristics
# ============================================================

def random_two_opt_perturbation(
    tour: np.ndarray,
    seed: int | None = None,
) -> np.ndarray:
    """
    Apply one random 2-opt perturbation.
    """
    tour = np.asarray(tour, dtype=int)
    n_cities = len(tour)

    if n_cities < 4:
        raise ValueError("tour must contain at least 4 cities")

    rng = np.random.default_rng(seed)
    i, k = sorted(rng.choice(np.arange(1, n_cities), size=2, replace=False))

    return two_opt_move(tour, int(i), int(k))


def city_swap_perturbation(
    tour: np.ndarray,
    seed: int | None = None,
) -> np.ndarray:
    """
    Apply one city swap perturbation.
    """
    return random_swap_move(tour, seed=seed)


def insertion_perturbation(
    tour: np.ndarray,
    seed: int | None = None,
) -> np.ndarray:
    """
    Apply one insertion perturbation.
    """
    return random_insertion_move(tour, seed=seed)


def apply_perturbation_v3(
    tour: np.ndarray,
    method: str,
    seed: int | None = None,
) -> np.ndarray:
    """
    Apply one V3 perturbation heuristic.

    Supported methods:
    - random_2opt
    - city_swap
    - insertion
    """
    if method == "random_2opt":
        return random_two_opt_perturbation(tour, seed=seed)

    if method == "city_swap":
        return city_swap_perturbation(tour, seed=seed)

    if method == "insertion":
        return insertion_perturbation(tour, seed=seed)

    raise ValueError(f"Unknown perturbation method: {method}")