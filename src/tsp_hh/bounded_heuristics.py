import numpy as np

from tsp_hh.tour import tour_length, validate_tour
from tsp_hh.heuristics import (
    two_opt_move,
    random_swap_move,
    random_insertion_move,
    perturb_tour,
)


def random_two_opt_move(
    tour: np.ndarray,
    seed: int | None = None,
) -> np.ndarray:
    """
    Apply one random 2-opt move.

    This is a bounded alternative to full best-improvement 2-opt.
    """
    tour = np.asarray(tour, dtype=int)
    n_cities = len(tour)

    if n_cities < 4:
        raise ValueError("tour must contain at least 4 cities")

    rng = np.random.default_rng(seed)

    i, k = sorted(rng.choice(np.arange(1, n_cities), size=2, replace=False))

    if i == k:
        return tour.copy()

    return two_opt_move(tour, int(i), int(k))


def best_of_k_random_two_opt(
    tour: np.ndarray,
    distance_matrix: np.ndarray,
    k: int = 10,
    seed: int | None = None,
) -> tuple[np.ndarray, float]:
    """
    Sample k random 2-opt moves and return the best improving one.

    If no sampled move improves the solution, return the original tour.
    """
    if k < 1:
        raise ValueError("k must be at least 1")

    tour = np.asarray(tour, dtype=int)
    n_cities = len(tour)

    if not validate_tour(tour, n_cities):
        raise ValueError("Invalid tour")

    rng = np.random.default_rng(seed)

    current_length = tour_length(tour, distance_matrix)
    best_tour = tour.copy()
    best_length = current_length

    for _ in range(k):
        move_seed = int(rng.integers(0, 1_000_000_000))
        candidate = random_two_opt_move(tour, seed=move_seed)
        candidate_length = tour_length(candidate, distance_matrix)

        if candidate_length < best_length:
            best_length = candidate_length
            best_tour = candidate

    improvement = current_length - best_length
    return best_tour, float(improvement)


def best_of_k_random_swaps(
    tour: np.ndarray,
    distance_matrix: np.ndarray,
    k: int = 10,
    seed: int | None = None,
) -> tuple[np.ndarray, float]:
    """
    Sample k random swap moves and return the best improving one.

    If no sampled swap improves the solution, return the original tour.
    """
    if k < 1:
        raise ValueError("k must be at least 1")

    tour = np.asarray(tour, dtype=int)
    n_cities = len(tour)

    if not validate_tour(tour, n_cities):
        raise ValueError("Invalid tour")

    rng = np.random.default_rng(seed)

    current_length = tour_length(tour, distance_matrix)
    best_tour = tour.copy()
    best_length = current_length

    for _ in range(k):
        move_seed = int(rng.integers(0, 1_000_000_000))
        candidate = random_swap_move(tour, seed=move_seed)
        candidate_length = tour_length(candidate, distance_matrix)

        if candidate_length < best_length:
            best_length = candidate_length
            best_tour = candidate

    improvement = current_length - best_length
    return best_tour, float(improvement)


def best_of_k_random_insertions(
    tour: np.ndarray,
    distance_matrix: np.ndarray,
    k: int = 10,
    seed: int | None = None,
) -> tuple[np.ndarray, float]:
    """
    Sample k random insertion moves and return the best improving one.

    If no sampled insertion improves the solution, return the original tour.
    """
    if k < 1:
        raise ValueError("k must be at least 1")

    tour = np.asarray(tour, dtype=int)
    n_cities = len(tour)

    if not validate_tour(tour, n_cities):
        raise ValueError("Invalid tour")

    rng = np.random.default_rng(seed)

    current_length = tour_length(tour, distance_matrix)
    best_tour = tour.copy()
    best_length = current_length

    for _ in range(k):
        move_seed = int(rng.integers(0, 1_000_000_000))
        candidate = random_insertion_move(tour, seed=move_seed)
        candidate_length = tour_length(candidate, distance_matrix)

        if candidate_length < best_length:
            best_length = candidate_length
            best_tour = candidate

    improvement = current_length - best_length
    return best_tour, float(improvement)


def perturb_then_best_of_k_two_opt(
    tour: np.ndarray,
    distance_matrix: np.ndarray,
    perturbation_moves: int = 3,
    k: int = 10,
    seed: int | None = None,
) -> tuple[np.ndarray, float]:
    """
    Diversification action.

    First perturb the current tour, then apply bounded 2-opt sampling.
    The returned solution is accepted only if it improves the original tour.
    """
    tour = np.asarray(tour, dtype=int)
    n_cities = len(tour)

    if not validate_tour(tour, n_cities):
        raise ValueError("Invalid tour")

    rng = np.random.default_rng(seed)

    current_length = tour_length(tour, distance_matrix)

    perturb_seed = int(rng.integers(0, 1_000_000_000))
    search_seed = int(rng.integers(0, 1_000_000_000))

    perturbed = perturb_tour(
        tour,
        n_moves=perturbation_moves,
        seed=perturb_seed,
    )

    candidate, _ = best_of_k_random_two_opt(
        perturbed,
        distance_matrix,
        k=k,
        seed=search_seed,
    )

    candidate_length = tour_length(candidate, distance_matrix)

    if candidate_length < current_length:
        return candidate, float(current_length - candidate_length)

    return tour.copy(), 0.0