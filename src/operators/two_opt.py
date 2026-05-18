import numpy as np

from src.tsp.instance import TSPInstance
from src.tsp.tour import tour_length


def apply_two_opt_move(tour: np.ndarray, i: int, k: int) -> np.ndarray:
    """
    Apply a 2-opt move by reversing the segment between positions i and k.
    """
    new_tour = tour.copy()
    new_tour[i : k + 1] = new_tour[i : k + 1][::-1]
    return new_tour


def two_opt_best_improvement(
    tour: np.ndarray,
    instance: TSPInstance,
    max_trials: int | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, float, bool]:
    """
    Apply one best-improvement 2-opt step.

    Returns
    -------
    new_tour:
        Tour after applying the best found 2-opt move.

    new_length:
        Length of the returned tour.

    improved:
        True if an improving move was found.
    """
    if rng is None:
        rng = np.random.default_rng()

    n = len(tour)
    current_length = tour_length(tour, instance)

    best_tour = tour.copy()
    best_length = current_length

    all_moves = [(i, k) for i in range(1, n - 1) for k in range(i + 1, n)]

    if max_trials is not None and max_trials < len(all_moves):
        selected_indices = rng.choice(len(all_moves), size=max_trials, replace=False)
        moves = [all_moves[idx] for idx in selected_indices]
    else:
        moves = all_moves

    for i, k in moves:
        candidate = apply_two_opt_move(tour, i, k)
        candidate_length = tour_length(candidate, instance)

        if candidate_length < best_length:
            best_tour = candidate
            best_length = candidate_length

    improved = best_length < current_length

    return best_tour, best_length, improved


def two_opt_first_improvement(
    tour: np.ndarray,
    instance: TSPInstance,
    max_trials: int | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, float, bool]:
    """
    Apply one first-improvement 2-opt step.
    """
    if rng is None:
        rng = np.random.default_rng()

    n = len(tour)
    current_length = tour_length(tour, instance)

    all_moves = [(i, k) for i in range(1, n - 1) for k in range(i + 1, n)]
    rng.shuffle(all_moves)

    if max_trials is not None:
        all_moves = all_moves[:max_trials]

    for i, k in all_moves:
        candidate = apply_two_opt_move(tour, i, k)
        candidate_length = tour_length(candidate, instance)

        if candidate_length < current_length:
            return candidate, candidate_length, True

    return tour.copy(), current_length, False