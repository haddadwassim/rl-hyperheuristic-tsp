import numpy as np

from src.tsp.instance import TSPInstance
from src.tsp.tour import tour_length


def apply_swap_move(tour: np.ndarray, i: int, j: int) -> np.ndarray:
    """
    Swap the cities at positions i and j.
    """
    new_tour = tour.copy()
    new_tour[i], new_tour[j] = new_tour[j], new_tour[i]
    return new_tour


def swap_best_improvement(
    tour: np.ndarray,
    instance: TSPInstance,
    max_trials: int | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, float, bool]:
    """
    Apply one best-improvement swap step.
    """
    if rng is None:
        rng = np.random.default_rng()

    n = len(tour)
    current_length = tour_length(tour, instance)

    best_tour = tour.copy()
    best_length = current_length

    all_moves = [(i, j) for i in range(n - 1) for j in range(i + 1, n)]

    if max_trials is not None and max_trials < len(all_moves):
        selected_indices = rng.choice(len(all_moves), size=max_trials, replace=False)
        moves = [all_moves[idx] for idx in selected_indices]
    else:
        moves = all_moves

    for i, j in moves:
        candidate = apply_swap_move(tour, i, j)
        candidate_length = tour_length(candidate, instance)

        if candidate_length < best_length:
            best_tour = candidate
            best_length = candidate_length

    improved = best_length < current_length

    return best_tour, best_length, improved