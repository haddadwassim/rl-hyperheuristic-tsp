import numpy as np

from src.tsp.instance import TSPInstance
from src.tsp.tour import tour_length


def apply_relocate_move(tour: np.ndarray, i: int, j: int) -> np.ndarray:
    """
    Move the city at position i and insert it before position j.

    If i == j, the tour is unchanged.
    """
    if i == j:
        return tour.copy()

    tour_list = tour.tolist()
    city = tour_list.pop(i)

    if j > i:
        j -= 1

    tour_list.insert(j, city)

    return np.asarray(tour_list, dtype=np.int64)


def relocate_best_improvement(
    tour: np.ndarray,
    instance: TSPInstance,
    max_trials: int | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, float, bool]:
    """
    Apply one best-improvement relocate step.
    """
    if rng is None:
        rng = np.random.default_rng()

    n = len(tour)
    current_length = tour_length(tour, instance)

    best_tour = tour.copy()
    best_length = current_length

    all_moves = [(i, j) for i in range(n) for j in range(n) if i != j]

    if max_trials is not None and max_trials < len(all_moves):
        selected_indices = rng.choice(len(all_moves), size=max_trials, replace=False)
        moves = [all_moves[idx] for idx in selected_indices]
    else:
        moves = all_moves

    for i, j in moves:
        candidate = apply_relocate_move(tour, i, j)
        candidate_length = tour_length(candidate, instance)

        if candidate_length < best_length:
            best_tour = candidate
            best_length = candidate_length

    improved = best_length < current_length

    return best_tour, best_length, improved