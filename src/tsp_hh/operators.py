import random
from dataclasses import dataclass

import numpy as np

from tsp_hh.tour import tour_length
from tsp_hh.heuristics_v3 import two_opt_local_search_limited


@dataclass
class OperatorResult:
    tour: list[int]
    length: float
    improved: bool
    improvement: float
    operator_name: str


def _as_list(tour) -> list[int]:
    return list(map(int, tour))


def _result(
    old_tour: list[int],
    new_tour: list[int],
    distance_matrix: np.ndarray,
    operator_name: str,
) -> OperatorResult:
    old_length = tour_length(np.asarray(old_tour, dtype=int), distance_matrix)
    new_length = tour_length(np.asarray(new_tour, dtype=int), distance_matrix)

    improvement = old_length - new_length

    return OperatorResult(
        tour=_as_list(new_tour),
        length=float(new_length),
        improved=improvement > 1e-12,
        improvement=float(improvement),
        operator_name=operator_name,
    )


def apply_2opt_limited(
    tour: list[int],
    distance_matrix: np.ndarray,
    max_iterations: int = 50,
) -> OperatorResult:
    old_tour = _as_list(tour)

    new_tour, _, _ = two_opt_local_search_limited(
        np.asarray(old_tour, dtype=int),
        distance_matrix,
        max_iterations=max_iterations,
    )

    return _result(
        old_tour=old_tour,
        new_tour=_as_list(new_tour),
        distance_matrix=distance_matrix,
        operator_name="2opt_limited",
    )


def apply_random_swap_best_of_k(
    tour: list[int],
    distance_matrix: np.ndarray,
    max_trials: int = 100,
    seed: int | None = None,
) -> OperatorResult:
    rng = random.Random(seed)

    old_tour = _as_list(tour)
    n = len(old_tour)

    old_length = tour_length(np.asarray(old_tour, dtype=int), distance_matrix)

    best_tour = old_tour[:]
    best_length = old_length

    for _ in range(max_trials):
        i, j = rng.sample(range(1, n), 2)

        candidate = old_tour[:]
        candidate[i], candidate[j] = candidate[j], candidate[i]

        length = tour_length(np.asarray(candidate, dtype=int), distance_matrix)

        if length < best_length:
            best_tour = candidate
            best_length = length

    return OperatorResult(
        tour=best_tour,
        length=float(best_length),
        improved=(old_length - best_length) > 1e-12,
        improvement=float(old_length - best_length),
        operator_name="swap_best_of_k",
    )


def apply_random_relocate_best_of_k(
    tour: list[int],
    distance_matrix: np.ndarray,
    max_trials: int = 100,
    seed: int | None = None,
) -> OperatorResult:
    rng = random.Random(seed)

    old_tour = _as_list(tour)
    n = len(old_tour)

    old_length = tour_length(np.asarray(old_tour, dtype=int), distance_matrix)

    best_tour = old_tour[:]
    best_length = old_length

    for _ in range(max_trials):
        i, j = rng.sample(range(1, n), 2)

        candidate = old_tour[:]
        city = candidate.pop(i)
        candidate.insert(j, city)

        length = tour_length(np.asarray(candidate, dtype=int), distance_matrix)

        if length < best_length:
            best_tour = candidate
            best_length = length

    return OperatorResult(
        tour=best_tour,
        length=float(best_length),
        improved=(old_length - best_length) > 1e-12,
        improvement=float(old_length - best_length),
        operator_name="relocate_best_of_k",
    )


def apply_random_2opt_move_best_of_k(
    tour: list[int],
    distance_matrix: np.ndarray,
    max_trials: int = 100,
    seed: int | None = None,
) -> OperatorResult:
    rng = random.Random(seed)

    old_tour = _as_list(tour)
    n = len(old_tour)

    old_length = tour_length(np.asarray(old_tour, dtype=int), distance_matrix)

    best_tour = old_tour[:]
    best_length = old_length

    for _ in range(max_trials):
        i, j = sorted(rng.sample(range(1, n), 2))

        if j <= i + 1:
            continue

        candidate = old_tour[:]
        candidate[i:j] = reversed(candidate[i:j])

        length = tour_length(np.asarray(candidate, dtype=int), distance_matrix)

        if length < best_length:
            best_tour = candidate
            best_length = length

    return OperatorResult(
        tour=best_tour,
        length=float(best_length),
        improved=(old_length - best_length) > 1e-12,
        improvement=float(old_length - best_length),
        operator_name="random_2opt_best_of_k",
    )


def apply_perturb_then_2opt(
    tour: list[int],
    distance_matrix: np.ndarray,
    perturb_swaps: int = 3,
    two_opt_iterations: int = 50,
    seed: int | None = None,
) -> OperatorResult:
    rng = random.Random(seed)

    old_tour = _as_list(tour)
    n = len(old_tour)

    candidate = old_tour[:]

    for _ in range(perturb_swaps):
        i, j = rng.sample(range(1, n), 2)
        candidate[i], candidate[j] = candidate[j], candidate[i]

    repaired, _, _ = two_opt_local_search_limited(
        np.asarray(candidate, dtype=int),
        distance_matrix,
        max_iterations=two_opt_iterations,
    )

    return _result(
        old_tour=old_tour,
        new_tour=_as_list(repaired),
        distance_matrix=distance_matrix,
        operator_name="perturb_then_2opt",
    )


def apply_operator(
    action: int,
    tour: list[int],
    distance_matrix: np.ndarray,
    seed: int | None = None,
    two_opt_iterations: int = 50,
    max_trials: int = 100,
    perturb_swaps: int = 3,
) -> OperatorResult:
    """
    Action space:
      0 = STOP
      1 = 2-opt limited
      2 = random 2-opt move best-of-k
      3 = relocate best-of-k
      4 = swap best-of-k
      5 = perturb + 2-opt
    """
    old_tour = _as_list(tour)
    old_length = tour_length(np.asarray(old_tour, dtype=int), distance_matrix)

    if action == 0:
        return OperatorResult(
            tour=old_tour,
            length=float(old_length),
            improved=False,
            improvement=0.0,
            operator_name="stop",
        )

    if action == 1:
        return apply_2opt_limited(
            tour=old_tour,
            distance_matrix=distance_matrix,
            max_iterations=two_opt_iterations,
        )

    if action == 2:
        return apply_random_2opt_move_best_of_k(
            tour=old_tour,
            distance_matrix=distance_matrix,
            max_trials=max_trials,
            seed=seed,
        )

    if action == 3:
        return apply_random_relocate_best_of_k(
            tour=old_tour,
            distance_matrix=distance_matrix,
            max_trials=max_trials,
            seed=seed,
        )

    if action == 4:
        return apply_random_swap_best_of_k(
            tour=old_tour,
            distance_matrix=distance_matrix,
            max_trials=max_trials,
            seed=seed,
        )

    if action == 5:
        return apply_perturb_then_2opt(
            tour=old_tour,
            distance_matrix=distance_matrix,
            perturb_swaps=perturb_swaps,
            two_opt_iterations=two_opt_iterations,
            seed=seed,
        )

    raise ValueError(f"Unknown action: {action}")


ACTION_NAMES = {
    0: "stop",
    1: "2opt_limited",
    2: "random_2opt_best_of_k",
    3: "relocate_best_of_k",
    4: "swap_best_of_k",
    5: "perturb_then_2opt",
}