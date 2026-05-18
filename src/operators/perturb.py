import numpy as np

from src.tsp.instance import TSPInstance
from src.tsp.tour import tour_length
from src.operators.two_opt import two_opt_best_improvement


def double_bridge_perturbation(
    tour: np.ndarray,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Apply a double-bridge perturbation.

    This is a classical TSP perturbation used to escape local minima.
    """
    if rng is None:
        rng = np.random.default_rng()

    n = len(tour)

    if n < 8:
        return rng.permutation(tour).astype(np.int64)

    cuts = sorted(rng.choice(np.arange(1, n), size=4, replace=False))
    a, b, c, d = cuts

    part1 = tour[:a]
    part2 = tour[a:b]
    part3 = tour[b:c]
    part4 = tour[c:d]
    part5 = tour[d:]

    new_tour = np.concatenate([part1, part3, part2, part4, part5])

    return new_tour.astype(np.int64)


def perturb_then_two_opt(
    tour: np.ndarray,
    instance: TSPInstance,
    perturb_strength: int = 1,
    two_opt_trials: int = 100,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, float, bool]:
    """
    Apply perturbation followed by one 2-opt improvement step.

    This action is useful for escaping local minima.
    """
    if rng is None:
        rng = np.random.default_rng()

    current_length = tour_length(tour, instance)

    candidate = tour.copy()

    for _ in range(perturb_strength):
        candidate = double_bridge_perturbation(candidate, rng=rng)

    candidate, candidate_length, _ = two_opt_best_improvement(
        candidate,
        instance,
        max_trials=two_opt_trials,
        rng=rng,
    )

    improved = candidate_length < current_length

    # Important design choice:
    # For the DRL environment, we allow non-improving perturbations because
    # they may help escape local minima. The reward will decide if this is useful.
    return candidate, candidate_length, improved