import numpy as np

from src.tsp.instance import TSPInstance
from src.tsp.tour import tour_length
from src.operators.two_opt import two_opt_best_improvement
from src.operators.perturb import double_bridge_perturbation


def full_two_opt(
    tour: np.ndarray,
    instance: TSPInstance,
    max_passes: int = 1000,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, float, bool]:
    """
    Run full 2-opt until no improving move is found.

    This is a macro-action used for intensification.
    """
    if rng is None:
        rng = np.random.default_rng()

    current_tour = tour.copy()
    initial_length = tour_length(current_tour, instance)
    current_length = initial_length

    improved_any = False

    for _ in range(max_passes):
        new_tour, new_length, improved = two_opt_best_improvement(
            tour=current_tour,
            instance=instance,
            max_trials=None,
            rng=rng,
        )

        if not improved:
            break

        current_tour = new_tour
        current_length = new_length
        improved_any = True

    return current_tour, current_length, improved_any


def perturb_then_full_two_opt(
    tour: np.ndarray,
    instance: TSPInstance,
    perturb_strength: int = 1,
    max_passes: int = 1000,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, float, bool]:
    """
    Apply perturbation followed by full 2-opt.

    This is a diversification + intensification macro-action.
    """
    if rng is None:
        rng = np.random.default_rng()

    initial_length = tour_length(tour, instance)

    candidate = tour.copy()

    for _ in range(perturb_strength):
        candidate = double_bridge_perturbation(candidate, rng=rng)

    candidate, candidate_length, _ = full_two_opt(
        tour=candidate,
        instance=instance,
        max_passes=max_passes,
        rng=rng,
    )

    improved = candidate_length < initial_length

    return candidate, candidate_length, improved

def multi_perturb_then_full_two_opt(
    tour: np.ndarray,
    instance: TSPInstance,
    perturb_strength: int = 1,
    num_trials: int = 3,
    max_passes: int = 1000,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, float, bool]:
    """
    Apply several perturbation + full 2-opt trials and return the best result.

    This is a stronger diversification macro-action than a single perturbation.
    """
    if rng is None:
        rng = np.random.default_rng()

    initial_length = tour_length(tour, instance)

    best_tour = tour.copy()
    best_length = initial_length

    for _ in range(num_trials):
        candidate = tour.copy()

        for _ in range(perturb_strength):
            candidate = double_bridge_perturbation(candidate, rng=rng)

        candidate, candidate_length, _ = full_two_opt(
            tour=candidate,
            instance=instance,
            max_passes=max_passes,
            rng=rng,
        )

        if candidate_length < best_length:
            best_tour = candidate.copy()
            best_length = candidate_length

    improved = best_length < initial_length

    return best_tour, best_length, improved