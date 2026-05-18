import numpy as np

from src.tsp.instance import TSPInstance
from src.operators.two_opt import two_opt_best_improvement, two_opt_first_improvement
from src.operators.swap import swap_best_improvement
from src.operators.relocate import relocate_best_improvement
from src.operators.perturb import perturb_then_two_opt
from src.operators.full_two_opt import (
    full_two_opt,
    perturb_then_full_two_opt,
    multi_perturb_then_full_two_opt,
)


STOP_ACTION = 0

ACTION_NAMES = {
    0: "stop",
    1: "sampled_two_opt",
    2: "swap",
    3: "relocate",
    4: "perturb_sampled_two_opt",
    5: "full_two_opt",
    6: "multi_perturb_full_two_opt",
}


def num_actions() -> int:
    return len(ACTION_NAMES)


def action_name(action: int) -> str:
    if action not in ACTION_NAMES:
        raise ValueError(f"Unknown action: {action}")

    return ACTION_NAMES[action]


def apply_operator(
    action: int,
    tour: np.ndarray,
    instance: TSPInstance,
    config: dict | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, float, bool, dict]:
    """
    Apply one operator action to a TSP tour.

    Returns
    -------
    new_tour:
        Tour after applying the selected operator.

    new_length:
        Length of the new tour.

    improved:
        Whether the action improved the tour compared to its previous length.

    info:
        Dictionary containing metadata about the applied action.
    """
    if rng is None:
        rng = np.random.default_rng()

    if config is None:
        config = {}

    if action == 0:
        from src.tsp.tour import tour_length

        length = tour_length(tour, instance)
        return tour.copy(), length, False, {"operator": "stop"}

    if action == 1:
        op_config = config.get("two_opt", {})
        mode = op_config.get("mode", "best_improvement")
        max_trials = op_config.get("max_trials", 200)

        if mode == "first_improvement":
            new_tour, new_length, improved = two_opt_first_improvement(
                tour,
                instance,
                max_trials=max_trials,
                rng=rng,
            )
        else:
            new_tour, new_length, improved = two_opt_best_improvement(
                tour,
                instance,
                max_trials=max_trials,
                rng=rng,
            )

        return new_tour, new_length, improved, {"operator": "two_opt"}

    if action == 2:
        op_config = config.get("swap", {})
        max_trials = op_config.get("max_trials", 200)

        new_tour, new_length, improved = swap_best_improvement(
            tour,
            instance,
            max_trials=max_trials,
            rng=rng,
        )

        return new_tour, new_length, improved, {"operator": "swap"}

    if action == 3:
        op_config = config.get("relocate", {})
        max_trials = op_config.get("max_trials", 200)

        new_tour, new_length, improved = relocate_best_improvement(
            tour,
            instance,
            max_trials=max_trials,
            rng=rng,
        )

        return new_tour, new_length, improved, {"operator": "relocate"}

    if action == 4:
        op_config = config.get("perturb_two_opt", {})
        perturb_strength = op_config.get("perturb_strength", 1)
        two_opt_trials = op_config.get("two_opt_trials", 100)

        new_tour, new_length, improved = perturb_then_two_opt(
            tour,
            instance,
            perturb_strength=perturb_strength,
            two_opt_trials=two_opt_trials,
            rng=rng,
        )

        return new_tour, new_length, improved, {"operator": "perturb_two_opt"}
    
    if action == 5:
        op_config = config.get("full_two_opt", {})
        max_passes = op_config.get("max_passes", 1000)

        new_tour, new_length, improved = full_two_opt(
            tour,
            instance,
            max_passes=max_passes,
            rng=rng,
        )

        return new_tour, new_length, improved, {"operator": "full_two_opt"}

    if action == 6:
        op_config = config.get("multi_perturb_full_two_opt", {})
        perturb_strength = op_config.get("perturb_strength", 1)
        num_trials = op_config.get("num_trials", 3)
        max_passes = op_config.get("max_passes", 1000)

        new_tour, new_length, improved = multi_perturb_then_full_two_opt(
            tour,
            instance,
            perturb_strength=perturb_strength,
            num_trials=num_trials,
            max_passes=max_passes,
            rng=rng,
        )

        return new_tour, new_length, improved, {
            "operator": "multi_perturb_full_two_opt"
        }

    raise ValueError(f"Unknown action: {action}")