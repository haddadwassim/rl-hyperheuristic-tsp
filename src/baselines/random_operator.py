import time
from collections import Counter

import numpy as np

from src.tsp.instance import TSPInstance
from src.tsp.tour import nearest_neighbor_tour, random_tour, tour_length
from src.operators.registry import apply_operator, ACTION_NAMES


def run_random_operator(
    instance: TSPInstance,
    operator_config: dict,
    max_steps: int = 100,
    initial_solution_method: str = "nearest_neighbor",
    seed: int | None = None,
    stop_probability: float = 0.05,
) -> dict:
    """
    Randomly select operators, including STOP with a given probability.
    """
    rng = np.random.default_rng(seed)

    start_time = time.perf_counter()

    if initial_solution_method == "random":
        tour = random_tour(instance, rng=rng)
    else:
        tour = nearest_neighbor_tour(instance)

    initial_length = tour_length(tour, instance)
    current_length = initial_length

    best_tour = tour.copy()
    best_length = initial_length

    actions = []

    non_stop_actions = [a for a in ACTION_NAMES.keys() if ACTION_NAMES[a] != "stop"]

    for step in range(max_steps):
        if rng.random() < stop_probability:
            action = 0
        else:
            action = int(rng.choice(non_stop_actions))

        actions.append(action)

        if action == 0:
            break

        new_tour, new_length, improved, _ = apply_operator(
            action=action,
            tour=tour,
            instance=instance,
            config=operator_config,
            rng=rng,
        )

        tour = new_tour
        current_length = new_length

        if new_length < best_length:
            best_length = new_length
            best_tour = new_tour.copy()

    runtime = time.perf_counter() - start_time
    action_counter = Counter(actions)

    result = {
        "method": "random_operator",
        "initial_length": initial_length,
        "final_length": best_length,
        "best_length": best_length,
        "relative_improvement": (initial_length - best_length) / initial_length,
        "num_steps": len(actions),
        "runtime_sec": runtime,
        "tour": best_tour,
    }

    for action_id, action_name in ACTION_NAMES.items():
        result[f"count_{action_name}"] = action_counter.get(action_id, 0)

    return result