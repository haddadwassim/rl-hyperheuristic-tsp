import time
from collections import Counter

import numpy as np

from src.tsp.instance import TSPInstance
from src.tsp.tour import nearest_neighbor_tour, random_tour, tour_length
from src.operators.registry import apply_operator, ACTION_NAMES


def run_fixed_schedule(
    instance: TSPInstance,
    operator_config: dict,
    schedule: list[int] | None = None,
    max_steps: int = 100,
    initial_solution_method: str = "nearest_neighbor",
    seed: int | None = None,
    stop_on_no_improvement_round: bool = True,
) -> dict:
    """
    Apply operators according to a fixed repeating schedule.

    Default schedule:
        2-opt -> swap -> relocate
    """
    rng = np.random.default_rng(seed)

    if schedule is None:
        schedule = [1, 2, 3]

    start_time = time.perf_counter()

    if initial_solution_method == "random":
        tour = random_tour(instance, rng=rng)
    else:
        tour = nearest_neighbor_tour(instance)

    initial_length = tour_length(tour, instance)
    best_tour = tour.copy()
    best_length = initial_length

    actions = []
    no_improvement_in_round = 0

    for step in range(max_steps):
        action = schedule[step % len(schedule)]
        actions.append(action)

        old_best = best_length

        new_tour, new_length, improved, _ = apply_operator(
            action=action,
            tour=tour,
            instance=instance,
            config=operator_config,
            rng=rng,
        )

        tour = new_tour

        if new_length < best_length:
            best_length = new_length
            best_tour = new_tour.copy()

        if best_length < old_best:
            no_improvement_in_round = 0
        else:
            no_improvement_in_round += 1

        if stop_on_no_improvement_round and no_improvement_in_round >= len(schedule):
            break

    runtime = time.perf_counter() - start_time
    action_counter = Counter(actions)

    result = {
        "method": "fixed_schedule",
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