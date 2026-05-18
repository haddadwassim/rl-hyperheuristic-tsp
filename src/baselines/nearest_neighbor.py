import time
import numpy as np

from src.tsp.instance import TSPInstance
from src.tsp.tour import nearest_neighbor_tour, tour_length
from src.operators.two_opt import two_opt_best_improvement


def run_nearest_neighbor(instance: TSPInstance) -> dict:
    """
    Construct a tour using nearest neighbor only.
    """
    start_time = time.perf_counter()

    tour = nearest_neighbor_tour(instance)
    length = tour_length(tour, instance)

    runtime = time.perf_counter() - start_time

    return {
        "method": "nearest_neighbor",
        "initial_length": length,
        "final_length": length,
        "best_length": length,
        "relative_improvement": 0.0,
        "num_steps": 0,
        "runtime_sec": runtime,
        "tour": tour,
    }


def run_nearest_neighbor_two_opt(
    instance: TSPInstance,
    max_steps: int = 100,
    max_trials: int = 200,
    seed: int | None = None,
) -> dict:
    """
    Construct a nearest-neighbor tour and repeatedly apply 2-opt.
    """
    rng = np.random.default_rng(seed)

    start_time = time.perf_counter()

    tour = nearest_neighbor_tour(instance)
    initial_length = tour_length(tour, instance)

    current_length = initial_length
    best_length = initial_length
    best_tour = tour.copy()

    steps = 0

    for _ in range(max_steps):
        new_tour, new_length, improved = two_opt_best_improvement(
            tour=tour,
            instance=instance,
            max_trials=max_trials,
            rng=rng,
        )

        steps += 1

        if not improved:
            break

        tour = new_tour
        current_length = new_length

        if new_length < best_length:
            best_length = new_length
            best_tour = new_tour.copy()

    runtime = time.perf_counter() - start_time

    return {
        "method": "nearest_neighbor_two_opt",
        "initial_length": initial_length,
        "final_length": best_length,
        "best_length": best_length,
        "relative_improvement": (initial_length - best_length) / initial_length,
        "num_steps": steps,
        "runtime_sec": runtime,
        "tour": best_tour,
    }