import time
import numpy as np

from src.tsp.instance import TSPInstance
from src.tsp.tour import nearest_neighbor_tour, random_tour, tour_length
from src.operators.two_opt import two_opt_best_improvement


def run_full_two_opt(
    instance: TSPInstance,
    initial_solution_method: str = "nearest_neighbor",
    max_passes: int = 1000,
    seed: int | None = None,
) -> dict:
    """
    Run full 2-opt until no improving move is found.

    This is a stronger local-search baseline than sampled 2-opt.
    """
    rng = np.random.default_rng(seed)

    start_time = time.perf_counter()

    if initial_solution_method == "random":
        tour = random_tour(instance, rng=rng)
    else:
        tour = nearest_neighbor_tour(instance)

    initial_length = tour_length(tour, instance)
    best_length = initial_length
    best_tour = tour.copy()

    steps = 0

    for _ in range(max_passes):
        new_tour, new_length, improved = two_opt_best_improvement(
            tour=tour,
            instance=instance,
            max_trials=None,
            rng=rng,
        )

        steps += 1

        if not improved:
            break

        tour = new_tour
        best_tour = new_tour.copy()
        best_length = new_length

    runtime = time.perf_counter() - start_time

    return {
        "method": "full_two_opt",
        "initial_length": initial_length,
        "final_length": best_length,
        "best_length": best_length,
        "relative_improvement": (initial_length - best_length) / initial_length,
        "num_steps": steps,
        "runtime_sec": runtime,
        "tour": best_tour,
    }