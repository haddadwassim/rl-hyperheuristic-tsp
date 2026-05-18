import time
import numpy as np

from src.tsp.instance import TSPInstance
from src.tsp.tour import nearest_neighbor_tour, random_tour, tour_length
from src.operators.two_opt import two_opt_best_improvement


def _two_opt_until_convergence(
    tour,
    instance: TSPInstance,
    max_passes: int,
    rng: np.random.Generator,
):
    current_length = tour_length(tour, instance)
    best_tour = tour.copy()
    best_length = current_length
    steps = 0

    for _ in range(max_passes):
        new_tour, new_length, improved = two_opt_best_improvement(
            tour=best_tour,
            instance=instance,
            max_trials=None,
            rng=rng,
        )

        steps += 1

        if not improved:
            break

        best_tour = new_tour
        best_length = new_length

    return best_tour, best_length, steps


def run_multistart_two_opt(
    instance: TSPInstance,
    num_starts: int = 10,
    max_passes_per_start: int = 1000,
    include_nearest_neighbor: bool = True,
    seed: int | None = None,
) -> dict:
    """
    Multi-start full 2-opt.

    This is a much stronger baseline because it uses several initial tours.
    """
    rng = np.random.default_rng(seed)

    start_time = time.perf_counter()

    global_best_tour = None
    global_best_length = float("inf")
    total_steps = 0

    initial_lengths = []

    starts_done = 0

    if include_nearest_neighbor:
        initial_tour = nearest_neighbor_tour(instance)
        initial_lengths.append(tour_length(initial_tour, instance))

        tour, length, steps = _two_opt_until_convergence(
            initial_tour,
            instance,
            max_passes=max_passes_per_start,
            rng=rng,
        )

        total_steps += steps
        starts_done += 1

        if length < global_best_length:
            global_best_length = length
            global_best_tour = tour.copy()

    while starts_done < num_starts:
        initial_tour = random_tour(instance, rng=rng)
        initial_lengths.append(tour_length(initial_tour, instance))

        tour, length, steps = _two_opt_until_convergence(
            initial_tour,
            instance,
            max_passes=max_passes_per_start,
            rng=rng,
        )

        total_steps += steps
        starts_done += 1

        if length < global_best_length:
            global_best_length = length
            global_best_tour = tour.copy()

    runtime = time.perf_counter() - start_time

    reference_initial = initial_lengths[0]

    return {
        "method": f"multistart_two_opt_{num_starts}",
        "initial_length": reference_initial,
        "final_length": global_best_length,
        "best_length": global_best_length,
        "relative_improvement": (reference_initial - global_best_length) / reference_initial,
        "num_steps": total_steps,
        "runtime_sec": runtime,
        "tour": global_best_tour,
    }