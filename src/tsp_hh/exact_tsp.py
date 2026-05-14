import itertools

import numpy as np

from tsp_hh.tour import tour_length, validate_tour


def brute_force_optimal_tour(
    distance_matrix: np.ndarray,
    start_city: int = 0,
) -> tuple[list[int], float]:
    """
    Exact brute-force TSP solver for very small instances.

    Fixes the start city to remove rotational symmetry.

    Suitable for n <= 10 approximately.
    """
    distance_matrix = np.asarray(distance_matrix, dtype=float)
    n_cities = distance_matrix.shape[0]

    if distance_matrix.shape != (n_cities, n_cities):
        raise ValueError("distance_matrix must be square")

    if not 0 <= start_city < n_cities:
        raise ValueError("Invalid start_city")

    remaining = [city for city in range(n_cities) if city != start_city]

    best_tour = None
    best_length = float("inf")

    for perm in itertools.permutations(remaining):
        tour = [start_city, *perm]

        length = tour_length(np.asarray(tour), distance_matrix)

        if length < best_length:
            best_length = length
            best_tour = tour

    return best_tour, float(best_length)


def held_karp_optimal_tour(
    distance_matrix: np.ndarray,
    start_city: int = 0,
) -> tuple[list[int], float]:
    """
    Exact Held-Karp dynamic programming solver.

    More scalable than brute force for small instances.
    Suitable for around n <= 14 or n <= 16 depending on machine.

    Returns a tour path without repeating the start city at the end.
    """
    distance_matrix = np.asarray(distance_matrix, dtype=float)
    n = distance_matrix.shape[0]

    if distance_matrix.shape != (n, n):
        raise ValueError("distance_matrix must be square")

    if not 0 <= start_city < n:
        raise ValueError("Invalid start_city")

    cities = [c for c in range(n) if c != start_city]

    # dp[(subset_mask, last_city)] = (cost, previous_city)
    # subset_mask indexes cities list positions, not raw city ids.
    dp = {}

    for i, city in enumerate(cities):
        mask = 1 << i
        dp[(mask, city)] = (
            distance_matrix[start_city, city],
            start_city,
        )

    for subset_size in range(2, len(cities) + 1):
        for subset_indices in itertools.combinations(range(len(cities)), subset_size):
            mask = 0
            for idx in subset_indices:
                mask |= 1 << idx

            for last_idx in subset_indices:
                last_city = cities[last_idx]
                prev_mask = mask & ~(1 << last_idx)

                best_cost = float("inf")
                best_prev = None

                for prev_idx in subset_indices:
                    if prev_idx == last_idx:
                        continue

                    prev_city = cities[prev_idx]

                    if (prev_mask, prev_city) not in dp:
                        continue

                    prev_cost, _ = dp[(prev_mask, prev_city)]
                    cost = prev_cost + distance_matrix[prev_city, last_city]

                    if cost < best_cost:
                        best_cost = cost
                        best_prev = prev_city

                dp[(mask, last_city)] = (best_cost, best_prev)

    full_mask = (1 << len(cities)) - 1

    best_total = float("inf")
    best_last = None

    for last_city in cities:
        cost, _ = dp[(full_mask, last_city)]
        total = cost + distance_matrix[last_city, start_city]

        if total < best_total:
            best_total = total
            best_last = last_city

    # Reconstruct path.
    tour_reversed = []
    mask = full_mask
    last_city = best_last

    while last_city != start_city:
        tour_reversed.append(last_city)

        cost, prev_city = dp[(mask, last_city)]

        if prev_city == start_city:
            break

        last_idx = cities.index(last_city)
        mask = mask & ~(1 << last_idx)
        last_city = prev_city

    tour = [start_city] + list(reversed(tour_reversed))

    if not validate_tour(np.asarray(tour), n):
        raise RuntimeError("Held-Karp reconstructed invalid tour")

    return tour, float(best_total)