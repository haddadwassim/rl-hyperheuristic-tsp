import numpy as np

from tsp_hh.tour import tour_length, validate_tour, random_tour


def nearest_neighbor_tour(
    distance_matrix: np.ndarray,
    start_city: int = 0,
) -> np.ndarray:
    """
    Construct a TSP tour using the nearest-neighbor heuristic.

    Starting from start_city, repeatedly visit the nearest unvisited city.
    """
    n_cities = distance_matrix.shape[0]

    if distance_matrix.shape != (n_cities, n_cities):
        raise ValueError("distance_matrix must be square")

    if not 0 <= start_city < n_cities:
        raise ValueError("start_city must be a valid city index")

    unvisited = set(range(n_cities))
    tour = [start_city]
    unvisited.remove(start_city)

    current_city = start_city

    while unvisited:
        next_city = min(
            unvisited,
            key=lambda city: distance_matrix[current_city, city],
        )
        tour.append(next_city)
        unvisited.remove(next_city)
        current_city = next_city

    return np.array(tour, dtype=int)


def two_opt_move(
    tour: np.ndarray,
    i: int,
    k: int,
) -> np.ndarray:
    """
    Apply one 2-opt move by reversing the segment tour[i:k+1].
    """
    new_tour = tour.copy()
    new_tour[i:k + 1] = new_tour[i:k + 1][::-1]
    return new_tour


def best_improvement_two_opt(
    tour: np.ndarray,
    distance_matrix: np.ndarray,
) -> tuple[np.ndarray, float]:
    """
    Apply the best improving 2-opt move once.

    Returns
    -------
    new_tour:
        The improved tour if an improving move exists.
        Otherwise, the original tour.
    improvement:
        Positive value equal to old_length - new_length.
        Returns 0.0 if no improvement is found.
    """
    tour = np.asarray(tour, dtype=int)
    n_cities = len(tour)

    if not validate_tour(tour, n_cities):
        raise ValueError("Invalid tour")

    current_length = tour_length(tour, distance_matrix)

    best_tour = tour.copy()
    best_length = current_length

    for i in range(1, n_cities - 1):
        for k in range(i + 1, n_cities):
            candidate = two_opt_move(tour, i, k)
            candidate_length = tour_length(candidate, distance_matrix)

            if candidate_length < best_length:
                best_length = candidate_length
                best_tour = candidate

    improvement = current_length - best_length
    return best_tour, float(improvement)


def first_improvement_two_opt(
    tour: np.ndarray,
    distance_matrix: np.ndarray,
) -> tuple[np.ndarray, float]:
    """
    Apply the first improving 2-opt move found.

    This is usually faster than best-improvement 2-opt.
    """
    tour = np.asarray(tour, dtype=int)
    n_cities = len(tour)

    if not validate_tour(tour, n_cities):
        raise ValueError("Invalid tour")

    current_length = tour_length(tour, distance_matrix)

    for i in range(1, n_cities - 1):
        for k in range(i + 1, n_cities):
            candidate = two_opt_move(tour, i, k)
            candidate_length = tour_length(candidate, distance_matrix)

            if candidate_length < current_length:
                improvement = current_length - candidate_length
                return candidate, float(improvement)

    return tour.copy(), 0.0


def random_swap_move(
    tour: np.ndarray,
    seed: int | None = None,
) -> np.ndarray:
    """
    Apply a random swap move to a tour.

    Two cities are selected randomly and exchanged.
    """
    tour = np.asarray(tour, dtype=int)
    n_cities = len(tour)

    if n_cities < 3:
        raise ValueError("tour must contain at least 3 cities")

    rng = np.random.default_rng(seed)
    i, j = rng.choice(n_cities, size=2, replace=False)

    new_tour = tour.copy()
    new_tour[i], new_tour[j] = new_tour[j], new_tour[i]

    return new_tour


def random_insertion_move(
    tour: np.ndarray,
    seed: int | None = None,
) -> np.ndarray:
    """
    Apply a random insertion move.

    One city is removed from its current position and inserted elsewhere.
    """
    tour = np.asarray(tour, dtype=int)
    n_cities = len(tour)

    if n_cities < 3:
        raise ValueError("tour must contain at least 3 cities")

    rng = np.random.default_rng(seed)
    i, j = rng.choice(n_cities, size=2, replace=False)

    new_tour_list = tour.tolist()
    city = new_tour_list.pop(i)
    new_tour_list.insert(j, city)

    return np.array(new_tour_list, dtype=int)


def perturb_tour(
    tour: np.ndarray,
    n_moves: int = 3,
    seed: int | None = None,
) -> np.ndarray:
    """
    Apply several random swap moves.

    This is useful to escape local minima.
    """
    if n_moves < 1:
        raise ValueError("n_moves must be at least 1")

    rng = np.random.default_rng(seed)
    new_tour = np.asarray(tour, dtype=int).copy()

    for _ in range(n_moves):
        move_seed = int(rng.integers(0, 1_000_000_000))
        new_tour = random_swap_move(new_tour, seed=move_seed)

    return new_tour


def repeated_two_opt(
    tour: np.ndarray,
    distance_matrix: np.ndarray,
    max_iterations: int = 100,
    use_best_improvement: bool = False,
) -> tuple[np.ndarray, float, int]:
    """
    Repeatedly apply 2-opt until no improvement is found or max_iterations is reached.

    Returns
    -------
    final_tour:
        Final improved tour.
    total_improvement:
        Total decrease in tour length.
    n_iterations:
        Number of successful improving moves.
    """
    if max_iterations < 1:
        raise ValueError("max_iterations must be at least 1")

    current_tour = np.asarray(tour, dtype=int).copy()
    initial_length = tour_length(current_tour, distance_matrix)

    improvement_function = (
        best_improvement_two_opt
        if use_best_improvement
        else first_improvement_two_opt
    )

    n_iterations = 0

    for _ in range(max_iterations):
        new_tour, improvement = improvement_function(current_tour, distance_matrix)

        if improvement <= 0:
            break

        current_tour = new_tour
        n_iterations += 1

    final_length = tour_length(current_tour, distance_matrix)
    total_improvement = initial_length - final_length

    return current_tour, float(total_improvement), n_iterations


def create_initial_tour(
    distance_matrix: np.ndarray,
    method: str = "nearest_neighbor",
    seed: int | None = None,
) -> np.ndarray:
    """
    Create an initial TSP tour.

    Supported methods:
    - "nearest_neighbor"
    - "random"
    """
    n_cities = distance_matrix.shape[0]

    if method == "nearest_neighbor":
        return nearest_neighbor_tour(distance_matrix, start_city=0)

    if method == "random":
        return random_tour(n_cities, seed=seed)

    raise ValueError(f"Unknown initial tour method: {method}")