import numpy as np

from tsp_hh.instances import generate_euclidean_instance
from tsp_hh.tour import validate_tour
from tsp_hh.hyper_env_v3 import TSPHyperHeuristicEnvV3, HyperHeuristicStateV3


def test_env_v3_reset_returns_valid_state():
    instance = generate_euclidean_instance(n_cities=30, seed=42)

    env = TSPHyperHeuristicEnvV3(
        distance_matrix=instance.distance_matrix,
        max_steps=10,
        seed=123,
        construction_starts=5,
    )

    state = env.reset()

    assert isinstance(state, HyperHeuristicStateV3)
    assert validate_tour(env.current_tour, n_cities=30)
    assert validate_tour(env.best_tour, n_cities=30)
    assert state.step_count == 0


def test_env_v3_all_categories_keep_valid_tours():
    instance = generate_euclidean_instance(n_cities=30, seed=42)

    env = TSPHyperHeuristicEnvV3(
        distance_matrix=instance.distance_matrix,
        max_steps=10,
        seed=123,
        construction_starts=5,
        two_opt_iterations=5,
        three_opt_samples=10,
    )

    for category in range(3):
        state, reward, done, info = env.step(category)

        assert isinstance(state, HyperHeuristicStateV3)
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert isinstance(info, dict)
        assert validate_tour(env.current_tour, n_cities=30)
        assert validate_tour(env.best_tour, n_cities=30)
        assert "category_name" in info
        assert "heuristic_name" in info


def test_env_v3_done_after_max_steps():
    instance = generate_euclidean_instance(n_cities=30, seed=42)

    env = TSPHyperHeuristicEnvV3(
        distance_matrix=instance.distance_matrix,
        max_steps=3,
        seed=123,
        construction_starts=5,
    )

    done = False

    for _ in range(3):
        _, _, done, _ = env.step(1)

    assert done


def test_env_v3_observation_shape_and_type():
    instance = generate_euclidean_instance(n_cities=30, seed=42)

    env = TSPHyperHeuristicEnvV3(
        distance_matrix=instance.distance_matrix,
        max_steps=10,
        seed=123,
        construction_starts=5,
    )

    obs = env.get_observation()

    assert isinstance(obs, np.ndarray)
    assert obs.shape == (7,)
    assert obs.dtype == np.float32


def test_env_v3_category_counts_are_updated():
    instance = generate_euclidean_instance(n_cities=30, seed=42)

    env = TSPHyperHeuristicEnvV3(
        distance_matrix=instance.distance_matrix,
        max_steps=10,
        seed=123,
        construction_starts=5,
    )

    env.step(0)
    env.step(1)
    env.step(2)

    assert env.category_counts["construction"] == 1
    assert env.category_counts["improvement"] == 1
    assert env.category_counts["perturbation"] == 1


def test_env_v3_heuristic_counts_are_updated():
    instance = generate_euclidean_instance(n_cities=30, seed=42)

    env = TSPHyperHeuristicEnvV3(
        distance_matrix=instance.distance_matrix,
        max_steps=10,
        seed=123,
        construction_starts=5,
    )

    env.step(1)

    assert len(env.heuristic_counts) >= 1