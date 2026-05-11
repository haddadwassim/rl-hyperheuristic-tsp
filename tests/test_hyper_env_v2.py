import numpy as np

from tsp_hh.instances import generate_euclidean_instance
from tsp_hh.tour import validate_tour
from tsp_hh.hyper_env_v2 import TSPHyperHeuristicEnvV2, HyperHeuristicStateV2


def test_env_v2_reset_returns_valid_state():
    instance = generate_euclidean_instance(n_cities=30, seed=42)

    env = TSPHyperHeuristicEnvV2(
        distance_matrix=instance.distance_matrix,
        max_steps=10,
        seed=123,
    )

    state = env.reset()

    assert isinstance(state, HyperHeuristicStateV2)
    assert validate_tour(env.current_tour, n_cities=30)
    assert validate_tour(env.best_tour, n_cities=30)
    assert state.step_count == 0


def test_env_v2_all_actions_keep_valid_tours():
    instance = generate_euclidean_instance(n_cities=30, seed=42)

    env = TSPHyperHeuristicEnvV2(
        distance_matrix=instance.distance_matrix,
        max_steps=10,
        seed=123,
        k_neighbors=5,
    )

    for action in range(6):
        state, reward, done, info = env.step(action)

        assert validate_tour(env.current_tour, n_cities=30)
        assert validate_tour(env.best_tour, n_cities=30)
        assert isinstance(state, HyperHeuristicStateV2)
        assert isinstance(reward, float)
        assert "action_name" in info


def test_env_v2_done_after_max_steps():
    instance = generate_euclidean_instance(n_cities=30, seed=42)

    env = TSPHyperHeuristicEnvV2(
        distance_matrix=instance.distance_matrix,
        max_steps=3,
        seed=123,
    )

    done = False

    for _ in range(3):
        _, _, done, _ = env.step(0)

    assert done


def test_env_v2_observation_shape_and_type():
    instance = generate_euclidean_instance(n_cities=30, seed=42)

    env = TSPHyperHeuristicEnvV2(
        distance_matrix=instance.distance_matrix,
        max_steps=10,
        seed=123,
    )

    obs = env.get_observation()

    assert isinstance(obs, np.ndarray)
    assert obs.shape == (6,)
    assert obs.dtype == np.float32


def test_env_v2_reward_is_normalized():
    instance = generate_euclidean_instance(n_cities=30, seed=42)

    env = TSPHyperHeuristicEnvV2(
        distance_matrix=instance.distance_matrix,
        max_steps=10,
        seed=123,
        reward_scale="initial_length",
    )

    _, reward, _, info = env.step(0)

    assert reward == info["raw_improvement"] / env.initial_length