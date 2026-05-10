import numpy as np

from tsp_hh.instances import generate_euclidean_instance
from tsp_hh.tour import validate_tour
from tsp_hh.hyper_env import TSPHyperHeuristicEnv, HyperHeuristicState


def test_env_reset_returns_valid_state():
    instance = generate_euclidean_instance(n_cities=20, seed=42)

    env = TSPHyperHeuristicEnv(
        distance_matrix=instance.distance_matrix,
        max_steps=10,
        seed=123,
    )

    state = env.reset()

    assert isinstance(state, HyperHeuristicState)
    assert validate_tour(env.current_tour, n_cities=20)
    assert validate_tour(env.best_tour, n_cities=20)
    assert state.step_count == 0
    assert state.steps_without_improvement == 0


def test_env_step_returns_valid_transition():
    instance = generate_euclidean_instance(n_cities=20, seed=42)

    env = TSPHyperHeuristicEnv(
        distance_matrix=instance.distance_matrix,
        max_steps=10,
        seed=123,
    )

    state, reward, done, info = env.step(0)

    assert isinstance(state, HyperHeuristicState)
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    assert isinstance(info, dict)
    assert validate_tour(env.current_tour, n_cities=20)
    assert validate_tour(env.best_tour, n_cities=20)
    assert state.step_count == 1
    assert "action_name" in info


def test_env_done_after_max_steps():
    instance = generate_euclidean_instance(n_cities=20, seed=42)

    env = TSPHyperHeuristicEnv(
        distance_matrix=instance.distance_matrix,
        max_steps=3,
        seed=123,
    )

    done = False

    for _ in range(3):
        _, _, done, _ = env.step(0)

    assert done


def test_all_actions_keep_valid_tours():
    instance = generate_euclidean_instance(n_cities=30, seed=42)

    env = TSPHyperHeuristicEnv(
        distance_matrix=instance.distance_matrix,
        max_steps=10,
        seed=123,
    )

    for action in range(5):
        state, reward, done, info = env.step(action)

        assert validate_tour(env.current_tour, n_cities=30)
        assert validate_tour(env.best_tour, n_cities=30)
        assert isinstance(reward, float)
        assert info["action_name"] in env.ACTION_NAMES.values()


def test_observation_shape_and_type():
    instance = generate_euclidean_instance(n_cities=20, seed=42)

    env = TSPHyperHeuristicEnv(
        distance_matrix=instance.distance_matrix,
        max_steps=10,
        seed=123,
    )

    obs = env.get_observation()

    assert isinstance(obs, np.ndarray)
    assert obs.shape == (5,)
    assert obs.dtype == np.float32


def test_reward_is_non_negative_when_best_is_tracked():
    instance = generate_euclidean_instance(n_cities=20, seed=42)

    env = TSPHyperHeuristicEnv(
        distance_matrix=instance.distance_matrix,
        max_steps=10,
        seed=123,
    )

    for action in [0, 1, 2, 3, 4]:
        _, reward, _, _ = env.step(action)
        assert reward >= 0.0