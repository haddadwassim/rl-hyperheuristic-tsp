import numpy as np

from tsp_hh.hyper_env import HyperHeuristicState
from tsp_hh.q_learning import discretize_state, QLearningAgent


def test_discretize_state_returns_tuple():
    state = HyperHeuristicState(
        current_length=105.0,
        best_length=100.0,
        steps_without_improvement=5,
        step_count=10,
        n_cities=20,
    )

    key = discretize_state(
        state=state,
        initial_length=120.0,
        max_steps=50,
    )

    assert isinstance(key, tuple)
    assert len(key) == 3
    assert all(isinstance(x, int) for x in key)


def test_q_values_are_initialized():
    agent = QLearningAgent(n_actions=5, seed=42)

    q_values = agent.get_q_values((0, 0, 0))

    assert isinstance(q_values, np.ndarray)
    assert q_values.shape == (5,)
    assert np.allclose(q_values, 0.0)


def test_select_action_returns_valid_action():
    agent = QLearningAgent(n_actions=5, seed=42)

    action = agent.select_action((0, 0, 0), training=True)

    assert 0 <= action < 5


def test_q_update_changes_value():
    agent = QLearningAgent(
        n_actions=5,
        learning_rate=0.5,
        discount_factor=0.9,
        epsilon=0.0,
        seed=42,
    )

    state_key = (0, 0, 0)
    next_state_key = (1, 0, 0)

    old_value = agent.get_q_values(state_key)[2]

    agent.update(
        state_key=state_key,
        action=2,
        reward=1.0,
        next_state_key=next_state_key,
        done=False,
    )

    new_value = agent.get_q_values(state_key)[2]

    assert new_value != old_value
    assert new_value > old_value


def test_epsilon_decay_respects_minimum():
    agent = QLearningAgent(
        n_actions=5,
        epsilon=1.0,
        epsilon_min=0.2,
        epsilon_decay=0.1,
        seed=42,
    )

    for _ in range(10):
        agent.decay_epsilon()

    assert agent.epsilon == 0.2