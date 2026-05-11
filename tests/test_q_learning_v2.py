from tsp_hh.hyper_env_v2 import HyperHeuristicStateV2
from tsp_hh.q_learning_v2 import discretize_state_v2, QLearningAgent


def test_discretize_state_v2_returns_tuple():
    state = HyperHeuristicStateV2(
        current_length=105.0,
        best_length=100.0,
        last_improvement=2.0,
        steps_without_improvement=5,
        step_count=10,
        n_cities=50,
    )

    key = discretize_state_v2(
        state=state,
        initial_length=120.0,
        max_steps=50,
    )

    assert isinstance(key, tuple)
    assert len(key) == 4
    assert all(isinstance(x, int) for x in key)


def test_q_learning_agent_can_use_v2_state_key():
    agent = QLearningAgent(n_actions=6, seed=42)

    key = (0, 0, 0, 0)
    action = agent.select_action(key, training=True)

    assert 0 <= action < 6