from tsp_hh.hyper_env_v3 import HyperHeuristicStateV3
from tsp_hh.q_learning_v3 import QLearningAgent, discretize_state_v3


def test_discretize_state_v3_returns_tuple():
    state = HyperHeuristicStateV3(
        current_length=105.0,
        best_length=100.0,
        initial_length=120.0,
        last_improvement=2.0,
        steps_without_improvement=5,
        step_count=10,
        n_cities=50,
        last_category=1,
    )

    key = discretize_state_v3(
        state=state,
        max_steps=100,
    )

    assert isinstance(key, tuple)
    assert len(key) == 5
    assert all(isinstance(x, int) for x in key)


def test_q_learning_agent_can_use_v3_state_key():
    agent = QLearningAgent(n_actions=3, seed=42)

    key = (0, 0, 0, 0, 0)
    action = agent.select_action(key, training=True)

    assert 0 <= action < 3