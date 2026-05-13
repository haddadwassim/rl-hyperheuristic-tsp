import numpy as np

from tsp_hh.path_env import TSPPathEnv


def make_toy_env():
    coords = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
            [0.5, 0.5],
        ]
    )

    optimal_tour = [0, 1, 2, 3, 4]

    return TSPPathEnv(
        coords=coords,
        optimal_tour=optimal_tour,
        start_city=0,
        teacher_bonus=1.0,
        final_optimal_bonus=10.0,
    )


def test_path_env_reset():
    env = make_toy_env()

    obs, info = env.reset(seed=42)

    assert obs["current_city"][0] == 0
    assert obs["visited"][0] == 1
    assert obs["step_count"][0] == 0
    assert obs["target_next_city"][0] == 1
    assert info["path"] == [0]
    assert obs["current_city_onehot"][0] == 1.0
    assert obs["target_next_city_onehot"][1] == 1.0


def test_action_mask_excludes_visited_city():
    env = make_toy_env()
    env.reset(seed=42)

    mask = env.action_masks()

    assert mask[0] is False or mask[0] == False
    assert mask[1] is True or mask[1] == True


def test_follow_teacher_path_terminates_optimal():
    env = make_toy_env()
    env.reset(seed=42)

    done = False

    for action in [1, 2, 3, 4]:
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

    assert done
    assert info["path"] == [0, 1, 2, 3, 4]
    assert info["is_optimal_path"] is True


def test_invalid_action_terminates_with_penalty():
    env = make_toy_env()
    env.reset(seed=42)

    obs, reward, terminated, truncated, info = env.step(0)

    assert terminated
    assert reward < 0