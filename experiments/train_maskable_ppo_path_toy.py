from pathlib import Path

import numpy as np

from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.evaluation import evaluate_policy
from sb3_contrib.common.maskable.utils import get_action_masks
from stable_baselines3.common.monitor import Monitor

from tsp_hh.path_env import TSPPathEnv


def make_env():
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

    env = TSPPathEnv(
        coords=coords,
        optimal_tour=optimal_tour,
        start_city=0,
        teacher_bonus=1.0,
        final_optimal_bonus=10.0,
    )

    return Monitor(env)


def run_one_episode(model, env, deterministic: bool = True):
    obs, info = env.reset()

    done = False
    total_reward = 0.0

    while not done:
        action_masks = get_action_masks(env)

        action, _ = model.predict(
            obs,
            deterministic=deterministic,
            action_masks=action_masks,
        )

        obs, reward, terminated, truncated, info = env.step(action)

        total_reward += reward
        done = terminated or truncated

    return info, total_reward


def main():
    out_dir = Path("results/maskable_ppo_path_toy")
    out_dir.mkdir(parents=True, exist_ok=True)

    env = make_env()

    model = MaskablePPO(
        policy="MultiInputPolicy",
        env=env,
        learning_rate=3e-4,
        n_steps=64,
        batch_size=64,
        gamma=0.99,
        ent_coef=0.01,
        verbose=1,
        seed=42,
    )

    model.learn(total_timesteps=10_000)

    model_path = out_dir / "maskable_ppo_path_toy"
    model.save(model_path)

    print(f"Saved model to: {model_path}")

    mean_reward, std_reward = evaluate_policy(
        model,
        env,
        n_eval_episodes=20,
        deterministic=True,
        warn=False,
    )

    print(f"Mean reward: {mean_reward:.3f} +/- {std_reward:.3f}")

    for i in range(5):
        info, total_reward = run_one_episode(model, env, deterministic=True)

        print(
            f"Episode {i + 1}: "
            f"path={info['path']}, "
            f"distance={info['total_distance']:.3f}, "
            f"optimal={info['is_optimal_path']}, "
            f"reward={total_reward:.3f}"
        )


if __name__ == "__main__":
    main()