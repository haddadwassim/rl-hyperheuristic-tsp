from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
from stable_baselines3 import DQN

from src.env.tsp_operator_env import TSPOperatorEnv
from src.operators.registry import ACTION_NAMES


def evaluate_model(
    model_path: str,
    config: dict,
    num_episodes: int = 100,
    deterministic: bool = True,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Evaluate a trained DQN model on multiple TSP instances.

    Returns
    -------
    pd.DataFrame
        One row per evaluation episode.
    """
    model = DQN.load(model_path)

    env = TSPOperatorEnv(config=config)

    results = []

    for episode in range(num_episodes):
        obs, info = env.reset(seed=seed + episode)

        terminated = False
        truncated = False

        total_reward = 0.0
        actions = []

        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=deterministic)
            action = int(action)

            obs, reward, terminated, truncated, info = env.step(action)

            total_reward += reward
            actions.append(action)

        action_counter = Counter(actions)

        row = {
            "episode": episode + 1,
            "initial_length": info["initial_length"],
            "final_length": info["current_length"],
            "best_length": info["best_length"],
            "relative_improvement": info["relative_improvement"],
            "total_reward": total_reward,
            "num_steps": info["current_step"],
            "terminated": terminated,
            "truncated": truncated,
        }

        for action_id, action_name in ACTION_NAMES.items():
            row[f"count_{action_name}"] = action_counter.get(action_id, 0)

        results.append(row)

    env.close()

    return pd.DataFrame(results)


def summarize_results(df: pd.DataFrame) -> dict:
    """
    Compute aggregate evaluation statistics.
    """
    summary = {
        "episodes": len(df),
        "mean_initial_length": df["initial_length"].mean(),
        "mean_final_length": df["final_length"].mean(),
        "mean_best_length": df["best_length"].mean(),
        "mean_relative_improvement": df["relative_improvement"].mean(),
        "std_relative_improvement": df["relative_improvement"].std(),
        "mean_total_reward": df["total_reward"].mean(),
        "mean_num_steps": df["num_steps"].mean(),
        "stop_rate": (df["count_stop"] > 0).mean(),
        "truncation_rate": df["truncated"].mean(),
    }

    for action_name in ACTION_NAMES.values():
        col = f"count_{action_name}"
        if col in df.columns:
            summary[f"mean_{col}"] = df[col].mean()

    return summary


def save_evaluation_results(
    df: pd.DataFrame,
    output_dir: str = "results",
    filename: str = "dqn_evaluation.csv",
) -> Path:
    """
    Save evaluation results to CSV.
    """
    output_path = Path(output_dir) / "logs" / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_path, index=False)

    return output_path