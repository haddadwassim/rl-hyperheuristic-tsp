from pathlib import Path
from collections import Counter
import time

import numpy as np
import pandas as pd
from stable_baselines3 import DQN

from src.tsp.generators import generate_instance
from src.tsp.tour import nearest_neighbor_tour, random_tour, tour_length
from src.operators.registry import ACTION_NAMES
from src.env.tsp_operator_env import TSPOperatorEnv
from src.baselines.nearest_neighbor import (
    run_nearest_neighbor,
    run_nearest_neighbor_two_opt,
)
from src.baselines.random_operator import run_random_operator
from src.baselines.fixed_schedule import run_fixed_schedule


def run_dqn_on_instance(
    model: DQN,
    instance,
    config: dict,
    max_steps: int = 100,
    deterministic: bool = True,
    seed: int | None = None,
) -> dict:
    """
    Run the trained DQN policy on a fixed TSP instance.

    We manually inject the instance into the environment to ensure that DQN and
    baselines are evaluated on the exact same problem.
    """
    rng = np.random.default_rng(seed)

    env = TSPOperatorEnv(config=config)
    obs, info = env.reset(seed=seed)

    env.instance = instance

    initial_method = config.get("initial_solution", {}).get("method", "nearest_neighbor")

    if initial_method == "random":
        env.current_tour = random_tour(instance, rng=rng)
    else:
        env.current_tour = nearest_neighbor_tour(instance)

    env.current_length = tour_length(env.current_tour, instance)
    env.initial_length = env.current_length
    env.best_tour = env.current_tour.copy()
    env.best_length = env.current_length
    env.current_step = 0
    env.steps_since_improvement = 0
    env.last_improvement = 0.0
    env.last_action = 0
    env.max_steps = max_steps

    obs = env._get_observation()

    start_time = time.perf_counter()

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

    runtime = time.perf_counter() - start_time
    action_counter = Counter(actions)

    result = {
        "method": "dqn_operator_selector",
        "initial_length": info["initial_length"],
        "final_length": info["current_length"],
        "best_length": info["best_length"],
        "relative_improvement": info["relative_improvement"],
        "num_steps": info["current_step"],
        "runtime_sec": runtime,
        "total_reward": total_reward,
    }

    for action_id, action_name in ACTION_NAMES.items():
        result[f"count_{action_name}"] = action_counter.get(action_id, 0)

    env.close()

    return result


def run_benchmark(
    config: dict,
    model_path: str | None = None,
    num_instances: int = 100,
    num_nodes: int | None = None,
    distribution: str | None = None,
    seed: int = 42,
    deterministic: bool = True,
) -> pd.DataFrame:
    """
    Run all methods on the same generated TSP instances.
    """
    tsp_config = config.get("tsp", {})
    env_config = config.get("environment", {})

    if num_nodes is None:
        num_nodes = tsp_config.get("eval_num_nodes", [50])[0]
        if isinstance(num_nodes, list):
            num_nodes = num_nodes[0]

    if distribution is None:
        distribution = tsp_config.get("train_distribution", "uniform")

    max_steps = env_config.get("max_steps", 100)
    operator_config = config.get("operators", {})
    initial_solution_method = config.get("initial_solution", {}).get(
        "method", "nearest_neighbor"
    )

    model = None
    if model_path is not None:
        model = DQN.load(model_path)

    rows = []

    for idx in range(num_instances):
        instance_seed = seed + idx

        instance = generate_instance(
            distribution=distribution,
            num_nodes=num_nodes,
            seed=instance_seed,
        )

        methods = [
            run_nearest_neighbor(instance),
            run_nearest_neighbor_two_opt(
                instance=instance,
                max_steps=max_steps,
                max_trials=operator_config.get("two_opt", {}).get("max_trials", 200),
                seed=instance_seed,
            ),
            run_random_operator(
                instance=instance,
                operator_config=operator_config,
                max_steps=max_steps,
                initial_solution_method=initial_solution_method,
                seed=instance_seed,
            ),
            run_fixed_schedule(
                instance=instance,
                operator_config=operator_config,
                max_steps=max_steps,
                initial_solution_method=initial_solution_method,
                seed=instance_seed,
            ),
        ]

        if model is not None:
            methods.append(
                run_dqn_on_instance(
                    model=model,
                    instance=instance,
                    config=config,
                    max_steps=max_steps,
                    deterministic=deterministic,
                    seed=instance_seed,
                )
            )

        for result in methods:
            result = result.copy()
            result.pop("tour", None)
            result["instance_id"] = idx + 1
            result["num_nodes"] = num_nodes
            result["distribution"] = distribution
            rows.append(result)

    return pd.DataFrame(rows)


def summarize_benchmark(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate benchmark results by method.
    """
    summary = (
        df.groupby("method")
        .agg(
            mean_initial_length=("initial_length", "mean"),
            mean_final_length=("final_length", "mean"),
            mean_relative_improvement=("relative_improvement", "mean"),
            std_relative_improvement=("relative_improvement", "std"),
            mean_num_steps=("num_steps", "mean"),
            mean_runtime_sec=("runtime_sec", "mean"),
        )
        .reset_index()
    )

    return summary


def save_benchmark_results(
    df: pd.DataFrame,
    output_dir: str = "results",
    filename: str = "benchmark.csv",
) -> Path:
    output_path = Path(output_dir) / "logs" / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return output_path