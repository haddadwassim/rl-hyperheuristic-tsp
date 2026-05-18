from pathlib import Path
import yaml

from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor

from src.env.tsp_operator_env import TSPOperatorEnv


def load_config(config_path: str) -> dict:
    """
    Load a YAML configuration file.
    """
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def train_dqn(config: dict):
    """
    Train a DQN agent for TSP local-search operator selection.
    """
    project_config = config.get("project", {})
    agent_config = config.get("agent", {})
    logging_config = config.get("logging", {})

    output_dir = Path(project_config.get("output_dir", "results"))
    model_dir = output_dir / "models"
    log_dir = output_dir / "logs"
    tensorboard_dir = output_dir / "tb"

    model_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_dir.mkdir(parents=True, exist_ok=True)

    env = TSPOperatorEnv(config=config)
    env = Monitor(env, filename=str(log_dir / "train_monitor.csv"))

    model = DQN(
        policy=agent_config.get("policy", "MlpPolicy"),
        env=env,
        learning_rate=agent_config.get("learning_rate", 1e-4),
        buffer_size=agent_config.get("buffer_size", 100000),
        learning_starts=agent_config.get("learning_starts", 1000),
        batch_size=agent_config.get("batch_size", 64),
        gamma=agent_config.get("gamma", 0.99),
        train_freq=agent_config.get("train_freq", 4),
        gradient_steps=agent_config.get("gradient_steps", 1),
        target_update_interval=agent_config.get("target_update_interval", 1000),
        exploration_initial_eps=agent_config.get("exploration_initial_eps", 1.0),
        exploration_final_eps=agent_config.get("exploration_final_eps", 0.05),
        exploration_fraction=agent_config.get("exploration_fraction", 0.2),
        tensorboard_log=str(tensorboard_dir)
        if logging_config.get("tensorboard", True)
        else None,
        verbose=logging_config.get("verbose", 1),
        seed=project_config.get("seed", 42),
    )

    total_timesteps = agent_config.get("total_timesteps", 100000)

    model.learn(
        total_timesteps=total_timesteps,
        progress_bar=True,
    )

    model_path = model_dir / "dqn_operator_selector.zip"
    model.save(str(model_path))

    print(f"Saved model to: {model_path}")

    env.close()

    return model_path