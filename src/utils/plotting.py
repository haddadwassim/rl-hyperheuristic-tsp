from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def load_monitor_csv(path: str) -> pd.DataFrame:
    """
    Load a Stable-Baselines3 Monitor CSV file.

    Monitor files usually contain one metadata line starting with '#',
    followed by columns:
        r,l,t
    where:
        r = episode reward
        l = episode length
        t = elapsed wall-clock time
    """
    return pd.read_csv(path, comment="#")


def add_training_timesteps(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add cumulative environment timesteps based on episode lengths.
    """
    df = df.copy()
    df["timesteps"] = df["l"].cumsum()
    df["episode"] = range(1, len(df) + 1)
    return df


def plot_episode_reward(
    monitor_path: str,
    output_path: str,
    window: int = 20,
    title: str = "DQN Training Reward",
) -> None:
    """
    Plot episode reward and moving-average reward over training timesteps.
    """
    df = load_monitor_csv(monitor_path)
    df = add_training_timesteps(df)

    df["reward_ma"] = df["r"].rolling(window=window, min_periods=1).mean()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 5))
    plt.plot(df["timesteps"], df["r"], alpha=0.35, label="Episode reward")
    plt.plot(df["timesteps"], df["reward_ma"], label=f"Moving average ({window})")
    plt.xlabel("Training timesteps")
    plt.ylabel("Episode reward")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_episode_length(
    monitor_path: str,
    output_path: str,
    window: int = 20,
    title: str = "DQN Episode Length",
) -> None:
    """
    Plot episode length and moving-average episode length.
    This helps us see whether the agent learns to stop earlier.
    """
    df = load_monitor_csv(monitor_path)
    df = add_training_timesteps(df)

    df["length_ma"] = df["l"].rolling(window=window, min_periods=1).mean()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 5))
    plt.plot(df["timesteps"], df["l"], alpha=0.35, label="Episode length")
    plt.plot(df["timesteps"], df["length_ma"], label=f"Moving average ({window})")
    plt.xlabel("Training timesteps")
    plt.ylabel("Episode length")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def plot_stop_rate_proxy(
    monitor_path: str,
    output_path: str,
    window: int = 20,
    max_steps: int = 100,
    title: str = "DQN STOP Behavior Proxy",
) -> None:
    """
    Plot a proxy for STOP behavior.

    In this environment:
    - episodes shorter than max_steps usually ended by STOP
    - episodes equal to max_steps usually ended by truncation

    So we approximate STOP rate using:
        stopped = episode_length < max_steps
    """
    df = load_monitor_csv(monitor_path)
    df = add_training_timesteps(df)

    df["stopped"] = (df["l"] < max_steps).astype(float)
    df["stop_rate_ma"] = df["stopped"].rolling(window=window, min_periods=1).mean()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 5))
    plt.plot(df["timesteps"], df["stop_rate_ma"], label=f"STOP rate moving average ({window})")
    plt.xlabel("Training timesteps")
    plt.ylabel("STOP rate")
    plt.title(title)
    plt.ylim(-0.05, 1.05)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()