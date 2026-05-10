import argparse
import pickle
from pathlib import Path

import pandas as pd

from tsp_hh.instances import generate_euclidean_instance
from tsp_hh.hyper_env import TSPHyperHeuristicEnv
from tsp_hh.q_learning import QLearningAgent, discretize_state


def train_one_episode(
    agent: QLearningAgent,
    n_cities: int,
    seed: int,
    max_steps: int,
    initial_method: str,
) -> dict:
    """
    Train the Q-learning agent on one randomly generated TSP instance.
    """
    instance = generate_euclidean_instance(n_cities=n_cities, seed=seed)

    env = TSPHyperHeuristicEnv(
        distance_matrix=instance.distance_matrix,
        initial_method=initial_method,
        max_steps=max_steps,
        seed=seed,
    )

    state = env.reset()

    initial_length = env.best_length
    total_reward = 0.0
    action_counts = {name: 0 for name in env.ACTION_NAMES.values()}

    done = False

    while not done:
        state_key = discretize_state(
            state=state,
            initial_length=initial_length,
            max_steps=max_steps,
        )

        action = agent.select_action(state_key, training=True)

        next_state, reward, done, info = env.step(action)

        next_state_key = discretize_state(
            state=next_state,
            initial_length=initial_length,
            max_steps=max_steps,
        )

        agent.update(
            state_key=state_key,
            action=action,
            reward=reward,
            next_state_key=next_state_key,
            done=done,
        )

        total_reward += reward
        action_counts[info["action_name"]] += 1

        state = next_state

    agent.decay_epsilon()

    result = {
        "episode": seed,
        "n_cities": n_cities,
        "seed": seed,
        "initial_method": initial_method,
        "max_steps": max_steps,
        "initial_length": initial_length,
        "best_length": env.best_length,
        "improvement": initial_length - env.best_length,
        "total_reward": total_reward,
        "epsilon": agent.epsilon,
        "q_table_size": len(agent.q_table),
    }

    for action_name, count in action_counts.items():
        result[f"count_{action_name}"] = count

    return result


def save_q_table(agent: QLearningAgent, path: Path) -> None:
    """
    Save the learned Q-table and agent configuration.
    """
    payload = {
        "n_actions": agent.n_actions,
        "learning_rate": agent.learning_rate,
        "discount_factor": agent.discount_factor,
        "epsilon": agent.epsilon,
        "epsilon_min": agent.epsilon_min,
        "epsilon_decay": agent.epsilon_decay,
        "q_table": agent.q_table,
    }

    with path.open("wb") as f:
        pickle.dump(payload, f)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--episodes",
        type=int,
        default=500,
        help="Number of training episodes.",
    )

    parser.add_argument(
        "--n-cities",
        type=int,
        nargs="+",
        default=[20, 50, 100],
        help="Training instance sizes. One size is sampled per episode.",
    )

    parser.add_argument(
        "--max-steps",
        type=int,
        default=50,
        help="Number of hyper-heuristic decisions per episode.",
    )

    parser.add_argument(
        "--initial-method",
        type=str,
        default="nearest_neighbor",
        choices=["nearest_neighbor", "random"],
        help="Initial tour construction method.",
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.1,
        help="Q-learning learning rate.",
    )

    parser.add_argument(
        "--discount-factor",
        type=float,
        default=0.95,
        help="Q-learning discount factor.",
    )

    parser.add_argument(
        "--epsilon",
        type=float,
        default=1.0,
        help="Initial epsilon value.",
    )

    parser.add_argument(
        "--epsilon-min",
        type=float,
        default=0.05,
        help="Minimum epsilon value.",
    )

    parser.add_argument(
        "--epsilon-decay",
        type=float,
        default=0.995,
        help="Multiplicative epsilon decay per episode.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Global seed.",
    )

    parser.add_argument(
        "--out-dir",
        type=str,
        default="results/q_learning",
        help="Output directory.",
    )

    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    agent = QLearningAgent(
        n_actions=5,
        learning_rate=args.learning_rate,
        discount_factor=args.discount_factor,
        epsilon=args.epsilon,
        epsilon_min=args.epsilon_min,
        epsilon_decay=args.epsilon_decay,
        seed=args.seed,
    )

    logs = []

    for episode in range(args.episodes):
        n_cities = args.n_cities[episode % len(args.n_cities)]

        # Different seed per episode, but reproducible.
        instance_seed = args.seed + episode

        row = train_one_episode(
            agent=agent,
            n_cities=n_cities,
            seed=instance_seed,
            max_steps=args.max_steps,
            initial_method=args.initial_method,
        )

        row["episode"] = episode
        logs.append(row)

        if (episode + 1) % 25 == 0 or episode == 0:
            print(
                f"Episode {episode + 1}/{args.episodes} | "
                f"n_cities={n_cities} | "
                f"best_length={row['best_length']:.3f} | "
                f"improvement={row['improvement']:.3f} | "
                f"epsilon={agent.epsilon:.3f} | "
                f"q_states={len(agent.q_table)}"
            )

    log_df = pd.DataFrame(logs)

    log_path = out_dir / "q_learning_train_log.csv"
    q_table_path = out_dir / "q_table.pkl"

    log_df.to_csv(log_path, index=False)
    save_q_table(agent, q_table_path)

    print("\nTraining finished.")
    print(f"Training log saved to: {log_path}")
    print(f"Q-table saved to: {q_table_path}")
    print(f"Final epsilon: {agent.epsilon:.4f}")
    print(f"Q-table states: {len(agent.q_table)}")


if __name__ == "__main__":
    main()