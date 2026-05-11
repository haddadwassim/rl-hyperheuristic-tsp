import argparse
import pickle
from pathlib import Path

import pandas as pd

from tsp_hh.instances import generate_euclidean_instance
from tsp_hh.hyper_env_v2 import TSPHyperHeuristicEnvV2
from tsp_hh.q_learning_v2 import QLearningAgent, discretize_state_v2


def train_one_episode_v2(
    agent: QLearningAgent,
    n_cities: int,
    seed: int,
    max_steps: int,
    initial_method: str,
    k_neighbors: int,
    perturbation_moves: int,
    reward_scale: str,
    accept_worse_current: bool,
) -> dict:
    instance = generate_euclidean_instance(n_cities=n_cities, seed=seed)

    env = TSPHyperHeuristicEnvV2(
        distance_matrix=instance.distance_matrix,
        initial_method=initial_method,
        max_steps=max_steps,
        seed=seed,
        k_neighbors=k_neighbors,
        perturbation_moves=perturbation_moves,
        reward_scale=reward_scale,
        accept_worse_current=accept_worse_current,
    )

    state = env.reset()

    initial_length = env.best_length
    total_reward = 0.0
    total_raw_improvement = 0.0

    action_counts = {name: 0 for name in env.ACTION_NAMES.values()}

    done = False

    while not done:
        state_key = discretize_state_v2(
            state=state,
            initial_length=initial_length,
            max_steps=max_steps,
        )

        action = agent.select_action(state_key, training=True)

        next_state, reward, done, info = env.step(action)

        next_state_key = discretize_state_v2(
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
        total_raw_improvement += info["raw_improvement"]
        action_counts[info["action_name"]] += 1

        state = next_state

    agent.decay_epsilon()

    row = {
        "n_cities": n_cities,
        "seed": seed,
        "initial_method": initial_method,
        "max_steps": max_steps,
        "k_neighbors": k_neighbors,
        "perturbation_moves": perturbation_moves,
        "reward_scale": reward_scale,
        "accept_worse_current": accept_worse_current,
        "initial_length": initial_length,
        "best_length": env.best_length,
        "improvement": initial_length - env.best_length,
        "total_raw_improvement": total_raw_improvement,
        "total_reward": total_reward,
        "epsilon": agent.epsilon,
        "q_table_size": len(agent.q_table),
    }

    for action_name, count in action_counts.items():
        row[f"count_{action_name}"] = count

    return row


def save_q_table(agent: QLearningAgent, path: Path) -> None:
    payload = {
        "version": "v2",
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

    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--n-cities", type=int, nargs="+", default=[20, 50, 100])
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument(
        "--initial-method",
        type=str,
        default="nearest_neighbor",
        choices=["nearest_neighbor", "random"],
    )
    parser.add_argument("--k-neighbors", type=int, default=10)
    parser.add_argument("--perturbation-moves", type=int, default=3)
    parser.add_argument(
        "--reward-scale",
        type=str,
        default="initial_length",
        choices=["initial_length", "none"],
    )
    parser.add_argument(
        "--accept-worse-current",
        action="store_true",
        help="Allow current solution to move to worse candidates.",
    )

    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--discount-factor", type=float, default=0.95)
    parser.add_argument("--epsilon", type=float, default=1.0)
    parser.add_argument("--epsilon-min", type=float, default=0.02)
    parser.add_argument("--epsilon-decay", type=float, default=0.998)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--out-dir", type=str, default="results/q_learning_v2")

    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    agent = QLearningAgent(
        n_actions=6,
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
        instance_seed = args.seed + episode

        row = train_one_episode_v2(
            agent=agent,
            n_cities=n_cities,
            seed=instance_seed,
            max_steps=args.max_steps,
            initial_method=args.initial_method,
            k_neighbors=args.k_neighbors,
            perturbation_moves=args.perturbation_moves,
            reward_scale=args.reward_scale,
            accept_worse_current=args.accept_worse_current,
        )

        row["episode"] = episode
        logs.append(row)

        if (episode + 1) % 50 == 0 or episode == 0:
            print(
                f"Episode {episode + 1}/{args.episodes} | "
                f"n={n_cities} | "
                f"best={row['best_length']:.3f} | "
                f"improvement={row['improvement']:.3f} | "
                f"reward={row['total_reward']:.6f} | "
                f"epsilon={agent.epsilon:.3f} | "
                f"q_states={len(agent.q_table)}"
            )

    log_df = pd.DataFrame(logs)

    log_path = out_dir / "q_learning_v2_train_log.csv"
    q_table_path = out_dir / "q_table_v2.pkl"

    log_df.to_csv(log_path, index=False)
    save_q_table(agent, q_table_path)

    print("\nTraining V2 finished.")
    print(f"Training log saved to: {log_path}")
    print(f"Q-table saved to: {q_table_path}")
    print(f"Final epsilon: {agent.epsilon:.4f}")
    print(f"Q-table states: {len(agent.q_table)}")


if __name__ == "__main__":
    main()