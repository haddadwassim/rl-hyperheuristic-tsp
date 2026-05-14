import argparse
import pickle
from pathlib import Path

import pandas as pd

from tsp_hh.tsplib_loader import load_tsplib_instance
from tsp_hh.hyper_env_v2 import TSPHyperHeuristicEnvV2
from tsp_hh.q_learning_v2 import QLearningAgent, discretize_state_v2


TSPLIB_OPTIMA = {
    "eil51": 426.0,
    "berlin52": 7542.0,
    "st70": 675.0,
    "eil76": 538.0,
    "kroA100": 21282.0,
    "ch130": 6110.0,
}


DEFAULT_CURRICULUM = [
    {
        "stage": 1,
        "instances": ["eil51", "berlin52"],
        "episodes": 500,
    },
    {
        "stage": 2,
        "instances": ["st70", "eil76"],
        "episodes": 700,
    },
    {
        "stage": 3,
        "instances": ["kroA100"],
        "episodes": 900,
    },
    {
        "stage": 4,
        "instances": ["ch130"],
        "episodes": 900,
    },
]


def load_curriculum_instances(data_dir: Path, instance_names: list[str]) -> dict:
    instances = {}

    for name in instance_names:
        path = data_dir / f"{name}.tsp"

        if not path.exists():
            raise FileNotFoundError(f"Missing TSPLIB file: {path}")

        if name not in TSPLIB_OPTIMA:
            raise ValueError(f"Known optimum missing for instance: {name}")

        instances[name] = load_tsplib_instance(path)

    return instances


def train_one_tsplib_episode(
    agent: QLearningAgent,
    instance_name: str,
    instance,
    optimum: float,
    episode_seed: int,
    max_steps: int,
    initial_method: str,
    k_neighbors: int,
    perturbation_moves: int,
    reward_scale: str,
    accept_worse_current: bool,
) -> dict:
    env = TSPHyperHeuristicEnvV2(
        distance_matrix=instance.distance_matrix,
        initial_method=initial_method,
        max_steps=max_steps,
        seed=episode_seed,
        k_neighbors=k_neighbors,
        perturbation_moves=perturbation_moves,
        reward_scale=reward_scale,
        accept_worse_current=accept_worse_current,
    )

    state = env.reset()

    initial_length = env.best_length
    initial_gap = (initial_length - optimum) / optimum * 100.0

    total_reward = 0.0
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
        action_counts[info["action_name"]] += 1

        state = next_state

    agent.decay_epsilon()

    final_gap = (env.best_length - optimum) / optimum * 100.0
    gap_improvement = initial_gap - final_gap

    row = {
        "instance": instance_name,
        "n_cities": instance.coords.shape[0],
        "optimum": optimum,
        "seed": episode_seed,
        "initial_method": initial_method,
        "max_steps": max_steps,
        "k_neighbors": k_neighbors,
        "initial_length": initial_length,
        "best_length": env.best_length,
        "improvement": initial_length - env.best_length,
        "initial_gap_percent": initial_gap,
        "final_gap_percent": final_gap,
        "gap_improvement_percent": gap_improvement,
        "total_reward": total_reward,
        "epsilon": agent.epsilon,
        "q_table_size": len(agent.q_table),
    }

    for action_name, count in action_counts.items():
        row[f"count_{action_name}"] = count

    return row


def save_q_table(agent: QLearningAgent, path: Path) -> None:
    payload = {
        "version": "v2_tsplib_curriculum",
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

    parser.add_argument("--data-dir", type=str, default="data/tsplib")
    parser.add_argument("--out-dir", type=str, default="results/q_learning_v2_tsplib_curriculum")

    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--initial-method", type=str, default="nearest_neighbor", choices=["nearest_neighbor", "random"])
    parser.add_argument("--k-neighbors", type=int, default=10)
    parser.add_argument("--perturbation-moves", type=int, default=3)
    parser.add_argument("--reward-scale", type=str, default="initial_length", choices=["initial_length", "none"])
    parser.add_argument("--accept-worse-current", action="store_true")

    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--discount-factor", type=float, default=0.95)
    parser.add_argument("--epsilon", type=float, default=1.0)
    parser.add_argument("--epsilon-min", type=float, default=0.02)
    parser.add_argument("--epsilon-decay", type=float, default=0.998)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_instance_names = sorted({
        name
        for stage in DEFAULT_CURRICULUM
        for name in stage["instances"]
    })

    instances = load_curriculum_instances(data_dir, all_instance_names)

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
    global_episode = 0

    for stage_cfg in DEFAULT_CURRICULUM:
        stage = stage_cfg["stage"]
        stage_instances = stage_cfg["instances"]
        stage_episodes = stage_cfg["episodes"]

        print("\n" + "=" * 80)
        print(f"Starting curriculum stage {stage}")
        print(f"Instances: {stage_instances}")
        print(f"Episodes: {stage_episodes}")
        print("=" * 80)

        for local_episode in range(stage_episodes):
            instance_name = stage_instances[local_episode % len(stage_instances)]
            instance = instances[instance_name]
            optimum = TSPLIB_OPTIMA[instance_name]

            episode_seed = args.seed + global_episode

            row = train_one_tsplib_episode(
                agent=agent,
                instance_name=instance_name,
                instance=instance,
                optimum=optimum,
                episode_seed=episode_seed,
                max_steps=args.max_steps,
                initial_method=args.initial_method,
                k_neighbors=args.k_neighbors,
                perturbation_moves=args.perturbation_moves,
                reward_scale=args.reward_scale,
                accept_worse_current=args.accept_worse_current,
            )

            row["global_episode"] = global_episode
            row["stage"] = stage
            row["stage_episode"] = local_episode

            logs.append(row)

            global_episode += 1

            if (local_episode + 1) % 50 == 0 or local_episode == 0:
                print(
                    f"Stage {stage} | "
                    f"episode {local_episode + 1}/{stage_episodes} | "
                    f"global={global_episode} | "
                    f"instance={instance_name} | "
                    f"gap={row['final_gap_percent']:.3f}% | "
                    f"gap_impr={row['gap_improvement_percent']:.3f}% | "
                    f"epsilon={agent.epsilon:.3f} | "
                    f"q_states={len(agent.q_table)}"
                )

    log_df = pd.DataFrame(logs)

    log_path = out_dir / "q_learning_v2_tsplib_curriculum_log.csv"
    q_table_path = out_dir / "q_table_v2_tsplib_curriculum.pkl"

    log_df.to_csv(log_path, index=False)
    save_q_table(agent, q_table_path)

    print("\nCurriculum training finished.")
    print(f"Training log saved to: {log_path}")
    print(f"Q-table saved to: {q_table_path}")
    print(f"Final epsilon: {agent.epsilon:.4f}")
    print(f"Q-table states: {len(agent.q_table)}")


if __name__ == "__main__":
    main()