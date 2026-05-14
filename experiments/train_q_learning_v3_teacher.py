import argparse
import pickle
from pathlib import Path

import pandas as pd

from tsp_hh.tsplib_loader import load_tsplib_instance
from tsp_hh.hyper_env_v3 import TSPHyperHeuristicEnvV3
from tsp_hh.q_learning_v3 import QLearningAgent, discretize_state_v3
from tsp_hh.rule_based_teacher_v3 import (
    RuleBasedTeacherV3,
    TeacherConfigV3,
    teacher_probability,
)


TSPLIB_OPTIMA = {
    "eil51": 426.0,
    "berlin52": 7542.0,
    "st70": 675.0,
    "eil76": 538.0,
    "pr76": 108159.0,
    "rat99": 1211.0,
    "kroA100": 21282.0,
    "kroB100": 22141.0,
    "eil101": 629.0,
    "ch130": 6110.0,
}

DEFAULT_CURRICULUM = [
    {
        "stage": 1,
        "instances": ["eil51", "berlin52", "st70"],
        "episodes": 700,
        "epsilon_reset": None,
    },
    {
        "stage": 2,
        "instances": ["eil76", "pr76", "rat99"],
        "episodes": 900,
        "epsilon_reset": 0.30,
    },
    {
        "stage": 3,
        "instances": ["kroA100", "kroB100", "eil101"],
        "episodes": 1000,
        "epsilon_reset": 0.25,
    },
    {
        "stage": 4,
        "instances": ["ch130"],
        "episodes": 900,
        "epsilon_reset": 0.25,
    },
]


def load_instances(data_dir: Path, names: list[str]) -> dict:
    instances = {}

    for name in names:
        path = data_dir / f"{name}.tsp"

        if not path.exists():
            raise FileNotFoundError(f"Missing TSPLIB file: {path}")

        if name not in TSPLIB_OPTIMA:
            raise ValueError(f"Known optimum missing for {name}")

        instances[name] = load_tsplib_instance(path)

    return instances


def save_q_table(agent: QLearningAgent, path: Path) -> None:
    payload = {
        "version": "v3_teacher_guided",
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


def train_one_episode(
    agent: QLearningAgent,
    teacher: RuleBasedTeacherV3,
    instance_name: str,
    instance,
    optimum: float,
    episode_seed: int,
    global_episode: int,
    total_episodes: int,
    max_steps: int,
    initial_method: str,
    construction_starts: int,
    two_opt_iterations: int,
    three_opt_samples: int,
    accept_worse_perturbation: bool,
    reward_scale: str,
    construction_penalty: float,
    teacher_start_prob: float,
    teacher_end_prob: float,
    teacher_decay_fraction: float,
) -> dict:
    env = TSPHyperHeuristicEnvV3(
        distance_matrix=instance.distance_matrix,
        initial_method=initial_method,
        max_steps=max_steps,
        seed=episode_seed,
        construction_starts=construction_starts,
        two_opt_iterations=two_opt_iterations,
        three_opt_samples=three_opt_samples,
        accept_worse_perturbation=accept_worse_perturbation,
        reward_scale=reward_scale,
        construction_penalty=construction_penalty,
    )

    state = env.reset()

    initial_length = env.best_length
    initial_gap = (initial_length - optimum) / optimum * 100.0

    teacher_prob = teacher_probability(
        episode=global_episode,
        total_episodes=total_episodes,
        start_prob=teacher_start_prob,
        end_prob=teacher_end_prob,
        decay_fraction=teacher_decay_fraction,
    )

    total_reward = 0.0
    teacher_actions = 0
    agent_actions = 0

    done = False

    while not done:
        state_key = discretize_state_v3(state, max_steps=max_steps)

        use_teacher = agent.rng.random() < teacher_prob

        if use_teacher:
            category = teacher.select_category(state)
            teacher_actions += 1
        else:
            category = agent.select_action(state_key, training=True)
            agent_actions += 1

        next_state, reward, done, info = env.step(category)

        next_state_key = discretize_state_v3(next_state, max_steps=max_steps)

        # Important: learn from both teacher-chosen and agent-chosen actions.
        agent.update(
            state_key=state_key,
            action=category,
            reward=reward,
            next_state_key=next_state_key,
            done=done,
        )

        total_reward += reward
        state = next_state

    agent.decay_epsilon()

    final_gap = (env.best_length - optimum) / optimum * 100.0
    gap_improvement = initial_gap - final_gap

    row = {
        "instance": instance_name,
        "n_cities": instance.coords.shape[0],
        "optimum": optimum,
        "seed": episode_seed,
        "initial_length": initial_length,
        "best_length": env.best_length,
        "improvement": initial_length - env.best_length,
        "initial_gap_percent": initial_gap,
        "final_gap_percent": final_gap,
        "gap_improvement_percent": gap_improvement,
        "total_reward": total_reward,
        "teacher_prob": teacher_prob,
        "teacher_actions": teacher_actions,
        "agent_actions": agent_actions,
        "epsilon": agent.epsilon,
        "q_table_size": len(agent.q_table),
    }

    for name, count in env.category_counts.items():
        row[f"count_category_{name}"] = count

    for name, count in env.heuristic_counts.items():
        clean = name.replace(":", "_")
        row[f"count_heuristic_{clean}"] = count

    return row


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data-dir", type=str, default="data/tsplib")
    parser.add_argument("--out-dir", type=str, default="results/q_learning_v3_teacher")

    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument(
        "--initial-method",
        type=str,
        default="greedy",
        choices=["random", "nearest_neighbor", "greedy"],
    )
    parser.add_argument("--construction-starts", type=int, default=30)
    parser.add_argument("--two-opt-iterations", type=int, default=100)
    parser.add_argument("--three-opt-samples", type=int, default=100)
    parser.add_argument("--accept-worse-perturbation", action="store_true")
    parser.add_argument("--reward-scale", type=str, default="initial_length", choices=["initial_length", "none"])
    parser.add_argument("--construction-penalty", type=float, default=0.0)

    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--discount-factor", type=float, default=0.95)
    parser.add_argument("--epsilon", type=float, default=1.0)
    parser.add_argument("--epsilon-min", type=float, default=0.02)
    parser.add_argument("--epsilon-decay", type=float, default=0.995)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--teacher-start-prob", type=float, default=1.0)
    parser.add_argument("--teacher-end-prob", type=float, default=0.0)
    parser.add_argument("--teacher-decay-fraction", type=float, default=0.85)

    parser.add_argument("--mild-stagnation", type=int, default=5)
    parser.add_argument("--medium-stagnation", type=int, default=15)
    parser.add_argument("--strong-stagnation", type=int, default=35)
    parser.add_argument("--restart-stagnation", type=int, default=60)

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_names = sorted({
        name
        for stage in DEFAULT_CURRICULUM
        for name in stage["instances"]
    })

    instances = load_instances(data_dir, all_names)

    total_episodes = sum(stage["episodes"] for stage in DEFAULT_CURRICULUM)

    agent = QLearningAgent(
        n_actions=3,
        learning_rate=args.learning_rate,
        discount_factor=args.discount_factor,
        epsilon=args.epsilon,
        epsilon_min=args.epsilon_min,
        epsilon_decay=args.epsilon_decay,
        seed=args.seed,
    )

    teacher = RuleBasedTeacherV3(
        TeacherConfigV3(
            mild_stagnation=args.mild_stagnation,
            medium_stagnation=args.medium_stagnation,
            strong_stagnation=args.strong_stagnation,
            restart_stagnation=args.restart_stagnation,
            force_improve_after_perturbation=True,
        )
    )

    logs = []
    global_episode = 0

    for stage_cfg in DEFAULT_CURRICULUM:
        stage = stage_cfg["stage"]
        stage_instances = stage_cfg["instances"]
        stage_episodes = stage_cfg["episodes"]

        epsilon_reset = stage_cfg.get("epsilon_reset")

        if epsilon_reset is not None:
            agent.epsilon = max(agent.epsilon, epsilon_reset)

        print("\n" + "=" * 80)
        print(f"Starting V3 teacher-guided stage {stage}")
        print(f"Instances: {stage_instances}")
        print(f"Episodes: {stage_episodes}")
        print(f"Epsilon: {agent.epsilon:.3f}")
        print("=" * 80)

        for local_episode in range(stage_episodes):
            instance_name = stage_instances[local_episode % len(stage_instances)]
            instance = instances[instance_name]
            optimum = TSPLIB_OPTIMA[instance_name]
            episode_seed = args.seed + global_episode

            row = train_one_episode(
                agent=agent,
                teacher=teacher,
                instance_name=instance_name,
                instance=instance,
                optimum=optimum,
                episode_seed=episode_seed,
                global_episode=global_episode,
                total_episodes=total_episodes,
                max_steps=args.max_steps,
                initial_method=args.initial_method,
                construction_starts=args.construction_starts,
                two_opt_iterations=args.two_opt_iterations,
                three_opt_samples=args.three_opt_samples,
                accept_worse_perturbation=args.accept_worse_perturbation,
                reward_scale=args.reward_scale,
                construction_penalty=args.construction_penalty,
                teacher_start_prob=args.teacher_start_prob,
                teacher_end_prob=args.teacher_end_prob,
                teacher_decay_fraction=args.teacher_decay_fraction,
            )

            row["global_episode"] = global_episode
            row["stage"] = stage
            row["stage_episode"] = local_episode

            logs.append(row)
            global_episode += 1

            if (local_episode + 1) % 25 == 0 or local_episode == 0:
                print(
                    f"Stage {stage} | "
                    f"ep {local_episode + 1}/{stage_episodes} | "
                    f"global={global_episode}/{total_episodes} | "
                    f"inst={instance_name} | "
                    f"gap={row['final_gap_percent']:.3f}% | "
                    f"gap_impr={row['gap_improvement_percent']:.3f}% | "
                    f"teacher_p={row['teacher_prob']:.3f} | "
                    f"eps={agent.epsilon:.3f} | "
                    f"q_states={len(agent.q_table)}"
                )

    log_df = pd.DataFrame(logs)

    log_path = out_dir / "q_learning_v3_teacher_log.csv"
    q_table_path = out_dir / "q_table_v3_teacher.pkl"

    log_df.to_csv(log_path, index=False)
    save_q_table(agent, q_table_path)

    print("\nV3 teacher-guided training finished.")
    print(f"Training log saved to: {log_path}")
    print(f"Q-table saved to: {q_table_path}")
    print(f"Final epsilon: {agent.epsilon:.4f}")
    print(f"Q-table states: {len(agent.q_table)}")


if __name__ == "__main__":
    main()