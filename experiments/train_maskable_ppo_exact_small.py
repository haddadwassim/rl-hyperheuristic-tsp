import argparse
from pathlib import Path

from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.evaluation import evaluate_policy
from sb3_contrib.common.maskable.utils import get_action_masks
from stable_baselines3.common.monitor import Monitor

from tsp_hh.instances import generate_euclidean_instance
from tsp_hh.exact_tsp import held_karp_optimal_tour
from tsp_hh.path_env import TSPPathEnv


def make_env(
    n_cities: int,
    instance_seed: int,
    teacher_bonus: float,
    final_optimal_bonus: float,
):
    instance = generate_euclidean_instance(
        n_cities=n_cities,
        seed=instance_seed,
    )

    optimal_tour, optimal_length = held_karp_optimal_tour(
        instance.distance_matrix,
        start_city=0,
    )

    env = TSPPathEnv(
        coords=instance.coords,
        optimal_tour=optimal_tour,
        start_city=0,
        teacher_bonus=teacher_bonus,
        final_optimal_bonus=final_optimal_bonus,
    )

    return Monitor(env), optimal_tour, optimal_length


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
    parser = argparse.ArgumentParser()

    parser.add_argument("--n-cities", type=int, default=10)
    parser.add_argument("--instance-seed", type=int, default=42)
    parser.add_argument("--model-seed", type=int, default=42)
    parser.add_argument("--timesteps", type=int, default=100_000)

    parser.add_argument("--teacher-bonus", type=float, default=3.0)
    parser.add_argument("--final-optimal-bonus", type=float, default=50.0)

    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--n-steps", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--ent-coef", type=float, default=0.01)

    parser.add_argument("--eval-episodes", type=int, default=50)
    parser.add_argument("--print-episodes", type=int, default=10)

    parser.add_argument(
        "--out-dir",
        type=str,
        default="results/maskable_ppo_exact_small",
    )

    args = parser.parse_args()

    out_dir = Path(args.out_dir) / f"n{args.n_cities}_seed{args.instance_seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    env, optimal_tour, optimal_length = make_env(
        n_cities=args.n_cities,
        instance_seed=args.instance_seed,
        teacher_bonus=args.teacher_bonus,
        final_optimal_bonus=args.final_optimal_bonus,
    )

    print("Exact optimal tour:")
    print(optimal_tour)
    print(f"Exact optimal length: {optimal_length:.6f}")

    model = MaskablePPO(
        policy="MultiInputPolicy",
        env=env,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        gamma=args.gamma,
        ent_coef=args.ent_coef,
        verbose=1,
        seed=args.model_seed,
    )

    model.learn(total_timesteps=args.timesteps)

    model_path = out_dir / "maskable_ppo_exact_small"
    model.save(model_path)

    print(f"Saved model to: {model_path}")

    mean_reward, std_reward = evaluate_policy(
        model,
        env,
        n_eval_episodes=args.eval_episodes,
        deterministic=True,
        warn=False,
    )

    print(f"Mean reward: {mean_reward:.3f} +/- {std_reward:.3f}")

    success_count = 0
    best_distance = float("inf")

    for i in range(args.print_episodes):
        info, total_reward = run_one_episode(model, env, deterministic=True)

        is_exact = info["path"] == optimal_tour
        distance = info["total_distance"]
        best_distance = min(best_distance, distance)

        if is_exact:
            success_count += 1

        gap_percent = (distance - optimal_length) / optimal_length * 100.0

        print(
            f"Episode {i + 1}: "
            f"path={info['path']}, "
            f"optimal_tour={optimal_tour}, "
            f"distance={distance:.6f}, "
            f"optimal_path={is_exact}, "
            f"gap={gap_percent:.3f}%, "
            f"optimal_length={optimal_length:.6f}, "
            f"reward={total_reward:.3f}"
        )

    best_gap_percent = (best_distance - optimal_length) / optimal_length * 100.0

    print(f"Exact-path success: {success_count}/{args.print_episodes}")
    print(f"Best distance observed: {best_distance:.6f}")
    print(f"Best gap observed: {best_gap_percent:.3f}%")


if __name__ == "__main__":
    main()