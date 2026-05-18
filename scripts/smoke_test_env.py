import yaml

from src.env.tsp_operator_env import TSPOperatorEnv
from src.operators.registry import ACTION_NAMES


def load_config(path: str = "configs/default.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    config = load_config()

    env = TSPOperatorEnv(config=config)

    obs, info = env.reset(seed=42)

    print("Initial observation:")
    print(obs)
    print("Initial info:")
    print(info)
    print()

    # Force a meaningful operator sequence:
    # 1 = two_opt, 2 = swap, 3 = relocate, 4 = perturb_two_opt, 0 = stop
    actions = [1, 2, 3, 4, 1, 2, 0]

    total_reward = 0.0

    for step, action in enumerate(actions, start=1):
        obs, reward, terminated, truncated, info = env.step(action)

        total_reward += reward

        print(f"Step {step}")
        print(f"  action: {action} ({ACTION_NAMES[action]})")
        print(f"  reward: {reward:.6f}")
        print(f"  current_length: {info['current_length']:.6f}")
        print(f"  best_length: {info['best_length']:.6f}")
        print(f"  relative_improvement: {info['relative_improvement']:.6f}")
        print(f"  terminated: {terminated}")
        print(f"  truncated: {truncated}")
        print()

        if terminated or truncated:
            break

    print("Total reward:", total_reward)


if __name__ == "__main__":
    main()