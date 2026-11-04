"""Check dependencies, the bundled robot model, and the Gymnasium contract."""

from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO
import argparse
from pathlib import Path
import numpy as np

from kinova_env import KinovaReachEnv
from policy_tools import add_controller_arguments, controller_options, CORE_ENV_DEFAULTS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default='models/ppo_gen3_reach_ball_best.zip')
    add_controller_arguments(parser)
    args = parser.parse_args()
    policy = PPO.load(args.model, device='cpu') if Path(args.model).is_file() else None
    saved = getattr(policy, 'reach_config', {})
    core = {key: saved.get(key, value) for key, value in CORE_ENV_DEFAULTS.items()}
    env = KinovaReachEnv(**core, **controller_options(args, policy))
    try:
        check_env(env, warn=True)
        observation, info = env.reset(seed=42)
        if policy is not None and observation.shape != policy.observation_space.shape:
            raise RuntimeError('Checkpoint observation size does not match the environment')
        print("Physics engine: MuJoCo (CPU)")
        print(f"Robot model: Kinova Gen3 ({env.model.njnt} joints / {env.model.nu} actuators)")
        print(f"Observation/action: {observation.shape} / {env.action_space.shape}")
        print(f"Simulation/control step: {env.model.opt.timestep:.4f} / {env.control_dt:.4f} s")
        print(f"Episode/goal refresh: {env.max_steps} / {env.command_resample_steps} control steps")
        print(f"Initial position/orientation error: {info['position_error']:.3f} m / "
              f"{info['orientation_error']:.3f} rad")

        for _ in range(40):
            observation, reward, terminated, truncated, info = env.step(env.action_space.sample())
            if terminated or truncated:
                observation, info = env.reset()
            if not (observation.shape == env.observation_space.shape and np.isfinite(reward)):
                raise RuntimeError("Non-finite reward or unexpected observation shape")
        print("Model load, reset, action, observation, and finite-reward checks passed")
    finally:
        env.close()


if __name__ == "__main__":
    main()
