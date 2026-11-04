"""Open the MuJoCo viewer and replay a policy or random actions."""

import argparse
import copy
from pathlib import Path
import time

import mujoco
import mujoco.viewer
import numpy as np
import torch
from stable_baselines3 import PPO

from kinova_env import KinovaReachEnv
from policy_tools import (add_controller_arguments, controller_options,
                          resolve_core_arguments, core_options, load_updated_checkpoint)


def set_target(viewer, target):
    with viewer.lock():
        viewer.user_scn.ngeom = 1
        mujoco.mjv_initGeom(
            viewer.user_scn.geoms[0],
            mujoco.mjtGeom.mjGEOM_SPHERE,
            np.array([0.005, 0.005, 0.005]),
            np.asarray(target, dtype=np.float64),
            np.eye(3).reshape(-1),
            np.array([0.15, 0.75, 1.0, 0.7]),
        )


def main():
    parser = argparse.ArgumentParser(description="View Kinova Gen3 reaching in MuJoCo")
    add_controller_arguments(parser)
    parser.add_argument("--model", default="models/ppo_gen3_reach_ball_best.zip")
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--action-scale", type=float, default=None,
                        help="Must match the action scale used during training")
    parser.add_argument("--target-frequency-hz", type=float, default=None,
                        help="Must match training; 0.25 means a new target every 4 seconds")
    parser.add_argument("--joint-target-speed-limit", type=float, default=None,
                        help="Must match training; maximum commanded joint-position rate in rad/s")
    parser.add_argument("--joint-target-accel-limit", type=float, default=None,
                        help="Must match training; maximum commanded joint acceleration in rad/s^2")
    parser.add_argument("--orientation-reward-weight", type=float, default=None,
                        help="Must match training; 0 for a sphere position-reaching task")
    parser.add_argument("--random-target-yaw", action=argparse.BooleanOptionalAction, default=None,
                        help="Randomize target yaw; off by default because the ball has no orientation")
    parser.add_argument("--position-reward-weight", type=float, default=None,
                        help="Must match training")
    parser.add_argument("--random", action="store_true")
    parser.add_argument("--watch-model", action="store_true",
                        help="Load new best checkpoints between targets during live CPU training")
    args = parser.parse_args()
    overrides = copy.copy(args)

    torch.set_num_threads(1)
    reference = PPO.load(args.model, device="cpu") if (not args.random or Path(args.model).is_file()) else None
    policy = None if args.random else reference
    resolve_core_arguments(args, reference)
    env = KinovaReachEnv(
        # Playback is a continuous sequence of 4 s targets. An episode reset
        # every 12 s would teleport the arm back to a randomized configuration.
        max_steps=2**31 - 1,
        action_scale=args.action_scale,
        target_frequency_hz=args.target_frequency_hz,
        joint_target_speed_limit=args.joint_target_speed_limit,
        joint_target_accel_limit=args.joint_target_accel_limit,
        position_reward_weight=args.position_reward_weight,
        orientation_reward_weight=args.orientation_reward_weight,
        random_target_yaw=args.random_target_yaw,
        **controller_options(args, reference),
    )
    env.action_space.seed(args.seed)
    observation, _ = env.reset(seed=args.seed)
    target = env.target_pos.copy()
    model_stamp = Path(args.model).stat().st_mtime_ns if args.watch_model else None

    try:
        with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
            viewer.cam.lookat[:] = [0.45, 0.0, 0.25]
            viewer.cam.distance = 2.3
            viewer.cam.azimuth = 135
            viewer.cam.elevation = -22
            set_target(viewer, env.reach_position)
            while viewer.is_running():
                tick = time.perf_counter()
                if policy is None:
                    action = env.action_space.sample()
                else:
                    action, _ = policy.predict(observation, deterministic=True)
                observation, _, terminated, truncated, _ = env.step(action)
                if not np.array_equal(target, env.target_pos):
                    if args.watch_model and policy is not None:
                        checkpoint = load_updated_checkpoint(args.model, model_stamp)
                        if checkpoint is not None:
                            candidate, stamp = checkpoint
                            updated_args = copy.copy(overrides)
                            resolve_core_arguments(updated_args, candidate)
                            updated_config = {**core_options(updated_args),
                                              **controller_options(updated_args, candidate)}
                            changed = [key for key, value in updated_config.items()
                                       if key != "action_scale" and getattr(env, key) != value]
                            if changed or candidate.observation_space.shape != env.observation_space.shape:
                                raise RuntimeError(f"Checkpoint simulator settings changed: {changed}; restart playback")
                            env.set_action_scale(updated_config["action_scale"])
                            observation = env._observation()
                            policy, model_stamp = candidate, stamp
                            print(f"Loaded best policy at {policy.num_timesteps:,} steps", flush=True)
                    target = env.target_pos.copy()
                    set_target(viewer, env.reach_position)
                if terminated:
                    raise RuntimeError("Simulation became non-finite; stopping playback")
                viewer.sync()
                elapsed = time.perf_counter() - tick
                time.sleep(max(0.0, env.control_dt - elapsed))
    finally:
        env.close()


if __name__ == "__main__":
    main()
