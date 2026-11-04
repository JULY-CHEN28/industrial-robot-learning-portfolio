"""Evaluate a learned or random policy on seeded reach episodes."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO

from kinova_env import KinovaReachEnv
from policy_tools import add_controller_arguments, controller_options, resolve_core_arguments, core_options


def main():
    parser = argparse.ArgumentParser(description="Evaluate the Gen3 reach policy on CPU")
    add_controller_arguments(parser)
    parser.add_argument("--model", default="models/ppo_gen3_reach_ball_best.zip")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--episode-seconds", type=float, default=12,
                        help="Use e.g. 60 for continuous reaching without a robot reset between targets")
    parser.add_argument("--report-json", type=Path, help="Save reproducible metrics and checkpoint settings")
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
    parser.add_argument("--random", action="store_true", help="Use uniform random actions as a baseline")
    args = parser.parse_args()
    torch.set_num_threads(1)
    reference = PPO.load(args.model, device="cpu") if (not args.random or Path(args.model).is_file()) else None
    policy = None if args.random else reference
    resolve_core_arguments(args, reference)
    if not np.isfinite(args.episode_seconds) or args.episode_seconds <= 0:
        parser.error("episode seconds must be finite and positive")
    if (args.episodes <= 0 or args.action_scale <= 0
            or not np.isfinite(args.target_frequency_hz) or args.target_frequency_hz <= 0
            or args.joint_target_speed_limit <= 0 or args.joint_target_accel_limit <= 0
            or args.orientation_reward_weight < 0 or args.position_reward_weight < 0):
        parser.error("episodes, action scale, target frequency and joint target speed must be positive; orientation weight cannot be negative")

    torch.set_num_threads(1)
    env = KinovaReachEnv(
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
    env.max_steps = max(1, int(round(args.episode_seconds / env.control_dt)))

    step_position_errors = []
    step_orientation_errors = []
    position_success_steps = 0
    pose_success_steps = 0
    total_steps = 0
    position_success_episodes = 0
    pose_success_episodes = 0
    target_success_windows = 0
    stable_target_windows = 0
    total_target_windows = 0
    target_success_latencies = []
    episode_final_errors = []
    episode_final_orientation_errors = []
    joint_speeds = []
    joint_command_speeds = []
    joint_command_accels = []
    action_changes = []
    surface_clearances = []
    obstacle_contact_steps = 0
    max_obstacle_penetration = 0.0
    joint_accelerations = []
    max_joint_speed = 0.0
    max_joint_step = 0.0
    max_force_limit_ratio = 0.0
    physics_warning_count = 0
    try:
        for episode in range(args.episodes):
            observation, _ = env.reset(seed=args.seed + episode)
            done = False
            episode_position_success = False
            episode_pose_success = False
            episode_target_steps = {}
            episode_target_success_steps = {}
            streaks = {}
            longest_streaks = {}
            while not done:
                if policy is None:
                    action = env.action_space.sample()
                else:
                    action, _ = policy.predict(observation, deterministic=True)
                old_position = env.data.qpos.copy()
                old_velocity = env.data.qvel.copy()
                observation, _, terminated, truncated, info = env.step(action)
                joint_accelerations.append(np.linalg.norm(env.data.qvel - old_velocity) / env.control_dt)
                max_joint_speed = max(max_joint_speed, float(np.max(np.abs(env.data.qvel))))
                max_joint_step = max(max_joint_step, float(np.max(np.abs(env.data.qpos - old_position))))
                max_force_limit_ratio = max(max_force_limit_ratio, float(np.max(
                    np.abs(env.data.actuator_force) / env.model.actuator_forcerange[:, 1]
                )))
                done = terminated or truncated

                step_position_errors.append(info["position_error"])
                step_orientation_errors.append(info["orientation_error"])
                joint_speeds.append(info["joint_speed"])
                joint_command_speeds.append(info["joint_command_speed"])
                joint_command_accels.append(info["joint_command_accel"])
                action_changes.append(info["action_change"])
                surface_clearances.append(info["ball_surface_clearance"])
                obstacle_contact_steps += int(info["obstacle_contacts"] > 0)
                max_obstacle_penetration = max(max_obstacle_penetration, info["obstacle_penetration"])
                position_success_steps += int(info["is_success"])
                pose_success_steps += int(info["is_pose_success"])
                episode_position_success |= bool(info["is_success"])
                episode_pose_success |= bool(info["is_pose_success"])
                target_index = int(info["target_index"])
                streaks[target_index] = streaks.get(target_index, 0) + 1 if info["is_success"] else 0
                longest_streaks[target_index] = max(
                    longest_streaks.get(target_index, 0), streaks[target_index]
                )
                episode_target_steps[target_index] = episode_target_steps.get(target_index, 0) + 1
                if info["is_success"]:
                    episode_target_success_steps.setdefault(
                        target_index, episode_target_steps[target_index]
                    )
                total_steps += 1
            position_success_episodes += int(episode_position_success)
            pose_success_episodes += int(episode_pose_success)
            total_target_windows += len(episode_target_steps)
            target_success_windows += len(episode_target_success_steps)
            stable_target_windows += sum(
                count * env.control_dt >= 0.5 for count in longest_streaks.values()
            )
            target_success_latencies.extend(episode_target_success_steps.values())
            episode_final_errors.append(info["position_error"])
            episode_final_orientation_errors.append(info["orientation_error"])
            physics_warning_count += sum(int(warning.number) for warning in env.data.warning)

        label = "Random baseline" if policy is None else "PPO policy"
        threshold = f"<{env.position_tolerance * 100:g} cm"
        print(f"Policy: {label}")
        print(f"Episodes: {args.episodes} ({total_steps} control steps)")
        print(f"Episodes with any position success ({threshold}): {position_success_episodes / args.episodes:.1%}")
        print(
            f"Target windows reached ({threshold}): "
            f"{target_success_windows}/{total_target_windows} "
            f"({target_success_windows / total_target_windows:.1%})"
        )
        if target_success_latencies:
            print(
                "Mean control steps to first reach within each successful target window: "
                f"{np.mean(target_success_latencies):.1f} "
                f"({np.mean(target_success_latencies) * env.control_dt:.2f} s)"
            )
        print(f"Target windows held {threshold} for at least 0.5 s: "
              f"{stable_target_windows}/{total_target_windows} "
              f"({stable_target_windows / total_target_windows:.1%})")
        print(f"Episodes with any pose success ({threshold} and <0.20 rad): {pose_success_episodes / args.episodes:.1%}")
        print(f"Position success steps ({threshold}): {position_success_steps / total_steps:.1%}")
        print(f"Pose success steps ({threshold} and <0.20 rad): {pose_success_steps / total_steps:.1%}")
        if env.physical_obstacles:
            print("Position success also requires >=5 mm robot-to-ball clearance and no obstacle contact")
            print(f"Obstacle-contact steps: {obstacle_contact_steps / total_steps:.1%}")
            print(f"Maximum obstacle penetration: {max_obstacle_penetration * 1000:.3f} mm")
            print(f"Minimum robot-to-ball surface clearance: {min(surface_clearances) * 1000:.3f} mm")
        print(f"Mean position error over all steps: {np.mean(step_position_errors):.3f} m")
        print(f"Mean orientation error over all steps: {np.mean(step_orientation_errors):.3f} rad")
        print(f"Mean joint speed: {np.mean(joint_speeds):.3f} rad/s")
        print(f"Mean commanded joint speed: {np.mean(joint_command_speeds):.3f} rad/s")
        print(f"Mean commanded joint acceleration: {np.mean(joint_command_accels):.3f} rad/s^2")
        print(f"Mean action change per control step: {np.mean(action_changes):.3f}")
        print(f"Mean error at episode end: {np.mean(episode_final_errors):.3f} m")
        print(f"Mean orientation error at episode end: {np.mean(episode_final_orientation_errors):.3f} rad")
        print(f"Maximum joint speed / step change: {max_joint_speed:.3f} rad/s / {max_joint_step:.4f} rad")
        print(f"Physics warnings: {physics_warning_count}; maximum actuator force/limit: {max_force_limit_ratio:.3f}")
        if args.report_json is not None:
            report = {
                "model": str(Path(args.model).resolve()),
                "checkpoint_steps": reference.num_timesteps if reference is not None else None,
                "model_sha256": hashlib.sha256(Path(args.model).read_bytes()).hexdigest()
                    if reference is not None else None,
                "policy": label, "device": str(reference.device) if reference is not None else "cpu",
                "seed": args.seed, "episodes": args.episodes,
                "episode_seconds": env.max_steps * env.control_dt,
                "control_steps": total_steps, "control_dt": env.control_dt,
                "target_interval_seconds": env.command_resample_steps * env.control_dt,
                "observation_size": env.observation_space.shape[0],
                "configuration": {**core_options(args), **controller_options(args, reference)},
                "position_tolerance_m": env.position_tolerance,
                "minimum_success_surface_gap_m": .005 if env.physical_obstacles else None,
                "hold_seconds": .5, "target_windows": total_target_windows,
                "reached_windows": target_success_windows,
                "held_windows": stable_target_windows,
                "reach_rate": target_success_windows / total_target_windows,
                "hold_rate": stable_target_windows / total_target_windows,
                "success_step_rate": position_success_steps / total_steps,
                "mean_first_reach_seconds": float(np.mean(target_success_latencies) * env.control_dt)
                    if target_success_latencies else None,
                "mean_position_error_m": float(np.mean(step_position_errors)),
                "mean_final_error_m": float(np.mean(episode_final_errors)),
                "mean_joint_speed_norm_rad_s": float(np.mean(joint_speeds)),
                "p95_joint_acceleration_norm_rad_s2": float(np.percentile(joint_accelerations, 95)),
                "maximum_joint_speed_rad_s": max_joint_speed,
                "maximum_joint_step_rad": max_joint_step,
                "obstacle_contact_step_rate": obstacle_contact_steps / total_steps,
                "maximum_obstacle_penetration_mm": max_obstacle_penetration * 1000,
                "minimum_ball_surface_clearance_mm": min(surface_clearances) * 1000,
                "physics_warnings": physics_warning_count,
                "maximum_actuator_force_limit_ratio": max_force_limit_ratio,
            }
            args.report_json.parent.mkdir(parents=True, exist_ok=True)
            args.report_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    finally:
        env.close()


if __name__ == "__main__":
    main()
