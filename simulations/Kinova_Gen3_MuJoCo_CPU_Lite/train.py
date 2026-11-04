"""Train the task-level PPO policy on CPU.

Examples:
    python train.py --steps 300000 --n-envs 4
    python train.py --steps 50000 --n-envs 1
"""

import argparse
import csv
import json
from functools import partial
from pathlib import Path

import numpy as np
import torch
from torch import nn
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import ConstantSchedule
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from kinova_env import KinovaReachEnv
from policy_tools import (add_controller_arguments, controller_options,
                          expand_controller_policy, rescale_target_inputs,
                          resolve_core_arguments, core_options, save_checkpoint_atomic)


class ReachEvalCallback(BaseCallback):
    """Periodically measure reach success on a fixed validation seed set."""

    def __init__(
        self, eval_env, eval_freq: int, episodes: int, seed: int, log_path: Path,
        initial_action_scale: float, final_action_scale: float, curriculum_steps: int,
        target_frequency_hz: float, position_reward_weight: float,
        orientation_reward_weight: float,
        joint_target_speed_limit: float, joint_target_accel_limit: float,
        random_target_yaw: bool,
    ):
        super().__init__(verbose=0)
        self.eval_env = eval_env
        self.eval_freq = int(eval_freq)
        self.episodes = int(episodes)
        self.seed = int(seed)
        self.log_path = log_path
        self.initial_action_scale = float(initial_action_scale)
        self.final_action_scale = float(final_action_scale)
        self.curriculum_steps = int(curriculum_steps)
        self.target_frequency_hz = float(target_frequency_hz)
        self.position_reward_weight = float(position_reward_weight)
        self.orientation_reward_weight = float(orientation_reward_weight)
        self.joint_target_speed_limit = float(joint_target_speed_limit)
        self.joint_target_accel_limit = float(joint_target_accel_limit)
        self.random_target_yaw = bool(random_target_yaw)
        self.last_eval = 0
        self.best_score = None
        self.start_timesteps = 0

    def _on_training_start(self) -> None:
        self.start_timesteps = self.num_timesteps
        if not self.log_path.exists():
            with self.log_path.open("w", newline="", encoding="utf-8") as handle:
                csv.writer(handle).writerow((
                    "timesteps", "mean_return", "position_success_episodes",
                    "pose_success_episodes", "position_success_steps",
                    "pose_success_steps", "target_success_windows",
                    "total_target_windows", "target_success_rate",
                    "mean_target_success_steps", "mean_final_position_error",
                    "mean_final_orientation_error", "mean_position_error",
                    "mean_joint_speed", "mean_joint_command_speed",
                    "mean_joint_command_accel", "mean_action_change", "action_scale",
                    "target_frequency_hz", "position_reward_weight",
                    "orientation_reward_weight",
                    "joint_target_speed_limit", "joint_target_accel_limit",
                    "random_target_yaw",
                    "stable_target_windows", "stable_target_rate",
                ))
        # Score the loaded policy before updates so a worse fine-tune cannot
        # replace the user's strongest starting checkpoint.
        self._evaluate_policy()
        self.last_eval = self.num_timesteps

    def _action_scale_at_current_step(self) -> float:
        if self.curriculum_steps <= 0:
            return self.final_action_scale
        progress = np.clip(
            (self.num_timesteps - self.start_timesteps) / self.curriculum_steps, 0.0, 1.0
        )
        return self.initial_action_scale + progress * (
            self.final_action_scale - self.initial_action_scale
        )

    def _evaluate_policy(self) -> None:
        action_scale = self._action_scale_at_current_step()
        self.eval_env.set_action_scale(action_scale)
        self.model.reach_config["action_scale"] = action_scale

        episode_returns = []
        position_success_episodes = 0
        pose_success_episodes = 0
        position_success_steps = 0
        pose_success_steps = 0
        target_success_windows = 0
        stable_target_windows = 0
        total_target_windows = 0
        target_success_latencies = []
        total_steps = 0
        final_position_errors = []
        final_orientation_errors = []
        step_position_errors = []
        joint_speeds = []
        joint_command_speeds = []
        joint_command_accels = []
        action_changes = []

        for episode in range(self.episodes):
            observation, _ = self.eval_env.reset(seed=self.seed + episode)
            done = False
            episode_position_success = False
            episode_pose_success = False
            episode_return = 0.0
            episode_target_steps = {}
            episode_target_success_steps = {}
            streaks = {}
            longest_streaks = {}
            while not done:
                action, _ = self.model.predict(observation, deterministic=True)
                observation, reward, terminated, truncated, info = self.eval_env.step(action)
                done = terminated or truncated
                episode_return += float(reward)
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
                position_success_steps += int(info["is_success"])
                pose_success_steps += int(info["is_pose_success"])
                step_position_errors.append(float(info["position_error"]))
                joint_speeds.append(float(info["joint_speed"]))
                joint_command_speeds.append(float(info["joint_command_speed"]))
                joint_command_accels.append(float(info["joint_command_accel"]))
                action_changes.append(float(info["action_change"]))
                total_steps += 1
            episode_returns.append(episode_return)
            position_success_episodes += int(episode_position_success)
            pose_success_episodes += int(episode_pose_success)
            total_target_windows += len(episode_target_steps)
            target_success_windows += len(episode_target_success_steps)
            stable_target_windows += sum(
                count * self.eval_env.control_dt >= 0.5 for count in longest_streaks.values()
            )
            target_success_latencies.extend(episode_target_success_steps.values())
            final_position_errors.append(float(info["position_error"]))
            final_orientation_errors.append(float(info["orientation_error"]))

        position_rate = position_success_episodes / self.episodes
        pose_rate = pose_success_episodes / self.episodes
        target_success_rate = target_success_windows / total_target_windows
        stable_target_rate = stable_target_windows / total_target_windows
        mean_target_success_steps = (
            float(np.mean(target_success_latencies)) if target_success_latencies else 0.0
        )
        mean_final_position_error = float(np.mean(final_position_errors))
        mean_final_orientation_error = float(np.mean(final_orientation_errors))
        mean_position_error = float(np.mean(step_position_errors))
        mean_joint_speed = float(np.mean(joint_speeds))
        mean_joint_command_speed = float(np.mean(joint_command_speeds))
        mean_joint_command_accel = float(np.mean(joint_command_accels))
        mean_action_change = float(np.mean(action_changes))
        print(
            f"\nEval at {self.num_timesteps:,} steps: "
            f"return={np.mean(episode_returns):.3f}, "
            f"position-success episodes={position_rate:.1%}, "
            f"target-window success={target_success_windows}/{total_target_windows} "
            f"({target_success_rate:.1%}), "
            f"stable 0.5s={stable_target_rate:.1%}, "
            f"pose-success episodes={pose_rate:.1%}, "
            f"position-success steps={position_success_steps / total_steps:.1%}, "
            f"mean position error={mean_position_error:.3f} m, "
            f"joint speed={mean_joint_speed:.3f} rad/s, "
            f"command speed={mean_joint_command_speed:.3f} rad/s, "
            f"command accel={mean_joint_command_accel:.3f} rad/s^2, "
            f"action change={mean_action_change:.3f}"
        )
        with self.log_path.open("a", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerow((
                self.num_timesteps, float(np.mean(episode_returns)), position_rate,
                pose_rate, position_success_steps / total_steps,
                pose_success_steps / total_steps, target_success_windows,
                total_target_windows, target_success_rate, mean_target_success_steps,
                mean_final_position_error,
                mean_final_orientation_error, mean_position_error, mean_joint_speed,
                mean_joint_command_speed, mean_joint_command_accel, mean_action_change,
                action_scale, self.target_frequency_hz,
                self.position_reward_weight, self.orientation_reward_weight,
                self.joint_target_speed_limit, self.joint_target_accel_limit,
                self.random_target_yaw,
                stable_target_windows, stable_target_rate,
            ))

        score = (
            stable_target_rate,
            target_success_rate,
            position_success_steps / total_steps,
            position_rate,
            -mean_position_error,
            -mean_final_position_error,
            -mean_joint_speed,
        )
        if self.best_score is None or score > self.best_score:
            self.best_score = score
            save_checkpoint_atomic(self.model, self.log_path.parent / "best_model.zip")
            print(f"Saved new best policy: {self.log_path.parent / 'best_model.zip'}")

    def _on_step(self) -> bool:
        if self.num_timesteps - self.last_eval < self.eval_freq:
            return True
        self.last_eval = self.num_timesteps
        self._evaluate_policy()
        return True

    def _on_training_end(self) -> None:
        self.eval_env.close()


class ActionScaleCurriculumCallback(BaseCallback):
    """Gradually widen joint-target offsets while fine-tuning a compatible policy."""

    def __init__(self, initial_scale: float, final_scale: float, duration: int, update_freq: int = 10_000):
        super().__init__(verbose=0)
        self.initial_scale = float(initial_scale)
        self.final_scale = float(final_scale)
        self.duration = int(duration)
        self.update_freq = int(update_freq)
        self.start_timesteps = 0
        self.last_update = -self.update_freq

    def _on_training_start(self) -> None:
        self.start_timesteps = self.num_timesteps

    def _on_step(self) -> bool:
        elapsed = self.num_timesteps - self.start_timesteps
        if elapsed - self.last_update < self.update_freq and elapsed < self.duration:
            return True
        progress = np.clip(elapsed / max(self.duration, 1), 0.0, 1.0)
        scale = self.initial_scale + progress * (self.final_scale - self.initial_scale)
        self.training_env.env_method("set_action_scale", float(scale))
        self.model.reach_config["action_scale"] = float(scale)
        self.last_update = elapsed
        return True


def _make_env(
    rank: int,
    monitor_dir: Path,
    action_scale: float,
    target_frequency_hz: float,
    joint_target_speed_limit: float,
    joint_target_accel_limit: float,
    position_reward_weight: float,
    orientation_reward_weight: float,
    random_target_yaw: bool,
    extra_options: dict,
):
    env = KinovaReachEnv(
        observation_noise=0.01,
        action_scale=action_scale,
        target_frequency_hz=target_frequency_hz,
        joint_target_speed_limit=joint_target_speed_limit,
        joint_target_accel_limit=joint_target_accel_limit,
        position_reward_weight=position_reward_weight,
        orientation_reward_weight=orientation_reward_weight,
        random_target_yaw=random_target_yaw,
        **extra_options,
    )
    return Monitor(
        env,
        filename=str(monitor_dir / f"env_{rank}"),
        info_keywords=("position_error", "orientation_error", "is_success", "is_pose_success"),
    )


def main():
    parser = argparse.ArgumentParser(description="Train Kinova Gen3 Reach with CPU PPO")
    add_controller_arguments(parser)
    parser.add_argument("--exploration-std", type=float,
                        help="Reset Gaussian action standard deviation for a controlled fine-tune")
    parser.add_argument("--entropy-coef", type=float,
                        help="Override entropy regularization when fine-tuning")
    parser.add_argument("--steps", type=int, default=300_000,
                        help="Environment transitions to train (additional transitions when resuming)")
    parser.add_argument("--n-envs", type=int, default=4, help="CPU simulation workers (use 1 for lowest overhead)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=Path("runs/reach_cpu"))
    parser.add_argument("--save-freq", type=int, default=50_000, help="Checkpoint interval in transitions")
    parser.add_argument("--resume", type=Path, help="Continue from an SB3 PPO checkpoint")
    parser.add_argument("--eval-freq", type=int, default=100_000, help="Evaluation interval in transitions")
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--eval-seed", type=int, default=10_000)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3,
                        help="PPO learning rate; set lower when fine-tuning a checkpoint")
    parser.add_argument("--action-scale", type=float, default=None,
                        help="Final joint target offset scale in radians")
    parser.add_argument("--target-frequency-hz", type=float, default=None,
                        help="Random target changes per second; 0.25 means one new target every 4 seconds")
    parser.add_argument("--joint-target-speed-limit", type=float, default=None,
                        help="Maximum change rate for commanded joint positions in rad/s")
    parser.add_argument("--joint-target-accel-limit", type=float, default=None,
                        help="Maximum change rate for commanded joint velocity in rad/s^2")
    parser.add_argument("--orientation-reward-weight", type=float, default=None,
                        help="Orientation penalty weight; use 0 for a sphere position-reaching task")
    parser.add_argument("--random-target-yaw", action=argparse.BooleanOptionalAction, default=None,
                        help="Randomize target yaw; off by default because the ball has no orientation")
    parser.add_argument("--position-reward-weight", type=float, default=None,
                        help="Dense position error penalty weight; raise it to emphasize ball reaching")
    parser.add_argument("--initial-action-scale", type=float,
                        help="Starting action scale for a gradual action-scale curriculum")
    parser.add_argument("--curriculum-steps", type=int, default=0,
                        help="Transitions over which action scale moves from initial to final")
    args = parser.parse_args()
    if args.resume is not None and not args.resume.is_file():
        parser.error(f"Resume checkpoint does not exist: {args.resume}")
    torch.set_num_threads(1)
    source = PPO.load(args.resume, device="cpu") if args.resume is not None else None
    resolve_core_arguments(args, source)
    if args.exploration_std is not None and (
            not np.isfinite(args.exploration_std) or args.exploration_std <= 0):
        parser.error("exploration standard deviation must be finite and positive")
    if args.entropy_coef is not None and (
            not np.isfinite(args.entropy_coef) or args.entropy_coef < 0):
        parser.error("entropy coefficient must be finite and nonnegative")
    if (args.steps <= 0 or args.n_envs <= 0 or args.save_freq <= 0
            or args.eval_freq <= 0 or args.eval_episodes <= 0 or args.learning_rate <= 0
            or args.action_scale <= 0 or args.curriculum_steps < 0
            or not np.isfinite(args.target_frequency_hz) or args.target_frequency_hz <= 0
            or args.joint_target_speed_limit <= 0
            or args.joint_target_accel_limit <= 0
            or not np.isfinite(args.position_reward_weight) or args.position_reward_weight < 0
            or not np.isfinite(args.orientation_reward_weight) or args.orientation_reward_weight < 0
            or (args.initial_action_scale is not None and args.initial_action_scale <= 0)):
        parser.error("steps, env count, save/eval frequencies, episodes, action scales, target frequency and joint target speed must be positive; reward weights and curriculum steps cannot be negative")
    initial_action_scale = (
        args.action_scale if args.initial_action_scale is None else args.initial_action_scale
    )
    if initial_action_scale != args.action_scale and args.curriculum_steps <= 0:
        parser.error("--curriculum-steps must be positive when initial and final action scales differ")
    if args.resume is not None and not args.resume.is_file():
        parser.error(f"Resume checkpoint does not exist: {args.resume}")

    args.out.mkdir(parents=True, exist_ok=True)
    monitor_dir = args.out / "monitor"
    monitor_dir.mkdir(parents=True, exist_ok=True)

    # A small MLP is faster with a single BLAS/PyTorch thread per worker.
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    extra_options = controller_options(args, source)
    (args.out / "training_config.json").write_text(json.dumps({
        "arguments": vars(args), "controller_options": extra_options,
        "starting_timesteps": source.num_timesteps if source is not None else 0,
    }, indent=2, default=str), encoding="utf-8")

    env_fns = [partial(
        _make_env, i, monitor_dir, initial_action_scale, args.target_frequency_hz,
        args.joint_target_speed_limit, args.joint_target_accel_limit,
        args.position_reward_weight, args.orientation_reward_weight,
        args.random_target_yaw,
        extra_options,
    ) for i in range(args.n_envs)]
    if args.n_envs == 1:
        env = DummyVecEnv(env_fns)
    else:
        env = SubprocVecEnv(env_fns, start_method="spawn")

    checkpoint_callback = CheckpointCallback(
        save_freq=max(args.save_freq // args.n_envs, 1),
        save_path=str(args.out / "checkpoints"),
        name_prefix="ppo_gen3_reach",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )

    eval_callback = ReachEvalCallback(
        eval_env=KinovaReachEnv(
            action_scale=initial_action_scale,
            target_frequency_hz=args.target_frequency_hz,
            joint_target_speed_limit=args.joint_target_speed_limit,
            joint_target_accel_limit=args.joint_target_accel_limit,
            position_reward_weight=args.position_reward_weight,
            orientation_reward_weight=args.orientation_reward_weight,
            random_target_yaw=args.random_target_yaw,
            **extra_options,
        ),
        eval_freq=args.eval_freq,
        episodes=args.eval_episodes,
        seed=args.eval_seed,
        log_path=args.out / "evaluation.csv",
        initial_action_scale=initial_action_scale,
        final_action_scale=args.action_scale,
        curriculum_steps=args.curriculum_steps,
        target_frequency_hz=args.target_frequency_hz,
        position_reward_weight=args.position_reward_weight,
        orientation_reward_weight=args.orientation_reward_weight,
        joint_target_speed_limit=args.joint_target_speed_limit,
        joint_target_accel_limit=args.joint_target_accel_limit,
        random_target_yaw=args.random_target_yaw,
    )
    if args.resume is not None:
        if source.observation_space.shape != env.observation_space.shape:
            model = expand_controller_policy(source, env, args.learning_rate)
            print("Expanded pretrained observations from 28 to 42; retained policy weights")
        else:
            model = source
            model.set_env(env)
        model.learning_rate = args.learning_rate
        model.lr_schedule = ConstantSchedule(args.learning_rate)
        for parameter_group in model.policy.optimizer.param_groups:
            parameter_group["lr"] = args.learning_rate
        print(f"Resuming from {args.resume}; current transition count: {model.num_timesteps:,}")
        print(f"Fine-tuning learning rate: {args.learning_rate:g}")
        if args.curriculum_steps > 0:
            print(
                f"Action-scale curriculum: {initial_action_scale:g} -> {args.action_scale:g} rad "
                f"over {args.curriculum_steps:,} transitions"
            )
    else:
        model = PPO(
            "MlpPolicy",
            env,
            device="cpu",
            verbose=1,
            seed=args.seed,
            n_steps=512,
            batch_size=256,
            n_epochs=8,
            learning_rate=args.learning_rate,
            gamma=0.99,
            gae_lambda=0.95,
            ent_coef=0.001,
            clip_range=0.2,
            target_kl=0.01,
            vf_coef=1.0,
            max_grad_norm=1.0,
            policy_kwargs={
                "net_arch": {"pi": [64, 64], "vf": [64, 64]},
                "activation_fn": nn.ELU,
                "log_std_init": 0.0,
            },
        )
    if source is not None:
        rescale_target_inputs(model, getattr(source, "reach_config", {}).get("target_delta_scale", 1.0),
                              extra_options["target_delta_scale"])
    model.reach_config = {**core_options(args), **extra_options, "action_scale": initial_action_scale}
    if args.entropy_coef is not None:
        model.ent_coef = args.entropy_coef
    if args.exploration_std is not None:
        with torch.no_grad():
            model.policy.log_std.fill_(np.log(args.exploration_std))
        model.policy.optimizer.state.pop(model.policy.log_std, None)
    try:
        model.learn(
            total_timesteps=args.steps,
            callback=CallbackList([
                ActionScaleCurriculumCallback(
                    initial_action_scale, args.action_scale, args.curriculum_steps
                ),
                checkpoint_callback,
                eval_callback,
            ]),
            reset_num_timesteps=args.resume is None,
            progress_bar=False,
            log_interval=10,
        )
        model.save(str(args.out / "ppo_gen3_reach"))
        print(f"Saved final policy: {args.out / 'ppo_gen3_reach.zip'}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
