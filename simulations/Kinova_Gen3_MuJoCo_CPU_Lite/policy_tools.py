"""Checkpoint-compatible controller observations and simulator settings."""
import argparse
from pathlib import Path
import time
import zipfile

import torch
from stable_baselines3 import PPO

CORE_ENV_DEFAULTS = {
    "action_scale": 1.5, "target_frequency_hz": 0.25,
    "joint_target_speed_limit": 0.8, "joint_target_accel_limit": 2.0,
    "position_reward_weight": 0.8, "orientation_reward_weight": 0.0,
    "random_target_yaw": False,
}


def resolve_core_arguments(args, policy=None):
    saved = getattr(policy, "reach_config", {}) if policy is not None else {}
    for name, default in CORE_ENV_DEFAULTS.items():
        if getattr(args, name) is None:
            setattr(args, name, saved.get(name, default))


def core_options(args):
    return {name: getattr(args, name) for name in CORE_ENV_DEFAULTS}


def add_controller_arguments(parser):
    parser.add_argument("--damping-multiplier", type=float, default=None,
                        help="Actuator damping scale; defaults to the checkpoint setting")
    parser.add_argument("--settling-reward-weight", type=float, default=None,
                        help="Reward for remaining close to the ball with low end-effector speed")
    parser.add_argument("--controller-observation", action=argparse.BooleanOptionalAction,
                        default=None, help="Include filtered joint targets and velocities in the state")
    parser.add_argument("--physical-obstacles", action=argparse.BooleanOptionalAction,
                        default=None, help="Enable solid table/ball and reach a hovering point with a gap")
    parser.add_argument("--ball-clearance", type=float, default=None,
                        help="Desired tip-to-ball surface gap in metres; default 0.02")
    parser.add_argument("--target-delta-scale", type=float, default=None,
                        help="Observation scale for the Cartesian target error")
    parser.add_argument("--approach-axis", choices=("above", "front"), default=None,
                        help="Safe approach direction; defaults to the checkpoint setting")
    parser.add_argument("--bias-compensation", action=argparse.BooleanOptionalAction,
                        default=None, help="Compensate gravity/dynamic bias inside actuator force limits")
    parser.add_argument("--velocity-feedforward", action=argparse.BooleanOptionalAction,
                        default=None, help="Track the filtered joint-reference velocity")
    parser.add_argument("--tracking-error-limit", type=float, default=None,
                        help="Bound reference backlog relative to actual joints, in radians")


def controller_options(args, policy=None):
    saved = getattr(policy, "reach_config", {}) if policy is not None else {}
    uses_controller_state = (
        policy.observation_space.shape == (42,) if policy is not None
        else bool(saved.get("controller_observation", False))
    )
    options = {
        "damping_multiplier": saved.get("damping_multiplier", 1.0),
        "settling_reward_weight": saved.get("settling_reward_weight", 0.0),
        "controller_observation": uses_controller_state,
        "physical_obstacles": saved.get("physical_obstacles", False),
        "ball_clearance": saved.get("ball_clearance", 0.02),
        "target_delta_scale": saved.get("target_delta_scale", 1.0),
        "approach_axis": saved.get("approach_axis", "above"),
        "bias_compensation": saved.get("bias_compensation", False),
        "velocity_feedforward": saved.get("velocity_feedforward", False),
        "tracking_error_limit": saved.get("tracking_error_limit", None),
    }
    for key in options:
        value = getattr(args, key)
        if value is not None:
            options[key] = value
    return options


def expand_controller_policy(source, env, learning_rate):
    """Append 14 inputs with zero weights, retaining the pretrained policy output."""
    if source.observation_space.shape != (28,) or env.observation_space.shape != (42,):
        raise ValueError("Only the 28-to-42 controller-state migration is supported")
    model = PPO(
        "MlpPolicy", env, device="cpu", verbose=source.verbose, seed=source.seed,
        n_steps=source.n_steps, batch_size=source.batch_size, n_epochs=source.n_epochs,
        learning_rate=learning_rate, gamma=source.gamma, gae_lambda=source.gae_lambda,
        ent_coef=source.ent_coef, clip_range=source.clip_range,
        target_kl=source.target_kl, vf_coef=source.vf_coef,
        max_grad_norm=source.max_grad_norm, policy_kwargs=source.policy_kwargs,
    )
    weights = source.policy.state_dict()
    for name in ("mlp_extractor.policy_net.0.weight", "mlp_extractor.value_net.0.weight"):
        weight = weights[name]
        weights[name] = torch.cat((weight, weight.new_zeros((weight.shape[0], 14))), dim=1)
    model.policy.load_state_dict(weights)
    model.num_timesteps = source.num_timesteps
    model._n_updates = source._n_updates
    # Optimizer moments for the expanded tensors cannot be reused safely.
    return model


def rescale_target_inputs(model, old_scale, new_scale):
    """Change observation units without changing the pretrained actor/critic output."""
    if old_scale == new_scale:
        return
    for network in (model.policy.mlp_extractor.policy_net, model.policy.mlp_extractor.value_net):
        weight = network[0].weight
        with torch.no_grad():
            weight[:, 17:20] *= old_scale / new_scale
        model.policy.optimizer.state.pop(weight, None)


def save_checkpoint_atomic(model, path):
    """Publish complete checkpoints so a live viewer cannot read half a ZIP."""
    destination = Path(path).with_suffix(".zip")
    pending = destination.with_name(destination.stem + ".pending.zip")
    model.save(str(pending))
    for attempt in range(3):
        try:
            pending.replace(destination)
            return
        except PermissionError:
            if attempt == 2:
                raise
            time.sleep(0.05)


def load_updated_checkpoint(path, previous_stamp):
    """Defer a checkpoint that is still being written by an older trainer."""
    path = Path(path)
    stamp = path.stat().st_mtime_ns
    if stamp == previous_stamp or time.time_ns() - stamp < 1_000_000_000:
        return None
    try:
        candidate = PPO.load(path, device="cpu")
    except (OSError, EOFError, zipfile.BadZipFile) as error:
        print(f"Checkpoint update in progress; retrying next target: {error}", flush=True)
        return None
    if path.stat().st_mtime_ns != stamp:
        return None
    return candidate, stamp
