"""Exploratory sensor, delay and model-mismatch tests, not a hardware calibration."""
import argparse
from collections import deque
from dataclasses import asdict, dataclass
from functools import partial
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
import torch

from continuous_zu5 import ContinuousReachEnv, evaluate_continuous
from zu5_env import CONFIG, HOME, Zu5ReachEnv, load_policy


@dataclass(frozen=True)
class Disturbance:
    joint_noise_rad: float = 0.
    velocity_noise_rad_s: float = 0.
    tcp_noise_m: float = 0.
    target_bias_x_m: float = 0.
    observation_delay_steps: int = 0
    command_delay_steps: int = 0
    mass_scale: float = 1.
    stiffness_scale: float = 1.
    damping_scale: float = 1.
    bias_scale: float = 1.

    def __post_init__(self):
        for name, value in asdict(self).items():
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite")
            if name.endswith("_steps") and (type(value) is not int or value < 0):
                raise ValueError(f"{name} must be a nonnegative integer")
            if ("noise" in name or name == "bias_scale") and value < 0:
                raise ValueError(f"{name} must be nonnegative")
            if name in ("mass_scale", "stiffness_scale", "damping_scale") and value <= 0:
                raise ValueError(f"{name} must be positive")


CASES = {
    "nominal": Disturbance(),
    "sensor_mild": Disturbance(joint_noise_rad=0.002, velocity_noise_rad_s=0.02, tcp_noise_m=0.001),
    "sensor_stress": Disturbance(joint_noise_rad=0.01, velocity_noise_rad_s=0.05, tcp_noise_m=0.005),
    "target_bias_10mm": Disturbance(target_bias_x_m=0.01),
    "target_bias_30mm": Disturbance(target_bias_x_m=0.03),
    "observation_delay_32ms": Disturbance(observation_delay_steps=1),
    "observation_delay_96ms": Disturbance(observation_delay_steps=3),
    "command_delay_32ms": Disturbance(command_delay_steps=1),
    "command_delay_96ms": Disturbance(command_delay_steps=3),
    "mass_plus20pct": Disturbance(mass_scale=1.2),
    "gains_minus20pct": Disturbance(stiffness_scale=0.8, damping_scale=0.8),
    "bias_minus20pct": Disturbance(bias_scale=0.8),
    "combined_mild": Disturbance(joint_noise_rad=0.002, velocity_noise_rad_s=0.02, tcp_noise_m=0.001,
                                  observation_delay_steps=1, command_delay_steps=1, mass_scale=1.2,
                                  stiffness_scale=0.8, damping_scale=1.2),
    "combined_stress": Disturbance(joint_noise_rad=0.01, velocity_noise_rad_s=0.05, tcp_noise_m=0.005,
                                    target_bias_x_m=0.01, observation_delay_steps=3, command_delay_steps=3,
                                    mass_scale=1.2, stiffness_scale=0.7, damping_scale=0.7),
}


class DisturbedReachEnv(ContinuousReachEnv):
    def __init__(self, windows, window_steps, disturbance):
        super().__init__(windows, window_steps)
        self.disturbance = disturbance
        # The plant may change mass, while compensation still uses nominal inertia.
        self.nominal = Zu5ReachEnv() if disturbance.mass_scale != 1 else None
        if self.nominal is not None:
            self.model.body_mass[1:] *= disturbance.mass_scale
            self.model.body_inertia[1:] *= disturbance.mass_scale
            mujoco.mj_setConst(self.model, self.data)
        self.model.actuator_gainprm[:, 0] *= disturbance.stiffness_scale
        self.model.actuator_biasprm[:, 1] *= disturbance.stiffness_scale
        self.model.actuator_biasprm[:, 2] *= disturbance.damping_scale

    def bias_force(self):
        if self.nominal is None:
            force = self.data.qfrc_bias
        else:
            self.nominal.data.qpos[:] = self.data.qpos
            self.nominal.data.qvel[:] = self.data.qvel
            mujoco.mj_forward(self.nominal.model, self.nominal.data)
            force = self.nominal.data.qfrc_bias
        return self.disturbance.bias_scale*force

    def observed(self, clean):
        self.observation_queue.append(clean.copy())
        delayed = self.observation_queue[0]
        observed = clean.copy()
        d = self.disturbance
        # Sensor packets are delayed; goal and controller-owned history remain current.
        observed[:6] = delayed[:6] + self.sensor_rng.normal(0, d.joint_noise_rad, 6)
        observed[6:12] = delayed[6:12] + self.sensor_rng.normal(0, d.velocity_noise_rad_s, 6)
        observed[12:15] = delayed[12:15] + self.sensor_rng.normal(0, d.tcp_noise_m, 3)
        # Derive both Cartesian channels from the same corrupted TCP measurement.
        # Preserve original floating-point values in the exact nominal case.
        if d.observation_delay_steps or d.tcp_noise_m or d.target_bias_x_m:
            observed[15:18] = (self.target+np.array([d.target_bias_x_m, 0, 0])-observed[12:15])*5
        return observed

    def reset(self, *, seed=None, options=None):
        clean, info = super().reset(seed=seed, options=options)
        self.sensor_rng = np.random.default_rng(np.random.SeedSequence([seed, 928173]))
        self.observation_queue = deque([clean.copy()]*(self.disturbance.observation_delay_steps+1),
                                       maxlen=self.disturbance.observation_delay_steps+1)
        hold = (self.data.qpos-HOME)/CONFIG["action_scale_rad"]
        self.command_queue = deque([hold.copy() for _ in range(self.disturbance.command_delay_steps)])
        return self.observed(clean), info

    def step(self, action):
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (6,) or not np.isfinite(action).all():
            raise ValueError("Action must contain six finite values")
        self.command_queue.append(action.copy())
        applied = self.command_queue.popleft()
        clean, reward, terminated, truncated, info = super().step(applied)
        return self.observed(clean), reward, terminated, truncated, info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--sequences", type=int, default=2)
    parser.add_argument("--windows", type=int, default=15)
    parser.add_argument("--seed", type=int, default=710000)
    parser.add_argument("--cases", nargs="+", choices=list(CASES), default=list(CASES))
    args = parser.parse_args()
    if min(args.sequences, args.windows) <= 0:
        parser.error("sequences and windows must be positive")
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    model = load_policy(args.model, Zu5ReachEnv())
    sha = hashlib.sha256(args.model.read_bytes()).hexdigest()
    rows = []
    for name in args.cases:
        disturbance = CASES[name]
        result = evaluate_continuous(model, args.sequences, args.windows, 125, args.seed,
                                     partial(DisturbedReachEnv, disturbance=disturbance))
        result.update(case=name, disturbance=asdict(disturbance), checkpoint_sha256=sha,
                      scope="Exploratory assumptions, not measured hardware error bounds")
        (args.out/f"{name}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        row = {k: v for k, v in result.items() if k not in ("goals_m", "per_target", "config")}
        rows.append(row)
        (args.out/"summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print(f"{name}: reach {result['reached']}/{result['targets']}, hold {result['held_0_5s']}/{result['targets']}, final {result['mean_final_error_m']*1000:.2f}mm, peak {result['max_physics_joint_speed_rad_s']:.3f}rad/s, contact_substeps {result['physics_contact_steps']}", flush=True)


if __name__ == "__main__":
    main()
