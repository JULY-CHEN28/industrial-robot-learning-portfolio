"""Continuous goal switching with preserved robot/controller state and fixed schedules."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from zu5_env import CONFIG, Zu5ReachEnv, load_policy


def make_schedule(seed, windows):
    # A separate sampler ensures targets are identical across policies, independent
    # of whether the evaluated arm has reached the preceding goal.
    sampler = Zu5ReachEnv()
    sampler.reset(seed=seed)
    goals = [sampler.target.copy()]
    for _ in range(windows-1):
        for _ in range(100):
            sampler.sample_target()
            if np.linalg.norm(sampler.target-goals[-1]) >= 0.05:
                goals.append(sampler.target.copy())
                break
        else:
            raise RuntimeError("Cannot sample distinct consecutive targets")
    return np.asarray(goals)


class ContinuousReachEnv(Zu5ReachEnv):
    def __init__(self, windows=15, window_steps=125):
        if not isinstance(windows, int) or not isinstance(window_steps, int) or min(windows, window_steps) < 1:
            raise ValueError("windows and window_steps must be positive integers")
        super().__init__()
        self.windows, self.window_steps = windows, window_steps
        self.max_steps = windows*window_steps
        self.goals = None
        self.target_index = 0

    def reset(self, *, seed=None, options=None):
        if seed is None:
            raise ValueError("An explicit seed is required for a reproducible continuous schedule")
        super().reset(seed=seed, options=options)
        self.goals = make_schedule(seed, self.windows)
        self.target_index = 0
        obs = self.set_target(self.goals[0])
        return obs, self.info()

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        # Metrics/reward belong to the goal just executed. The next observation
        # contains the new goal so the next policy action never uses a stale goal.
        info["target_index"] = self.target_index
        if not terminated and not truncated and self.steps % self.window_steps == 0:
            self.target_index += 1
            obs = self.set_target(self.goals[self.target_index])
        return obs, reward, terminated, truncated, info


def evaluate_continuous(model, sequences, windows, window_steps, seed, env_factory=None):
    factory = ContinuousReachEnv if env_factory is None else env_factory
    env = factory(windows, window_steps)
    records = []
    warnings = np.zeros(len(env.data.warning), dtype=np.int64)
    contact_physics_steps = total_physics_steps = 0
    max_speed = max_depth = max_reference_speed = max_reference_accel = 0.
    goals_by_sequence = []
    for sequence in range(sequences):
        obs, _ = env.reset(seed=seed+sequence)
        goals_by_sequence.append(env.goals.tolist())
        current = {"sequence": sequence, "target_index": 0, "reached": False,
                   "held_0_5s": False, "first_reach_s": None, "longest_success_steps": 0}
        streak = age = 0
        for _ in range(env.max_steps):
            previous_velocity = env.command_v.copy()
            obs, _, terminated, truncated, info = env.step(model.predict(obs, deterministic=True)[0])
            if terminated:
                raise RuntimeError("Nonfinite continuous simulation state")
            age += 1
            if info["is_success"] and not current["reached"]:
                current["first_reach_s"] = age*CONFIG["control_dt"]
            current["reached"] |= info["is_success"]
            streak = streak+1 if info["is_success"] else 0
            current["longest_success_steps"] = max(current["longest_success_steps"], streak)
            current["held_0_5s"] |= streak*CONFIG["control_dt"] >= 0.5
            contact_physics_steps += info["physics_contact_steps"]
            total_physics_steps += 16
            max_speed = max(max_speed, info["physics_peak_joint_speed_rad_s"])
            max_depth = max(max_depth, info["physics_peak_penetration_m"])
            max_reference_speed = max(max_reference_speed, float(np.max(np.abs(env.command_v))))
            max_reference_accel = max(max_reference_accel, float(np.max(np.abs(env.command_v-previous_velocity)))/CONFIG["control_dt"])
            if age == window_steps:
                current["final_error_m"] = info["position_error_m"]
                records.append(current)
                current = {"sequence": sequence, "target_index": info["target_index"]+1,
                           "reached": False, "held_0_5s": False, "first_reach_s": None,
                           "longest_success_steps": 0}
                streak = age = 0
            if truncated:
                break
        warnings += [int(w.number) for w in env.data.warning]
    assert len(records) == sequences*windows
    latencies = [row["first_reach_s"] for row in records if row["reached"]]
    return {
        "sequences": sequences, "windows_per_sequence": windows,
        "window_seconds": window_steps*CONFIG["control_dt"],
        "sequence_seconds": windows*window_steps*CONFIG["control_dt"], "seed": seed,
        "targets": len(records), "reached": sum(row["reached"] for row in records),
        "held_0_5s": sum(row["held_0_5s"] for row in records),
        "mean_final_error_m": float(np.mean([row["final_error_m"] for row in records])),
        "mean_first_reach_s": float(np.mean(latencies)) if latencies else None,
        "physics_contact_steps": contact_physics_steps, "total_physics_steps": total_physics_steps,
        "max_physics_joint_speed_rad_s": max_speed, "max_physics_penetration_m": max_depth,
        "max_reference_speed_rad_s": max_reference_speed, "max_reference_accel_rad_s2": max_reference_accel,
        "mujoco_warning_counts": warnings.tolist(), "config": CONFIG,
        "schedule_contract": "v1; separate seeded sampler; consecutive goals >=0.05m apart; no resets inside sequences",
        "goals_m": goals_by_sequence, "per_target": records,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--sequences", type=int, default=20)
    parser.add_argument("--windows", type=int, default=15)
    parser.add_argument("--window-steps", type=int, default=125)
    parser.add_argument("--seed", type=int, default=510000)
    args = parser.parse_args()
    if min(args.sequences, args.windows, args.window_steps) <= 0:
        parser.error("sequence/window counts must be positive")
    if args.out.exists():
        parser.error("output already exists; choose a new report path")
    torch.set_num_threads(2)
    model = load_policy(args.model, Zu5ReachEnv())
    report = evaluate_continuous(model, args.sequences, args.windows, args.window_steps, args.seed)
    report["checkpoint_sha256"] = hashlib.sha256(args.model.read_bytes()).hexdigest()
    report["trained_steps"] = model.num_timesteps
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k not in ("goals_m", "per_target", "config")}, indent=2))


if __name__ == "__main__":
    main()
