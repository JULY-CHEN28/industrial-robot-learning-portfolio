"""CPU PPO training and reproducible evaluation for the provisional ZU5 task."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from zu5_env import CONFIG, ROOT, Zu5ReachEnv, load_policy


def evaluate(env, model, episodes, seed, random=False):
    hits, holds, collisions, steps = 0, 0, 0, 0
    errors, final_errors = [], []
    max_speed, max_depth = 0., 0.
    latencies = []
    warnings = np.zeros(len(env.data.warning), dtype=np.int64)
    rng = np.random.default_rng(seed)
    for episode in range(episodes):
        obs, _ = env.reset(seed=seed+episode)
        streak, longest, reached = 0, 0, False
        for step in range(CONFIG["episode_steps"]):
            action = rng.uniform(-1, 1, 6) if random else model.predict(obs, deterministic=True)[0]
            obs, _, terminated, truncated, info = env.step(action)
            if terminated:
                raise RuntimeError("Nonfinite simulation state during evaluation")
            if info["is_success"] and not reached:
                latencies.append((step+1)*CONFIG["control_dt"])
            reached |= info["is_success"]
            streak = streak+1 if info["is_success"] else 0
            longest = max(longest, streak)
            collisions += int(info["contacts"] > 0)
            errors.append(info["position_error_m"])
            max_speed = max(max_speed, info["max_joint_speed_rad_s"])
            max_depth = max(max_depth, info["max_penetration_m"])
            steps += 1
            if truncated:
                break
        hits += int(reached)
        holds += int(longest*CONFIG["control_dt"] >= 0.5)
        final_errors.append(info["position_error_m"])
        warnings += [int(w.number) for w in env.data.warning]
    return {"episodes": episodes, "seed": seed, "reached": hits, "held_0_5s": holds,
            "reach_rate": hits/episodes, "hold_rate": holds/episodes,
            "mean_error_m": float(np.mean(errors)), "mean_final_error_m": float(np.mean(final_errors)),
            "mean_first_reach_s": float(np.mean(latencies)) if latencies else None,
            "collision_step_fraction": collisions/steps,
            "max_joint_speed_rad_s": max_speed, "max_penetration_m": max_depth,
            "mujoco_warning_counts": warnings.tolist(),
            "config": CONFIG}


class Validation(BaseCallback):
    def __init__(self, out, frequency):
        super().__init__()
        self.out, self.frequency = out, frequency
        self.env = Zu5ReachEnv()
        self.best = None
        self.last = 0

    def _on_step(self):
        if self.num_timesteps-self.last < self.frequency:
            return True
        self.last = self.num_timesteps
        metrics = evaluate(self.env, self.model, 12, 170000)
        metrics["timesteps"] = self.num_timesteps
        with (self.out/"validation.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(metrics)+"\n")
        score = (metrics["hold_rate"], metrics["reach_rate"], -metrics["mean_final_error_m"])
        if self.best is None or score > self.best:
            self.best = score
            self.model.save(self.out/"best_model.zip")
        print(f"Validation {self.num_timesteps}: reach={metrics['reach_rate']:.1%}, hold={metrics['hold_rate']:.1%}, final_error={metrics['mean_final_error_m']:.4f}m", flush=True)
        return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=300000)
    parser.add_argument("--out", type=Path, default=ROOT/"runs"/"zu5_first")
    parser.add_argument("--eval-model", type=Path)
    parser.add_argument("--random-baseline", action="store_true")
    parser.add_argument("--episodes", type=int, default=60)
    parser.add_argument("--seed", type=int, default=410000)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    if args.steps <= 0 or args.episodes <= 0:
        parser.error("steps and episodes must be positive")
    torch.set_num_threads(2)
    if args.eval_model and args.random_baseline:
        parser.error("choose either a checkpoint or the random baseline")
    if args.eval_model or args.random_baseline:
        args.out.mkdir(parents=True, exist_ok=True)
        env = Zu5ReachEnv()
        model = load_policy(args.eval_model, env) if args.eval_model else None
        metrics = evaluate(env, model, args.episodes, args.seed, random=args.random_baseline)
        if model is not None:
            metrics["checkpoint_sha256"] = hashlib.sha256(args.eval_model.read_bytes()).hexdigest()
            metrics["trained_steps"] = model.num_timesteps
        else:
            metrics["baseline"] = "uniform random normalized joint position actions"
        prefix = "random" if args.random_baseline else "evaluation"
        (args.out/f"{prefix}_{args.seed}.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print(json.dumps(metrics, indent=2))
        return
    # Refuse accidental overwrite of an earlier run.
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out/"config.json").write_text(json.dumps(CONFIG, indent=2), encoding="utf-8")
    envs = DummyVecEnv([lambda: Monitor(Zu5ReachEnv()) for _ in range(4)])
    if args.resume:
        model = load_policy(args.resume, envs)
        model.set_env(envs)
    else:
        model = PPO("MlpPolicy", envs, n_steps=1024, batch_size=256, learning_rate=3e-4,
                    policy_kwargs={"net_arch": dict(pi=[64, 64], vf=[64, 64]), "activation_fn": torch.nn.ELU},
                    target_kl=0.02, ent_coef=0.001, seed=42, device="cpu", verbose=0)
    model.zu5_config = CONFIG
    start = time.perf_counter()
    model.learn(args.steps, callback=Validation(args.out, 50000), reset_num_timesteps=False)
    model.save(args.out/"final_model.zip")
    print(f"Finished {model.num_timesteps} cumulative steps in {time.perf_counter()-start:.1f}s", flush=True)


if __name__ == "__main__":
    main()
