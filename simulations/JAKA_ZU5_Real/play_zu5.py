"""View a ZU5 checkpoint in MuJoCo; optionally render a GIF without a window."""
import argparse
from pathlib import Path
import time

import mujoco
from zu5_env import Zu5ReachEnv, load_policy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=410000)
    parser.add_argument("--gif", type=Path, help="Render one episode (continuous horizon when requested)")
    parser.add_argument("--continuous", action="store_true")
    parser.add_argument("--windows", type=int, default=15)
    parser.add_argument("--window-steps", type=int, default=125)
    args = parser.parse_args()
    if args.continuous:
        from continuous_zu5 import ContinuousReachEnv
        env = ContinuousReachEnv(args.windows, args.window_steps)
    else:
        env = Zu5ReachEnv()
    model = load_policy(args.model, env)
    obs, _ = env.reset(seed=args.seed)
    if args.gif:
        from PIL import Image
        frames = []
        with mujoco.Renderer(env.model, height=480, width=640) as renderer:
            for step in range(env.max_steps):
                obs, _, terminated, _, _ = env.step(model.predict(obs, deterministic=True)[0])
                if terminated:
                    raise RuntimeError("Invalid simulation state")
                if step % 2 == 0:
                    renderer.update_scene(env.data, camera="overview")
                    frames.append(Image.fromarray(renderer.render()).convert("P", palette=Image.Palette.ADAPTIVE))
        args.gif.parent.mkdir(parents=True, exist_ok=True)
        frames[0].save(args.gif, save_all=True, append_images=frames[1:], duration=64, loop=0)
        print(args.gif.resolve())
        return
    from mujoco import viewer as mujoco_viewer
    episode = 0
    with mujoco_viewer.launch_passive(env.model, env.data) as viewer:
        viewer.cam.lookat[:] = [0.35, 0, 0.35]
        viewer.cam.distance = 1.8
        viewer.cam.azimuth = 135
        viewer.cam.elevation = -25
        while viewer.is_running():
            start = time.perf_counter()
            obs, _, terminated, truncated, _ = env.step(model.predict(obs, deterministic=True)[0])
            if terminated:
                raise RuntimeError("Invalid simulation state")
            if truncated:
                episode += 1
                obs, _ = env.reset(seed=args.seed+episode)
            viewer.sync()
            time.sleep(max(0, 0.032-(time.perf_counter()-start)))


if __name__ == "__main__":
    main()
