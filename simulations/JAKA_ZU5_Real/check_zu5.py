"""Independent URDF FK checks and a simulated IK position-controller baseline."""
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from zu5_env import CONFIG, HOME, ROOT, Zu5ReachEnv


def urdf_fk(q):
    """URDF fixed-axis RPY and Rodrigues joint rotation, independent of MJCF."""
    robot = ET.parse(ROOT/"assets"/"jaka_zu5.urdf").getroot()
    transform = np.eye(4)
    index = 0
    for joint in robot.findall("joint"):
        origin = joint.find("origin")
        r, p, y = np.fromstring(origin.get("rpy"), sep=" ")
        rx = np.array([[1, 0, 0], [0, np.cos(r), -np.sin(r)], [0, np.sin(r), np.cos(r)]])
        ry = np.array([[np.cos(p), 0, np.sin(p)], [0, 1, 0], [-np.sin(p), 0, np.cos(p)]])
        rz = np.array([[np.cos(y), -np.sin(y), 0], [np.sin(y), np.cos(y), 0], [0, 0, 1]])
        local = np.eye(4)
        local[:3, :3] = rz @ ry @ rx
        local[:3, 3] = np.fromstring(origin.get("xyz"), sep=" ")
        transform = transform @ local
        if joint.get("type") == "revolute":
            axis = np.fromstring(joint.find("axis").get("xyz"), sep=" ")
            cross = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
            rotation = np.eye(4)
            rotation[:3, :3] = np.eye(3)+np.sin(q[index])*cross+(1-np.cos(q[index]))*(cross@cross)
            transform = transform @ rotation
            index += 1
    return transform


def solve_ik(env):
    data = mujoco.MjData(env.model)
    data.qpos[:] = HOME
    jac = np.zeros((3, 6))
    for _ in range(100):
        mujoco.mj_forward(env.model, data)
        error = env.target-data.site_xpos[env.site]
        if np.linalg.norm(error) < 0.001:
            return data.qpos.copy()
        mujoco.mj_jacSite(env.model, data, jac, None, env.site)
        delta = jac.T @ np.linalg.solve(jac@jac.T+0.001*np.eye(3), error)
        data.qpos[:] = np.clip(data.qpos+np.clip(delta, -0.1, 0.1), HOME-0.75, HOME+0.75)
    raise RuntimeError("Baseline IK did not converge")


def main():
    env = Zu5ReachEnv()
    rng = np.random.default_rng(20261016)
    errors = []
    for _ in range(100):
        env.data.qpos[:] = HOME + rng.uniform(-0.5, 0.5, 6)
        mujoco.mj_forward(env.model, env.data)
        expected = urdf_fk(env.data.qpos)
        errors.append(float(np.max(np.abs(expected[:3, 3]-env.position()))))
        np.testing.assert_allclose(env.data.site_xmat[env.site].reshape(3, 3), expected[:3, :3], atol=1e-12)
    assert max(errors) < 1e-12, errors
    reached, held, contact_steps, max_speed = 0, 0, 0, 0.
    for episode in range(20):
        env.reset(seed=80000+episode)
        q = solve_ik(env)
        action = (q-HOME)/CONFIG["action_scale_rad"]
        streak = longest = 0
        hit = False
        for _ in range(CONFIG["episode_steps"]):
            _, _, terminated, _, info = env.step(action)
            assert not terminated
            hit |= info["is_success"]
            streak = streak+1 if info["is_success"] else 0
            longest = max(longest, streak)
            contact_steps += bool(info["contacts"])
            max_speed = max(max_speed, info["max_joint_speed_rad_s"])
        reached += int(hit)
        held += int(longest*CONFIG["control_dt"] >= 0.5)
    report = {"fk_samples": 100, "max_fk_position_difference_m": max(errors),
              "baseline": "Damped least-squares IK + same simulated position controller",
              "episodes": 20, "seed": 80000, "reached": reached, "held_0_5s": held,
              "contact_steps": contact_steps, "max_joint_speed_rad_s": max_speed,
              "boundary": "URDF consistency and simulation only; not physical FK calibration"}
    path = ROOT/"reports"
    path.mkdir(exist_ok=True)
    (path/"model_check.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    assert reached == 20 and held == 20 and contact_steps == 0, "Baseline failed"
    env.reset(seed=80001)
    with mujoco.Renderer(env.model, height=720, width=960) as renderer:
        renderer.update_scene(env.data, camera="overview")
        from PIL import Image
        Image.fromarray(renderer.render()).save(path/"zu5_scene.png")


if __name__ == "__main__":
    main()
