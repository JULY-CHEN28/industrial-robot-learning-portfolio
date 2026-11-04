"""Provisional six-axis ZU5 position-reaching task; no hardware I/O."""
from pathlib import Path
import hashlib

import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parent
HOME = np.array([0.0, 0.5, 1.2, 0.0, 1.57, 0.0])
CONFIG = {
    "scene_sha256": hashlib.sha256((ROOT/"assets"/"scene.xml").read_bytes()).hexdigest(),
    "robot": "jaka_zu5", "contract_version": 1, "observation_size": 36,
    "action_size": 6, "control_dt": 0.032, "physics_dt": 0.002,
    "action_scale_rad": 0.8, "reference_speed_rad_s": 0.5,
    "reference_accel_rad_s2": 1.0, "tracking_error_rad": 0.15,
    "episode_steps": 250, "position_tolerance_m": 0.025,
    "target_joint_sampling_half_width_rad": [0.3, 0.2, 0.3, 0.25, 0.2, 0.25],
    "home_rad": HOME.tolist(), "tcp": "Link6 origin; no tool",
    "target": "non-colliding point marker; one target per episode",
    "controller": "provisional PD + bias compensation and velocity feedforward under shared torque clamp",
}


class Zu5ReachEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self):
        path = ROOT / "assets"
        self.model = mujoco.MjModel.from_xml_string(
            (path / "scene.xml").read_text(encoding="utf-8"),
            {f.name: f.read_bytes() for f in (path / "meshes").glob("*.STL")},
        )
        if (self.model.nq, self.model.nv, self.model.nu) != (6, 6, 6):
            raise ValueError("Expected a six-axis ZU5")
        self.data = mujoco.MjData(self.model)
        self.sample_data = mujoco.MjData(self.model)
        self.site = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "flange")
        self.target_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "target")
        self.mocap = self.model.body_mocapid[self.target_body]
        self.observation_space = spaces.Box(-np.inf, np.inf, (36,), np.float32)
        self.action_space = spaces.Box(-1, 1, (6,), np.float32)
        self.command_q = HOME.copy()
        self.command_v = np.zeros(6)
        self.last_action = np.zeros(6)
        self.target = np.zeros(3)
        self.steps = 0
        self.max_steps = CONFIG["episode_steps"]

    def position(self):
        return self.data.site_xpos[self.site].copy()

    def observation(self):
        return np.concatenate((self.data.qpos - HOME, self.data.qvel,
                               self.position(), (self.target-self.position())*5,
                               self.last_action,
                               (self.command_q-HOME)/CONFIG["action_scale_rad"],
                               self.command_v/CONFIG["reference_speed_rad_s"])).astype(np.float32)

    def info(self):
        error = float(np.linalg.norm(self.target - self.position()))
        contacts = sum(contact.dist < 0 for contact in self.data.contact)
        return {"position_error_m": error, "contacts": int(contacts),
                "is_success": error < CONFIG["position_tolerance_m"] and contacts == 0,
                "max_joint_speed_rad_s": float(np.max(np.abs(self.data.qvel))),
                "max_penetration_m": max((max(0., -float(c.dist)) for c in self.data.contact), default=0.)}

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = HOME + self.np_random.uniform(-0.03, 0.03, 6)
        self.command_q = self.data.qpos.copy()
        self.command_v.fill(0)
        self.last_action.fill(0)
        self.steps = 0
        self.data.ctrl[:] = self.command_q
        mujoco.mj_forward(self.model, self.data)
        if self.data.ncon:
            raise RuntimeError("Initial pose is in contact")
        self.sample_target()
        return self.observation(), self.info()

    def set_target(self, position):
        """Change only the point goal, preserving robot and controller state."""
        position = np.asarray(position, dtype=np.float64)
        if position.shape != (3,) or not np.isfinite(position).all():
            raise ValueError("Target must contain three finite coordinates in metres")
        self.target = position.copy()
        self.data.mocap_pos[self.mocap] = self.target
        mujoco.mj_forward(self.model, self.data)
        return self.observation()

    def sample_target(self):
        width = np.array(CONFIG["target_joint_sampling_half_width_rad"])
        for _ in range(100):
            mujoco.mj_resetData(self.model, self.sample_data)
            self.sample_data.qpos[:] = HOME + self.np_random.uniform(-width, width)
            mujoco.mj_forward(self.model, self.sample_data)
            target = self.sample_data.site_xpos[self.site].copy()
            if self.sample_data.ncon == 0 and target[2] > 0.15 and np.linalg.norm(target-self.position()) > 0.05:
                break
        else:
            raise RuntimeError("Could not sample a collision-free reachable target")
        return self.set_target(target)

    def bias_force(self):
        """Compensation hook; sensitivity tests may use a mismatched nominal model."""
        return self.data.qfrc_bias

    def step(self, action):
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (6,) or not np.isfinite(action).all():
            raise ValueError("Action must contain six finite values")
        action = np.clip(action, -1, 1)
        dt = CONFIG["control_dt"]
        target = HOME + CONFIG["action_scale_rad"] * action
        target = np.clip(target, self.data.qpos-CONFIG["tracking_error_rad"],
                         self.data.qpos+CONFIG["tracking_error_rad"])
        error = target - self.command_q
        desired_v = np.sign(error) * np.minimum(CONFIG["reference_speed_rad_s"],
                                                np.sqrt(2*CONFIG["reference_accel_rad_s2"]*np.abs(error)))
        self.command_v += np.clip(desired_v-self.command_v,
                                  -CONFIG["reference_accel_rad_s2"]*dt, CONFIG["reference_accel_rad_s2"]*dt)
        self.command_q += self.command_v*dt
        # Home +/- action scale is inside the official URDF limits; retain hard limits too.
        self.command_q = np.clip(self.command_q, self.model.jnt_range[:, 0], self.model.jnt_range[:, 1])
        self.data.ctrl[:] = self.command_q
        previous_position = self.position()
        peak_speed = 0.
        physics_contact_steps = 0
        peak_depth = 0.
        for _ in range(16):
            # MuJoCo actuator affine bias: const - kp*q - kv*qvel.
            # Feedforward shares actuator force limits, rather than bypassing them.
            self.model.actuator_biasprm[:, 0] = self.bias_force() - self.model.actuator_biasprm[:, 2]*self.command_v
            mujoco.mj_step(self.model, self.data)
            peak_speed = max(peak_speed, float(np.max(np.abs(self.data.qvel))))
            physics_contact_steps += int(any(c.dist < 0 for c in self.data.contact))
            peak_depth = max(peak_depth, max((-float(c.dist) for c in self.data.contact), default=0.))
        mujoco.mj_forward(self.model, self.data)
        self.steps += 1
        info = self.info()
        info.update(physics_peak_joint_speed_rad_s=peak_speed,
                    physics_contact_steps=physics_contact_steps,
                    physics_peak_penetration_m=peak_depth)
        speed = np.linalg.norm(self.position()-previous_position)/dt
        distance = info["position_error_m"]
        reward = (-5*distance + np.exp(-(distance/0.05)**2)
                  + 0.5*np.exp(-(distance/0.025)**2)*np.exp(-(speed/0.1)**2)
                  - 0.005*np.square(action-self.last_action).sum()
                  - 0.002*np.square(self.data.qvel).sum() - 2*bool(info["contacts"]))
        self.last_action = action.copy()
        observation = self.observation()
        invalid = not np.isfinite(observation).all() or not np.isfinite(reward)
        return observation, float(-100 if invalid else reward), bool(invalid), self.steps >= self.max_steps, info


def load_policy(path, env):
    from stable_baselines3 import PPO
    model = PPO.load(path, device="cpu")
    if getattr(model, "zu5_config", None) != CONFIG:
        raise ValueError("ZU5 configuration mismatch; refusing incompatible checkpoint")
    if model.observation_space.shape != (36,) or model.action_space.shape != (6,):
        raise ValueError("ZU5 policy shape mismatch")
    return model
