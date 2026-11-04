"""CPU MuJoCo version of the Kinova Gen3 Cartesian reach task.

The task follows the original Isaac Lab reach setup where practical: seven
joint-position actions, Cartesian/proprioceptive observations (28 values, or
42 when the filtered position/velocity controller state is included),
random Cartesian targets held for four seconds, and a twelve-second episode.
MuJoCo uses a single robot model rather than Isaac Lab's thousands of parallel
GPU environments, so this is a task-level reproduction, not a physics or
checkpoint-compatible port.
"""

from __future__ import annotations

from pathlib import Path

import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np


NUM_JOINTS = 7
OBSERVATION_SIZE = 28
DEFAULT_JOINT_POS = np.array([0.0, 0.65, 0.0, 1.89, 0.0, 0.60, -1.57], dtype=np.float64)
TARGET_POS_LOW = np.array([0.35, -0.20, 0.15], dtype=np.float64)
TARGET_POS_HIGH = np.array([0.65, 0.20, 0.50], dtype=np.float64)
POSITION_SUCCESS_TOLERANCE = 0.05
ORIENTATION_SUCCESS_TOLERANCE = 0.20


class KinovaReachEnv(gym.Env):
    """Seven-DoF Kinova Gen3 reach task using the vendored Menagerie model."""

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(
        self,
        *,
        max_steps: int = 375,
        frame_skip: int = 16,
        target_frequency_hz: float = 0.25,
        command_resample_seconds: float | None = None,
        action_scale: float = 0.5,
        joint_target_speed_limit: float = 1.5,
        joint_target_accel_limit: float = 4.0,
        position_reward_weight: float = 0.2,
        orientation_reward_weight: float = 0.0,
        random_target_yaw: bool = False,
        damping_multiplier: float = 1.0,
        controller_observation: bool = False,
        settling_reward_weight: float = 0.0,
        physical_obstacles: bool = False,
        ball_clearance: float = 0.02,
        target_delta_scale: float = 1.0,
        approach_axis: str = "above",
        bias_compensation: bool = False,
        velocity_feedforward: bool = False,
        tracking_error_limit: float | None = None,
        observation_noise: float = 0.0,
        model_path: str | Path | None = None,
    ):
        super().__init__()
        if not np.isfinite(damping_multiplier) or damping_multiplier <= 0:
            raise ValueError("damping_multiplier must be finite and positive")
        if not np.isfinite(settling_reward_weight) or settling_reward_weight < 0:
            raise ValueError("settling_reward_weight must be finite and nonnegative")
        if not np.isfinite(ball_clearance) or ball_clearance < 0:
            raise ValueError("ball_clearance must be finite and nonnegative")
        if not np.isfinite(target_delta_scale) or target_delta_scale <= 0:
            raise ValueError("target_delta_scale must be finite and positive")
        self.target_delta_scale = float(target_delta_scale)
        if approach_axis not in ("above", "front"):
            raise ValueError("approach_axis must be 'above' or 'front'")
        self.approach_axis = approach_axis
        self.bias_compensation = bool(bias_compensation)
        self.velocity_feedforward = bool(velocity_feedforward)
        if tracking_error_limit is not None and (
                not np.isfinite(tracking_error_limit) or tracking_error_limit <= 0):
            raise ValueError("tracking_error_limit must be finite and positive, or None")
        self.tracking_error_limit = tracking_error_limit
        if max_steps < 1 or frame_skip < 1:
            raise ValueError("max_steps and frame_skip must be positive")
        if (action_scale <= 0 or not np.isfinite(joint_target_speed_limit)
                or joint_target_speed_limit <= 0 or not np.isfinite(joint_target_accel_limit)
                or joint_target_accel_limit <= 0
                or observation_noise < 0
                or not np.isfinite(position_reward_weight) or position_reward_weight < 0
                or not np.isfinite(orientation_reward_weight) or orientation_reward_weight < 0):
            raise ValueError("action scale and joint target speed/acceleration limits must be positive; reward weights and observation noise cannot be negative")
        if command_resample_seconds is None:
            if not np.isfinite(target_frequency_hz) or target_frequency_hz <= 0:
                raise ValueError("target_frequency_hz must be finite and positive")
            command_resample_seconds = 1.0 / target_frequency_hz
        elif not np.isfinite(command_resample_seconds) or command_resample_seconds <= 0:
            raise ValueError("command_resample_seconds must be finite and positive")

        self.model_path = Path(model_path) if model_path else (
            Path(__file__).resolve().parent / "assets" / "kinova_gen3" / "scene.xml"
        )
        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"MuJoCo scene not found: {self.model_path}. "
                "Keep the assets/kinova_gen3 directory next to kinova_env.py."
            )

        # Load XML and meshes through MuJoCo's virtual file system. This avoids
        # native path encoding failures when the project folder contains CJK
        # characters on Windows, and uses only files shipped with the project.
        model_assets = {"gen3.xml": (self.model_path.parent / "gen3.xml").read_bytes()}
        mesh_dir = self.model_path.parent / "assets"
        for asset_path in mesh_dir.rglob("*"):
            if asset_path.is_file():
                virtual_path = (Path("assets") / asset_path.relative_to(mesh_dir)).as_posix()
                model_assets[virtual_path] = asset_path.read_bytes()
        scene_xml = self.model_path.read_text(encoding="utf-8")
        self.model = mujoco.MjModel.from_xml_string(scene_xml, assets=model_assets)
        self.data = mujoco.MjData(self.model)
        self._joint_ids = np.array([
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"joint_{i}")
            for i in range(1, NUM_JOINTS + 1)
        ], dtype=np.int32)
        self._actuator_ids = np.array([
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"joint_{i}")
            for i in range(1, NUM_JOINTS + 1)
        ], dtype=np.int32)
        self._site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "pinch_site")
        self.physical_obstacles = bool(physical_obstacles)
        self.ball_clearance = float(ball_clearance)
        self.position_tolerance = 0.025 if self.physical_obstacles else POSITION_SUCCESS_TOLERANCE
        self._ball_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "target_ball")
        self._tip_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "reach_tip")
        ball_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "target_ball_body")
        self._ball_mocap_id = int(self.model.body_mocapid[ball_body])
        self._obstacle_geom_ids = {
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
            for name in ("target_ball", "table_top", "table_leg_1", "table_leg_2",
                         "table_leg_3", "table_leg_4", "robot_pedestal")
        }
        self._robot_geom_ids = np.where(
            (self.model.geom_bodyid != 0) & (self.model.geom_bodyid != ball_body)
            & (self.model.geom_contype != 0)
        )[0]
        if not self.physical_obstacles:
            disabled = list(self._obstacle_geom_ids) + [self._tip_geom_id]
            self.model.geom_contype[disabled] = 0
            self.model.geom_conaffinity[disabled] = 0
        if np.any(self._joint_ids < 0) or np.any(self._actuator_ids < 0):
            raise RuntimeError("The Gen3 model must define joint_1..joint_7 and matching actuators")
        if self.model.nq != NUM_JOINTS or self.model.nv != NUM_JOINTS or self.model.nu != NUM_JOINTS:
            raise RuntimeError(
                "Expected a seven-joint Gen3 model; got "
                f"nq={self.model.nq}, nv={self.model.nv}, nu={self.model.nu}"
            )
        if self._site_id < 0:
            raise RuntimeError("The Gen3 model must define the 'pinch_site' end-effector site")
        if not np.array_equal(self.model.actuator_trnid[self._actuator_ids, 0], self._joint_ids):
            raise RuntimeError("Gen3 actuators must be position actuators paired with joint_1..joint_7")

        self._qpos_ids = self.model.jnt_qposadr[self._joint_ids].copy()
        self._qvel_ids = self.model.jnt_dofadr[self._joint_ids].copy()
        self._default_q = DEFAULT_JOINT_POS.copy()
        self._match_isaac_actuators()
        self.model.actuator_biasprm[self._actuator_ids, 2] *= damping_multiplier
        self._lower, self._upper = self._joint_limits()
        self._ctrl_lower, self._ctrl_upper = self._actuator_limits()

        self.max_steps = int(max_steps)
        self.frame_skip = int(frame_skip)
        self.action_scale = float(action_scale)
        self.observation_noise = float(observation_noise)
        self.control_dt = float(self.model.opt.timestep * self.frame_skip)
        self.target_frequency_hz = 1.0 / float(command_resample_seconds)
        self.command_resample_steps = max(1, int(round(command_resample_seconds / self.control_dt)))
        self.joint_target_speed_limit = float(joint_target_speed_limit)
        self.joint_target_accel_limit = float(joint_target_accel_limit)
        self.position_reward_weight = float(position_reward_weight)
        self.orientation_reward_weight = float(orientation_reward_weight)
        self.random_target_yaw = bool(random_target_yaw)
        self.damping_multiplier = float(damping_multiplier)
        self.controller_observation = bool(controller_observation)
        self.settling_reward_weight = float(settling_reward_weight)

        self.action_space = spaces.Box(-1.0, 1.0, shape=(NUM_JOINTS,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(OBSERVATION_SIZE + (14 if self.controller_observation else 0),),
            dtype=np.float32
        )
        self.target_pos = np.zeros(3, dtype=np.float64)
        self.target_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self.target_index = -1
        self._target_start_step = 0
        self._command_q = self._default_q.copy()
        self._command_velocity = np.zeros(NUM_JOINTS, dtype=np.float64)
        self.last_action = np.zeros(NUM_JOINTS, dtype=np.float32)
        self.steps = 0

    def _joint_limits(self) -> tuple[np.ndarray, np.ndarray]:
        lower = np.full(NUM_JOINTS, -2.0 * np.pi, dtype=np.float64)
        upper = np.full(NUM_JOINTS, 2.0 * np.pi, dtype=np.float64)
        for i, joint_id in enumerate(self._joint_ids):
            if self.model.jnt_limited[joint_id]:
                lower[i], upper[i] = self.model.jnt_range[joint_id]
        return lower, upper

    def _match_isaac_actuators(self) -> None:
        """Map the original task's joint drives onto MuJoCo position actuators."""
        stiffness = np.array([40.0, 40.0, 40.0, 40.0, 15.0, 15.0, 15.0])
        damping = np.array([1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5])
        effort_limit = np.array([39.0, 39.0, 39.0, 39.0, 9.0, 9.0, 9.0])
        for index, actuator_id in enumerate(self._actuator_ids):
            self.model.actuator_gainprm[actuator_id, 0] = stiffness[index]
            # MuJoCo affine position-actuator bias is [const, -kp, -kv].
            self.model.actuator_biasprm[actuator_id, 1] = -stiffness[index]
            self.model.actuator_biasprm[actuator_id, 2] = -damping[index]
            self.model.actuator_forcelimited[actuator_id] = True
            self.model.actuator_forcerange[actuator_id] = (-effort_limit[index], effort_limit[index])

    def _actuator_limits(self) -> tuple[np.ndarray, np.ndarray]:
        lower = np.full(NUM_JOINTS, -np.inf, dtype=np.float64)
        upper = np.full(NUM_JOINTS, np.inf, dtype=np.float64)
        for i, actuator_id in enumerate(self._actuator_ids):
            if self.model.actuator_ctrllimited[actuator_id]:
                lower[i], upper[i] = self.model.actuator_ctrlrange[actuator_id]
        return lower, upper

    def _sample_target(self) -> None:
        self.target_index += 1
        self._target_start_step = self.steps
        for _ in range(100):
            self.target_pos = self.np_random.uniform(TARGET_POS_LOW, TARGET_POS_HIGH)
            self.data.mocap_pos[self._ball_mocap_id] = self.target_pos
            mujoco.mj_forward(self.model, self.data)
            # A newly appearing target must not teleport into the moving arm.
            if not self.physical_obstacles or self._ball_surface_clearance() > 0.04:
                break
        else:
            raise RuntimeError("Could not sample a ball position clear of the robot")
        self._sample_target_orientation()

    def _sample_target_orientation(self) -> None:
        yaw = (
            float(self.np_random.uniform(-np.pi, np.pi))
            if self.random_target_yaw
            else 0.0
        )
        self.target_quat = np.zeros(4, dtype=np.float64)
        mujoco.mju_euler2Quat(
            self.target_quat,
            np.array([0.0, np.pi / 2.0, yaw], dtype=np.float64),
            "xyz",
        )

    def _current_position(self) -> np.ndarray:
        return self.data.site_xpos[self._site_id].copy()

    @property
    def reach_position(self) -> np.ndarray:
        """Stop before the ball surface; legacy models reach its center."""
        target = self.target_pos.copy()
        if self.physical_obstacles:
            target -= self.approach_direction * (
                self.model.geom_size[self._ball_geom_id, 0]
                + self.model.geom_size[self._tip_geom_id, 0] + self.ball_clearance
            )
        return target

    @property
    def approach_direction(self) -> np.ndarray:
        return np.array([1., 0., 0.]) if self.approach_axis == "front" else np.array([0., 0., -1.])

    def _ball_surface_clearance(self) -> float:
        distance = 0.1
        for geom in self._robot_geom_ids:
            center_distance = np.linalg.norm(self.data.geom_xpos[geom] - self.target_pos)
            if center_distance - self.model.geom_rbound[geom] - self.model.geom_size[self._ball_geom_id, 0] >= distance:
                continue
            distance = min(distance, mujoco.mj_geomDistance(
                self.model, self.data, int(geom), self._ball_geom_id, distance, None
            ))
        return float(distance)

    def _obstacle_contacts(self) -> tuple[int, float]:
        count, penetration = 0, 0.0
        for contact in self.data.contact:
            if (contact.geom1 in self._obstacle_geom_ids or contact.geom2 in self._obstacle_geom_ids):
                if contact.dist < 0:
                    count += 1
                    penetration = max(penetration, -float(contact.dist))
        return count, penetration

    def _pose_errors(self) -> tuple[float, float]:
        position_error = float(np.linalg.norm(self.reach_position - self._current_position()))
        current_rotation = self.data.site_xmat[self._site_id].reshape(3, 3)
        target_rotation = np.zeros(9, dtype=np.float64)
        mujoco.mju_quat2Mat(target_rotation, self.target_quat)
        relative_rotation = target_rotation.reshape(3, 3).T @ current_rotation
        cosine = np.clip((np.trace(relative_rotation) - 1.0) * 0.5, -1.0, 1.0)
        orientation_error = float(np.arccos(cosine))
        return position_error, orientation_error

    def _observation(self) -> np.ndarray:
        qpos = self.data.qpos[self._qpos_ids] - self._default_q
        qvel = self.data.qvel[self._qvel_ids]
        if self.observation_noise:
            qpos = qpos + self.np_random.uniform(-self.observation_noise, self.observation_noise, NUM_JOINTS)
            qvel = qvel + self.np_random.uniform(-self.observation_noise, self.observation_noise, NUM_JOINTS)
        end_effector_pos = self._current_position()
        target_delta = (self.reach_position - end_effector_pos) * self.target_delta_scale
        target_age = np.clip(
            (self.steps - self._target_start_step) / self.command_resample_steps, 0.0, 1.0
        )
        observation = np.concatenate((
            qpos, qvel, end_effector_pos, target_delta,
            np.array([target_age]), self.last_action,
        ))
        if self.controller_observation:
            observation = np.concatenate((
                observation,
                (self._command_q - self._default_q) / self.action_scale,
                self._command_velocity / self.joint_target_speed_limit,
            ))
        return observation.astype(np.float32)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        reset_scale = self.np_random.uniform(0.75, 1.25, size=NUM_JOINTS)
        qpos = self._default_q * reset_scale
        qpos = np.clip(qpos, self._lower, self._upper)
        self.data.qpos[self._qpos_ids] = qpos
        self.data.qvel[self._qvel_ids] = 0.0
        initial_control = np.clip(qpos if (self.physical_obstacles or self.tracking_error_limit is not None)
                                  else self._default_q,
                                  self._ctrl_lower, self._ctrl_upper)
        self.data.ctrl[self._actuator_ids] = initial_control
        self._command_q = initial_control.copy()
        self._command_velocity.fill(0.0)
        mujoco.mj_forward(self.model, self.data)

        if self.bias_compensation or self.velocity_feedforward:
            self.model.actuator_biasprm[self._actuator_ids, 0] = (
                self.data.qfrc_bias[self._qvel_ids] if self.bias_compensation else 0.0
            )
            mujoco.mj_forward(self.model, self.data)

        self.steps = 0
        self.last_action.fill(0.0)
        self.target_index = -1
        self._sample_target()
        position_error, orientation_error = self._pose_errors()
        observation = self._observation()
        position_success = position_error < self.position_tolerance and (
            not self.physical_obstacles or
            (self._ball_surface_clearance() >= 0.005 and self._obstacle_contacts()[0] == 0)
        )
        info = {
            "target_index": self.target_index,
            "position_error": position_error,
            "orientation_error": orientation_error,
            "is_success": position_success,
            "is_pose_success": (
                position_success
                and orientation_error < ORIENTATION_SUCCESS_TOLERANCE
            ),
        }
        return observation, info

    def step(self, action):
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (NUM_JOINTS,):
            raise ValueError(f"Expected an action with shape ({NUM_JOINTS},), got {action.shape}")
        action = np.clip(action, -1.0, 1.0)

        previous_action = self.last_action.astype(np.float64)
        previous_position = self._current_position()
        target_q = self._default_q + self.action_scale * action
        target_q = np.clip(target_q, self._lower, self._upper)
        target_q = np.clip(target_q, self._ctrl_lower, self._ctrl_upper)
        if self.tracking_error_limit is not None:
            # A collision can block the real joint while an open-loop reference
            # keeps moving. Govern the reference relative to measured position
            # so removal of the old target cannot release a large stored error.
            actual_q = self.data.qpos[self._qpos_ids]
            target_q = np.clip(target_q, actual_q - self.tracking_error_limit,
                               actual_q + self.tracking_error_limit)
        target_error = target_q - self._command_q
        max_speed_at_error = np.sqrt(2.0 * self.joint_target_accel_limit * np.abs(target_error))
        desired_velocity = np.sign(target_error) * np.minimum(
            self.joint_target_speed_limit, max_speed_at_error
        )
        previous_command_velocity = self._command_velocity.copy()
        max_velocity_delta = self.joint_target_accel_limit * self.control_dt
        velocity_delta = np.clip(
            desired_velocity - self._command_velocity,
            -max_velocity_delta,
            max_velocity_delta,
        )
        next_command_velocity = self._command_velocity + velocity_delta
        command_delta = next_command_velocity * self.control_dt
        self._command_q += command_delta
        self._command_velocity = next_command_velocity
        self.data.ctrl[self._actuator_ids] = self._command_q
        if self.bias_compensation or self.velocity_feedforward:
            # Bias and reference-velocity feedforward share the existing
            # actuator force clamp with position/velocity feedback.
            feedforward = (self.data.qfrc_bias[self._qvel_ids].copy()
                           if self.bias_compensation else np.zeros(NUM_JOINTS))
            if self.velocity_feedforward:
                feedforward -= self.model.actuator_biasprm[self._actuator_ids, 2] * self._command_velocity
            self.model.actuator_biasprm[self._actuator_ids, 0] = feedforward
        mujoco.mj_step(self.model, self.data, nstep=self.frame_skip)
        if self.physical_obstacles:
            mujoco.mj_forward(self.model, self.data)
        self.steps += 1

        # The final episode boundary is not a new target window: for a 12 s
        # episode and a 4 s target hold, targets appear at 0, 4, and 8 s.
        if self.steps < self.max_steps and self.steps % self.command_resample_steps == 0:
            self._sample_target()
        self.last_action = action.astype(np.float32)

        position_error, orientation_error = self._pose_errors()
        end_effector_speed = float(
            np.linalg.norm(self._current_position() - previous_position) / self.control_dt
        )
        joint_velocity = self.data.qvel[self._qvel_ids]
        action_rate = float(np.dot(action - previous_action, action - previous_action))
        velocity_cost = float(np.dot(joint_velocity, joint_velocity))
        ball_surface_clearance = self._ball_surface_clearance() if self.physical_obstacles else 0.1
        obstacle_contacts, penetration = self._obstacle_contacts() if self.physical_obstacles else (0, 0.0)
        approach_error = float(np.arccos(np.clip(np.dot(
            self.data.site_xmat[self._site_id].reshape(3, 3)[:, 2], self.approach_direction
        ), -1.0, 1.0)))
        clearance_factor = 1.0
        if self.physical_obstacles:
            clearance_factor = (float(np.clip(ball_surface_clearance / max(self.ball_clearance, 0.005), 0, 1))
                                if obstacle_contacts == 0 else 0.0)
        # The base terms follow the original reach task. The optional settling
        # term rewards staying near the sphere at low speed, rather than
        # briefly crossing its tolerance. Rewards accumulate per second.
        reward = self.control_dt * (
            -self.position_reward_weight * position_error
            + 0.1 * (1.0 - np.tanh(position_error / 0.1))
            - self.orientation_reward_weight * (approach_error if self.physical_obstacles else orientation_error)
            - (0.001 if self.physical_obstacles else 0.0001) * action_rate
            - 0.0001 * velocity_cost
            + self.settling_reward_weight
            * clearance_factor
            * np.exp(-np.square(position_error / POSITION_SUCCESS_TOLERANCE))
            * np.exp(-np.square(end_effector_speed / 0.1))
            - 0.1 * float(obstacle_contacts > 0)
            - max(0.0, self.ball_clearance - ball_surface_clearance) * (3.0 if self.physical_obstacles else 0.0)
        )

        observation = self._observation()
        bad_state = not (
            np.all(np.isfinite(observation))
            and np.isfinite(position_error)
            and np.isfinite(orientation_error)
        )
        terminated = bool(bad_state)
        truncated = self.steps >= self.max_steps
        if bad_state:
            reward = -100.0
        position_success = position_error < self.position_tolerance and (
            not self.physical_obstacles or (ball_surface_clearance >= 0.005 and obstacle_contacts == 0)
        )
        info = {
            "target_index": self.target_index,
            "position_error": position_error,
            "orientation_error": orientation_error,
            "joint_speed": float(np.linalg.norm(joint_velocity)),
            "end_effector_speed": end_effector_speed,
            "ball_surface_clearance": ball_surface_clearance,
            "obstacle_contacts": obstacle_contacts,
            "obstacle_penetration": penetration,
            "approach_error": approach_error,
            "joint_command_speed": float(np.linalg.norm(self._command_velocity)),
            "joint_command_accel": float(
                np.linalg.norm((self._command_velocity - previous_command_velocity) / self.control_dt)
            ),
            "action_change": float(np.linalg.norm(action - previous_action)),
            "is_success": position_success,
            "is_pose_success": (
                position_success
                and orientation_error < ORIENTATION_SUCCESS_TOLERANCE
            ),
        }
        return observation, float(reward), terminated, truncated, info

    def render(self):
        # Interactive display is owned by play.py so training stays headless.
        return None

    def set_action_scale(self, action_scale: float) -> None:
        """Set the joint-target offset scale, used by the optional training curriculum."""
        if action_scale <= 0:
            raise ValueError("action_scale must be positive")
        self.action_scale = float(action_scale)

    def close(self):
        pass
