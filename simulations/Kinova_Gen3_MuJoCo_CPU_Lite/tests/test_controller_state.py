import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import mujoco
from stable_baselines3 import PPO

from kinova_env import KinovaReachEnv
from policy_tools import (expand_controller_policy, rescale_target_inputs,
                          CORE_ENV_DEFAULTS, resolve_core_arguments, controller_options,
                          save_checkpoint_atomic)


class ControllerStateChecks(unittest.TestCase):
    def test_filtered_state_and_motion_limits(self):
        env = KinovaReachEnv(action_scale=1.5, controller_observation=True,
                             damping_multiplier=4, joint_target_speed_limit=0.8,
                             joint_target_accel_limit=2)
        obs, _ = env.reset(seed=91)
        self.assertEqual(obs.shape, (42,))
        np.testing.assert_array_equal(obs[-14:], np.zeros(14))
        np.testing.assert_allclose(env.model.actuator_biasprm[:, 2], [-4]*4 + [-2]*3)
        previous_velocity = env._command_velocity.copy()
        indices = set()
        for step in range(env.max_steps):
            action = np.full(7, 1 if step < 100 else -1, dtype=np.float32)
            obs, reward, terminated, truncated, info = env.step(action)
            self.assertTrue(np.isfinite(reward) and np.isfinite(obs).all())
            self.assertFalse(terminated)
            self.assertLessEqual(np.max(np.abs(env._command_velocity)), 0.8 + 1e-10)
            self.assertLessEqual(np.max(np.abs(env._command_velocity - previous_velocity)),
                                 2 * env.control_dt + 1e-10)
            previous_velocity = env._command_velocity.copy()
            np.testing.assert_allclose(obs[-7:], env._command_velocity / 0.8, atol=1e-7)
            indices.add(info["target_index"])
        self.assertTrue(truncated)
        self.assertEqual(indices, {0, 1, 2})

    def test_checkpoint_expansion_preserves_actor_and_critic(self):
        torch.set_num_threads(1)
        source = PPO("MlpPolicy", KinovaReachEnv(), n_steps=8, batch_size=8, seed=42,
                     policy_kwargs={"net_arch": {"pi": [64, 64], "vf": [64, 64]}})
        source.num_timesteps = 1450000
        env = KinovaReachEnv(controller_observation=True)
        expanded = expand_controller_policy(source, env, 3e-5)
        observation = np.random.default_rng(42).normal(size=(10, 28)).astype(np.float32)
        augmented = np.concatenate((observation, np.ones((10, 14), dtype=np.float32)), axis=1)
        np.testing.assert_allclose(source.predict(observation, deterministic=True)[0],
                                   expanded.predict(augmented, deterministic=True)[0], atol=1e-6)
        with torch.no_grad():
            np.testing.assert_allclose(
                source.policy.predict_values(torch.from_numpy(observation)).numpy(),
                expanded.policy.predict_values(torch.from_numpy(augmented)).numpy(), atol=1e-6)
        self.assertEqual(expanded.num_timesteps, source.num_timesteps)

    def test_settling_reward_uses_position_and_speed(self):
        base = KinovaReachEnv(damping_multiplier=4)
        shaped = KinovaReachEnv(damping_multiplier=4, settling_reward_weight=0.25)
        for env in (base, shaped):
            env.reset(seed=42)
            env.target_pos = env._current_position()
        _, baseline_reward, _, _, _ = base.step(np.zeros(7))
        _, shaped_reward, _, _, info = shaped.step(np.zeros(7))
        expected = 0.25 * shaped.control_dt * np.exp(-(info["position_error"] / 0.05)**2) \
            * np.exp(-(info["end_effector_speed"] / 0.1)**2)
        self.assertGreater(expected, 0)
        self.assertAlmostEqual(shaped_reward - baseline_reward, expected, places=10)

    def test_solid_ball_contact_and_surface_offset(self):
        env = KinovaReachEnv(physical_obstacles=True, controller_observation=True,
                             damping_multiplier=4)
        env.reset(seed=81)
        self.assertGreater(env._ball_surface_clearance(), 0.04)
        np.testing.assert_allclose(env.reach_position - env.target_pos, [0, 0, 0.053])
        for name in ("target_ball", "table_top", "table_leg_1", "robot_pedestal"):
            geom = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, name)
            self.assertNotEqual(env.model.geom_contype[geom], 0)
            self.assertEqual(env.model.geom_priority[geom], 1)
        # Deliberately overlap the tip and solid ball to verify a real contact,
        # independent of drawing the sphere in the viewer.
        env.target_pos = env._current_position()
        env.data.mocap_pos[env._ball_mocap_id] = env.target_pos
        mujoco.mj_forward(env.model, env.data)
        self.assertTrue(any(
            {int(contact.geom1), int(contact.geom2)} == {env._ball_geom_id, env._tip_geom_id}
            and contact.dist < 0 for contact in env.data.contact
        ))
        for _ in range(64):
            obs, reward, terminated, _, _ = env.step(np.zeros(7))
            self.assertFalse(terminated)
            self.assertTrue(np.isfinite(obs).all() and np.isfinite(reward))
        self.assertLess(env._obstacle_contacts()[1], 0.001)

    def test_cartesian_rescaling_preserves_prediction(self):
        source = PPO("MlpPolicy", KinovaReachEnv(controller_observation=True),
                     n_steps=8, batch_size=8, seed=42)
        observation = np.random.default_rng(42).normal(size=(10, 42)).astype(np.float32)
        original = source.predict(observation, deterministic=True)[0]
        rescale_target_inputs(source, 1, 10)
        observation[:, 17:20] *= 10
        np.testing.assert_allclose(source.predict(observation, deterministic=True)[0], original, atol=1e-6)

    def test_compensated_arm_holds_its_pose_without_falling(self):
        env = KinovaReachEnv(physical_obstacles=True, controller_observation=True,
                             damping_multiplier=4, action_scale=1.5, bias_compensation=True)
        env.reset(seed=81)
        initial = env.data.qpos.copy()
        action = (initial - env._default_q) / env.action_scale
        for _ in range(100):
            _, _, terminated, _, _ = env.step(action)
            self.assertFalse(terminated)
        np.testing.assert_allclose(env.data.qpos, initial, atol=1e-5)

    def test_checkpoint_settings_and_atomic_publication(self):
        model = PPO("MlpPolicy", KinovaReachEnv(controller_observation=True),
                    n_steps=8, batch_size=8, seed=42)
        model.reach_config = {"action_scale": 2.5, "orientation_reward_weight": .05,
                              "physical_obstacles": True, "bias_compensation": True,
                              "approach_axis": "front", "damping_multiplier": 4}
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "best_model.zip"
            save_checkpoint_atomic(model, path)
            loaded = PPO.load(path, device="cpu")
            self.assertEqual(loaded.reach_config, model.reach_config)
            self.assertFalse((Path(folder)/"best_model.pending.zip").exists())
        extra_names = ("damping_multiplier", "settling_reward_weight", "controller_observation",
                       "physical_obstacles", "ball_clearance", "target_delta_scale",
                       "approach_axis", "bias_compensation", "velocity_feedforward",
                       "tracking_error_limit")
        args = SimpleNamespace(**{name: None for name in (*CORE_ENV_DEFAULTS, *extra_names)})
        resolve_core_arguments(args, loaded)
        self.assertEqual(args.action_scale, 2.5)
        self.assertEqual(args.orientation_reward_weight, .05)
        self.assertTrue(controller_options(args, loaded)["bias_compensation"])
        self.assertEqual(controller_options(args, loaded)["approach_axis"], "front")

    def test_pose_success_uses_the_same_clearance_position_criterion(self):
        env = KinovaReachEnv(physical_obstacles=True, approach_axis="front",
                             bias_compensation=True, damping_multiplier=10, action_scale=2.5)
        env.reset(seed=81)
        action = (env.data.qpos.copy() - env._default_q) / env.action_scale
        # Isolate the success classifier from the deliberately nearby sphere's
        # contact response; solid-contact dynamics have their own test above.
        env.model.geom_contype[env._ball_geom_id] = 0
        env.model.geom_conaffinity[env._ball_geom_id] = 0
        env.target_pos = env._current_position() + np.array([.053, .04, 0])
        env.data.mocap_pos[env._ball_mocap_id] = env.target_pos
        mujoco.mju_mat2Quat(env.target_quat, env.data.site_xmat[env._site_id])
        _, _, _, _, info = env.step(action)
        self.assertGreater(info["position_error"], .025)
        self.assertLess(info["position_error"], .05)
        self.assertLess(info["orientation_error"], .2)
        self.assertFalse(info["is_success"])
        self.assertFalse(info["is_pose_success"])

    def test_table_generates_contact_at_its_visible_surface(self):
        env = KinovaReachEnv(physical_obstacles=True, action_scale=2.5)
        env.reset(seed=81)
        goal = np.array([.4, 0, -.01])
        q = env._default_q.copy()
        for _ in range(100):
            env.data.qpos[:] = q
            mujoco.mj_forward(env.model, env.data)
            error = goal - env._current_position()
            if np.linalg.norm(error) < .001:
                break
            jac = np.zeros((3, 7))
            mujoco.mj_jacSite(env.model, env.data, jac, None, env._site_id)
            delta = jac.T @ np.linalg.solve(jac @ jac.T + 1e-4*np.eye(3), error)
            q = np.clip(q + np.clip(delta, -.15, .15), env._lower, env._upper)
        table = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, "table_top")
        self.assertLess(np.linalg.norm(goal - env._current_position()), .001)
        self.assertTrue(any(
            {int(contact.geom1), int(contact.geom2)} == {table, env._tip_geom_id}
            and contact.dist < 0 for contact in env.data.contact
        ))

    def test_governor_limits_blocked_joint_backlog_and_release_speed(self):
        env = KinovaReachEnv(action_scale=2.5, controller_observation=True,
                             damping_multiplier=10, bias_compensation=True,
                             velocity_feedforward=True, tracking_error_limit=.18,
                             joint_target_speed_limit=.8, joint_target_accel_limit=2)
        env.reset(seed=81)
        joint = 5
        # Very high viscous damping physically blocks the wrist. This tests
        # the reference generator without overwriting simulated positions.
        env.model.dof_damping[joint] = 1e6
        action = np.zeros(7); action[joint] = 1
        previous = env._command_velocity.copy()
        for _ in range(200):
            env.step(action)
            self.assertLess(abs(env._command_q[joint] - env.data.qpos[joint]), .19)
            self.assertLessEqual(np.max(np.abs(env._command_velocity)), .8 + 1e-10)
            self.assertLessEqual(np.max(np.abs(env._command_velocity-previous)), 2*env.control_dt + 1e-10)
            previous = env._command_velocity.copy()
        env.model.dof_damping[joint] = 0
        speeds=[]
        for _ in range(100):
            env.step(action)
            speeds.append(abs(env.data.qvel[joint]))
        self.assertLess(max(speeds), 1.5)


if __name__ == "__main__":
    unittest.main()
