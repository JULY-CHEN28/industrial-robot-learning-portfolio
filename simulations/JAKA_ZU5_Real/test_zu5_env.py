import unittest
import tempfile
from pathlib import Path

from gymnasium.utils.env_checker import check_env
import numpy as np

from zu5_env import CONFIG, Zu5ReachEnv, load_policy


class EnvironmentTests(unittest.TestCase):
    def test_checkpoint_roundtrip_and_reject_mismatched_contract(self):
        from stable_baselines3 import PPO
        env = Zu5ReachEnv()
        obs, _ = env.reset(seed=1)
        model = PPO("MlpPolicy", env, n_steps=8, batch_size=8, seed=1, device="cpu")
        model.zu5_config = CONFIG.copy()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"test.zip"
            model.save(path)
            loaded = load_policy(path, env)
            np.testing.assert_array_equal(model.predict(obs, deterministic=True)[0],
                                          loaded.predict(obs, deterministic=True)[0])
            model.zu5_config = {**CONFIG, "action_size": 7}
            model.save(path)
            with self.assertRaisesRegex(ValueError, "configuration mismatch"):
                load_policy(path, env)

    def test_gym_contract_and_determinism(self):
        env = Zu5ReachEnv()
        check_env(env, skip_render_check=True)

    def test_reject_bad_action_without_mutating_state(self):
        env = Zu5ReachEnv()
        env.reset(seed=42)
        q = env.data.qpos.copy()
        for action in (np.zeros(7), np.full(6, np.nan), np.full(6, np.inf)):
            with self.assertRaises(ValueError):
                env.step(action)
            np.testing.assert_array_equal(q, env.data.qpos)

    def test_reference_limits_under_reversing_actions(self):
        env = Zu5ReachEnv()
        env.reset(seed=42)
        for index in range(250):
            before = env.command_v.copy()
            obs, reward, terminated, _, _ = env.step(np.full(6, 1 if index < 125 else -1))
            self.assertFalse(terminated)
            self.assertTrue(np.isfinite(obs).all() and np.isfinite(reward))
            self.assertLessEqual(np.max(np.abs(env.command_v)), CONFIG["reference_speed_rad_s"]+1e-12)
            self.assertLessEqual(np.max(np.abs(env.command_v-before)), CONFIG["reference_accel_rad_s2"]*CONFIG["control_dt"]+1e-12)
            self.assertTrue((env.data.actuator_force <= env.model.actuator_forcerange[:, 1]+1e-9).all())
            self.assertTrue((env.data.actuator_force >= env.model.actuator_forcerange[:, 0]-1e-9).all())


if __name__ == "__main__":
    unittest.main()
