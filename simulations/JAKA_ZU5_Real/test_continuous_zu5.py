import unittest

import numpy as np

from continuous_zu5 import ContinuousReachEnv, make_schedule
from zu5_env import Zu5ReachEnv


class ContinuousTests(unittest.TestCase):
    def test_switch_preserves_physical_and_controller_state(self):
        env = ContinuousReachEnv(windows=2, window_steps=2)
        baseline = Zu5ReachEnv()
        env.reset(seed=42)
        baseline.reset(seed=42)
        baseline.max_steps = 4
        for step in range(4):
            action = np.full(6, 0.1)
            obs, _, terminated, truncated, info = env.step(action)
            baseline.step(action)
            self.assertFalse(terminated)
            self.assertEqual(info["target_index"], step//2)
            self.assertEqual(truncated, step == 3)
            for value in ("qpos", "qvel", "ctrl"):
                np.testing.assert_array_equal(getattr(env.data, value), getattr(baseline.data, value))
            np.testing.assert_array_equal(env.command_q, baseline.command_q)
            np.testing.assert_array_equal(env.command_v, baseline.command_v)
            np.testing.assert_array_equal(env.last_action, baseline.last_action)
            self.assertEqual(env.data.time, baseline.data.time)
            np.testing.assert_allclose(obs[15:18], (env.target-env.position())*5, rtol=1e-6)
        self.assertEqual(env.target_index, 1)  # No phantom third goal at the horizon.

    def test_schedule_reproducible_and_goals_distinct(self):
        first = make_schedule(510000, 5)
        np.testing.assert_array_equal(first, make_schedule(510000, 5))
        self.assertTrue((np.linalg.norm(np.diff(first, axis=0), axis=1) >= 0.05).all())

    def test_invalid_target_does_not_mutate_state(self):
        env = Zu5ReachEnv()
        env.reset(seed=42)
        target = env.target.copy()
        for value in ([0, 0], [0, float("nan"), 0]):
            with self.assertRaises(ValueError):
                env.set_target(value)
            np.testing.assert_array_equal(env.target, target)


if __name__ == "__main__":
    unittest.main()
