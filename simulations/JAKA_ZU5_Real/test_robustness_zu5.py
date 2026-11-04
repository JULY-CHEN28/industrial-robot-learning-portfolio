import unittest

import numpy as np

from continuous_zu5 import ContinuousReachEnv
from robustness_zu5 import Disturbance, DisturbedReachEnv
from zu5_env import CONFIG, HOME


class RobustnessTests(unittest.TestCase):
    def test_nominal_wrapper_matches_clean_simulation(self):
        clean = ContinuousReachEnv(windows=2, window_steps=4)
        wrapped = DisturbedReachEnv(2, 4, Disturbance())
        first, _ = clean.reset(seed=41)
        second, _ = wrapped.reset(seed=41)
        np.testing.assert_array_equal(first, second)
        for _ in range(8):
            a, ra, ta, xa, _ = clean.step(np.full(6, 0.1))
            b, rb, tb, xb, _ = wrapped.step(np.full(6, 0.1))
            np.testing.assert_array_equal(a, b)
            self.assertEqual((ra, ta, xa), (rb, tb, xb))

    def test_command_delay_and_reset_flush(self):
        env = DisturbedReachEnv(2, 4, Disturbance(command_delay_steps=2))
        env.reset(seed=41)
        hold = (env.data.qpos-HOME)/CONFIG["action_scale_rad"]
        action = np.full(6, 0.2)
        for _ in range(2):
            env.step(action)
            np.testing.assert_array_equal(env.last_action, hold)
        env.step(action)
        np.testing.assert_array_equal(env.last_action, action)
        env.reset(seed=42)
        hold = (env.data.qpos-HOME)/CONFIG["action_scale_rad"]
        env.step(action)
        np.testing.assert_array_equal(env.last_action, hold)

    def test_observation_delay_keeps_new_goal_and_true_score(self):
        env = DisturbedReachEnv(2, 2, Disturbance(observation_delay_steps=1, target_bias_x_m=0.03))
        initial, _ = env.reset(seed=42)
        previous_tcp = initial[12:15].copy()
        for _ in range(2):
            obs, _, _, _, info = env.step(np.ones(6)*0.2)
            np.testing.assert_array_equal(obs[12:15], previous_tcp)
            np.testing.assert_allclose(obs[15:18],
                                       (env.target+np.array([0.03, 0, 0])-previous_tcp)*5, atol=1e-7)
            previous_tcp = env.position().astype(np.float32)
            # True score uses the executed target; do not use biased policy inputs.
            executed_goal = env.goals[info["target_index"]]
            self.assertAlmostEqual(info["position_error_m"], np.linalg.norm(executed_goal-env.position()))

    def test_mass_mismatch_compensation_stays_nominal(self):
        env = DisturbedReachEnv(1, 2, Disturbance(mass_scale=1.2))
        env.reset(seed=42)
        np.testing.assert_allclose(env.bias_force()*1.2, env.data.qfrc_bias, atol=1e-10)

    def test_invalid_assumptions_rejected(self):
        for options in ({"tcp_noise_m": -1}, {"mass_scale": 0},
                        {"command_delay_steps": 0.5}, {"stiffness_scale": np.nan}):
            with self.assertRaises(ValueError):
                Disturbance(**options)


if __name__ == "__main__":
    unittest.main()
