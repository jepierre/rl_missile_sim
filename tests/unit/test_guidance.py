import unittest
import numpy as np
from rl_missile_sim.envs.missile_sim import MissileSim
from rl_missile_sim.guidance import (
    lqr_gain,
    lqr_guidance,
    optimal_guidance_policy,
    pronav_policy,
)
import rl_missile_sim.eval_train as eval_train


class TestGuidancePolicies(unittest.TestCase):
    def setUp(self):
        self.env = MissileSim({"render_mode": None})
        self.obs, _ = self.env.reset(seed=123)

    def test_pronav_policy(self):
        # Action is N' * V_M * lambda_dot
        action = pronav_policy(self.obs, nav_constant=4.0)
        expected = float(4.0 * self.obs[12] * self.obs[16])
        self.assertAlmostEqual(action, expected, places=5)
        self.assertIsInstance(action, float)
        self.assertFalse(np.isnan(action))

    def test_lqr_guidance(self):
        missile_pos = self.obs[:2]
        missile_vel = self.obs[3:5]
        target_pos = self.obs[6:8]
        target_vel = self.obs[9:11]

        acc = lqr_guidance(missile_pos, missile_vel, target_pos, target_vel)
        self.assertIsInstance(acc, float)
        self.assertFalse(np.isnan(acc))

        # When relative state is zero, command should be 0
        zero_acc = lqr_guidance(missile_pos, missile_vel, missile_pos, missile_vel)
        self.assertAlmostEqual(zero_acc, 0.0, places=5)

    def test_optimal_guidance_policy(self):
        acc = optimal_guidance_policy(self.obs)
        self.assertIsInstance(acc, float)
        self.assertFalse(np.isnan(acc))

        # Co-located, co-velocity test
        obs_colocated = self.obs.copy()
        obs_colocated[6:8] = obs_colocated[0:2]
        obs_colocated[9:11] = obs_colocated[3:5]
        zero_acc = optimal_guidance_policy(obs_colocated)
        self.assertAlmostEqual(zero_acc, 0.0, places=5)

    def test_eval_train_reexports(self):
        # Verify backwards compatibility of imports from eval_train
        self.assertTrue(callable(eval_train.pronav_policy))
        self.assertTrue(callable(eval_train.lqr_guidance))
        self.assertTrue(callable(eval_train.optimal_guidance_policy))
        self.assertTrue(callable(eval_train.evaluate))
        self.assertTrue(callable(eval_train.train))
        self.assertTrue(callable(eval_train.train_ppo))
        self.assertTrue(callable(eval_train.init_seed))
        self.assertTrue(callable(eval_train.make_wrapped_env))
        self.assertTrue(hasattr(eval_train, "TensorboardRewardLoggerCallback"))


if __name__ == "__main__":
    unittest.main()
