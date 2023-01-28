import unittest

# import context

from stable_baselines3.common.env_checker import check_env
from rl_missile_sim.envs.missile_sim import MissileSim
import numpy as np


class TestMissileSimEnvCheck(unittest.TestCase):
    def setUp(self):
        self.env = MissileSim()

    def test_env_compliance(self):
        self.env = MissileSim({"norm_env": True})
        # This will raise an error if the environment does not comply with Gym API
        check_env(self.env, warn=True)

    def test_observation_space(self):
        obs, info = self.env.reset()
        self.assertIsInstance(obs, np.ndarray, "Observation should be a numpy array")
        if self.env.norm_env:
            self.assertEqual(
                len(obs), self.env.num_states, "Observation should have 5 elements"
            )
        for ob in obs:
            self.assertIsInstance(
                ob, np.float32, "Each observation element should be a np.float"
            )

    def test_reward(self):
        obs, info = self.env.reset()
        action = self.env.action_space.sample()
        obs, rew, term, trunc, info = self.env.step(action)
        self.assertIsInstance(rew, (float), "Reward must be float")

    def test_render_missile_sim(self):
        env = MissileSim({"render_mode": "human"})
        obs, info = env.reset()
        done = (False,)

        for _ in range(500):
            action = env.action_space.sample()
            obs, reward, done, _, info = env.step(action)
            env.render()
            if done:
                print("Episode finished. Resetting environment.")
                obs, info = env.reset()
        env.close()


if __name__ == "__main__":
    unittest.main()
