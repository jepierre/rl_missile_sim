import unittest
import context

from rl_missile_sim.envs.missile_sim import MissileSim


class TestMissileSim(unittest.TestCase):
    def setUp(self):
        self.env = MissileSim()

    def test_init(self):
        obs = self.env.reset()


if __name__ == "__main__":
    unittest.main()
