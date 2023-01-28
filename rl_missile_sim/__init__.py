from gym.envs.registration import register

register(id="missile_sim-v0", entry_point="rl_missile_sim.envs:MissileSim")
