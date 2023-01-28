from gymnasium.envs.registration import register

from .envs.missile_sim import MissileSim
from .guidance import (
    lqr_gain,
    lqr_guidance,
    optimal_guidance_policy,
    pronav_policy,
)

register(id="missile_sim-v0", entry_point="rl_missile_sim.envs:MissileSim")

__all__ = [
    "MissileSim",
    "pronav_policy",
    "optimal_guidance_policy",
    "lqr_guidance",
    "lqr_gain",
]
