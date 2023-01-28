import subprocess
import numpy as np
import gymnasium as gym
from gymnasium import spaces

def get_git_hash(short=True) -> str:
    """
    Returns the current git commit hash for the repository.
    If short is True, returns the short hash (default: True).
    Returns None if not in a git repo or git is not available.
    """
    if short:
        cmd = ["git", "rev-parse", "--short", "HEAD"]
    else:
        cmd = ["git", "rev-parse", "HEAd"]
    return (
        subprocess.check_output(cmd)
        .decode("ascii")
        .strip()
    )


class RescaleAction(gym.ActionWrapper):
    # why normalize env
    """Rescales the continuous action space of the environment to a range [a,b].
    Example::
        >>> RescaleAction(env, a, b).action_space == Box(a,b)
        True
    """

    def __init__(self, env, a, b):
        assert isinstance(
            env.action_space, spaces.Box
        ), "expected Box action space, got {}".format(type(env.action_space))
        assert np.less_equal(a, b).all(), (a, b)
        super(RescaleAction, self).__init__(env)
        self.a = np.zeros(env.action_space.shape, dtype=env.action_space.dtype) + a
        self.b = np.zeros(env.action_space.shape, dtype=env.action_space.dtype) + b
        self.action_space = spaces.Box(
            low=a, high=b, shape=env.action_space.shape, dtype=env.action_space.dtype
        )

    def action(self, action):
        assert np.all(np.greater_equal(action, self.a)), (action, self.a)
        assert np.all(np.less_equal(action, self.b)), (action, self.b)
        low = self.env.action_space.low
        high = self.env.action_space.high
        action = low + (high - low) * ((action - self.a) / (self.b - self.a))
        action = np.clip(action, low, high)
        return action


class NormalizedObservation(gym.ObservationWrapper):
    def __init__(self, env, a, b):
        assert isinstance(
            env.observation_space, spaces.Box
        ), "expected Box action space, got {}".format(type(env.observation_space))
        assert np.less_equal(a, b).all(), (a, b)
        super(NormalizedObservation, self).__init__(env)
        self.a = (
            np.zeros(env.observation_space.shape, dtype=env.observation_space.dtype) + a
        )
        self.b = (
            np.zeros(env.observation_space.shape, dtype=env.observation_space.dtype) + b
        )
        self.observation_space = spaces.Box(
            low=a,
            high=b,
            shape=env.observation_space.shape,
            dtype=env.observation_space.dtype,
        )

    def observation(self, observation):
        idx = np.greater_equal(observation, self.env.observation_space.low)
        assert np.all(idx), (idx,
            observation,
            self.env.observation_space.low,
        )
        idx = np.less_equal(observation, self.env.observation_space.high) 
        assert np.all(idx), (idx,
            observation,
            self.env.observation_space.high,
        )

        low = self.env.observation_space.low
        high = self.env.observation_space.high

        observation = (observation - low) / (high - low)
        observation = np.clip(observation, self.a, self.b)
        return observation


def init_seed(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility across random, numpy, and torch.
    """
    import random

    np.random.seed(seed)
    random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def make_wrapped_env(render_mode=None):
    """
    Factory creating a MissileSim environment wrapped with RescaleAction to [-1, 1]
    and NormalizedObservation to [0, 1].
    """
    from .envs.missile_sim import MissileSim

    return NormalizedObservation(
        RescaleAction(MissileSim({"render_mode": render_mode}), -1, 1), 0, 1
    )
