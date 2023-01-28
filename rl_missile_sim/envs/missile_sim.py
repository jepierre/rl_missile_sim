import gym
import numpy as np

from gym.utils import seeding


class Entity:
    def __init__(self, N=0, E=0, psi=0, V=0, Nv=0, Ev=0, dt=0.01):
        self.N = N
        self.E = E
        self.psi = psi
        self.V = V
        self.Nv = Nv
        self.Ev = Ev
        self.dt = dt

    def step(self, action):
        Na = action[0]
        Ea = action[1]

        self.Nv += Na * self.dt
        self.Ev += Ea * self.dt

        self.N += self.Nv * self.dt
        self.E += self.Ev * self.dt

        self.psi = np.arctan2(self.Ev, self.Nv)

        self.V = np.sqrt(self.Nv**2 + self.Ev**2)


class MissileSim(gym.Env):
    def __init__(self, env_config={}):
        super().__init__()
        self.max_sim_time = env_config.get("max_sim_time", 1250)
        self.dt = env_config.get("dt", 0.4)
        self.collision_rad = env_config.get("collision_rad", 0.25)
        self.max_num_iteration = self.max_sim_time / self.dt

    def seed(self, seed=None):
        """ Random value to seed"""
        np.random.seed(seed)

        seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        # init pursuer
        self.pursuer = Entity(N=0, E=0, psi=0, V=1.0, dt=self.dt)
        self.pursuer.Nv = self.pursuer.V * np.cos(self.pursuer.psi)
        self.pursuer.Ev = self.pursuer.V * np.sin(self.pursuer.psi)

        # init evader
        self.target = Entity(N=250, E=250, Nv=0.2, Ev=0.5, dt=self.dt)
        self.target.V = np.sqrt(self.target.Nv**2 + self.target.Ev**2)

        self.rtp_last = np.array(
            [self.target.N - self.pursuer.N, self.target.E - self.pursuer.E]
        )

        self.num_iteration = 0
        return self._get_obs()

    def step(self, action):
        pNa = -action * np.sin(self.pursuer.psi)
        pEa = action * np.cos(self.pursuer.psi)

        self.pursuer.step(np.array([pNa, pEa]))
        self.target.step(np.array([0, 0]))
        self.num_iteration += 1

        obs = self._get_obs()
        reward = self._get_reward()
        done = self._get_done()
        info = self._get_info()

        return obs, reward, done, info

    def _get_obs(self):
        rtp = np.array([self.target.N - self.pursuer.N, self.target.E - self.pursuer.E])

        self.r = np.linalg.norm(rtp)

        rtp_dot = (rtp - self.rtp_last) / self.dt

        self.r_dot = np.linalg.norm(rtp_dot)

        self.l = np.arctan2(rtp[1], rtp[0])

        self.l_dot = (rtp[0] * rtp_dot[1] - rtp[1] * rtp_dot[0]) / self.r**2

        return np.array(
            [
                self.pursuer.N,
                self.pursuer.E,
                self.pursuer.V,
                self.r,
                self.r_dot,
                self.l,
                self.l_dot,
            ]
        )

    def _get_info(self):
        info = {
            "t": self.num_iteration * self.dt,
            "pN": self.pursuer.N,
            "pE": self.pursuer.E,
            "pNv": self.pursuer.Nv,
            "pEv": self.pursuer.Ev,
            "p_psi": self.pursuer.psi,
            "tN": self.target.N,
            "tE": self.target.E,
            "tNv": self.target.Nv,
            "tEv": self.target.Ev,
            "r": self.r,
            "l": self.l,
            "r_dot": self.r_dot,
            "l_dot": self.l_dot,
        }

        return info

    def _get_done(self):
        return (
            self.num_iteration >= self.max_num_iteration or self.r < self.collision_rad
        )

    def _get_reward(self):
        return 0

    def render(self, mode="human"):
        pass

    def close(self):
        pass
