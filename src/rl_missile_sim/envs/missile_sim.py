import gymnasium as gym
from gymnasium import spaces
from gymnasium.error import DependencyNotInstalled

import matplotlib

# matplotlib.use("Qt5Agg")
matplotlib.use("TKAgg")
import matplotlib.pyplot as plt
import numpy as np
import math
import logging

logger = logging.getLogger(__name__)


class Entity:
    def __init__(self, N=0, E=0, psi=0, V=0, rad=1, dt=0.1):
        self.N = N
        self.E = E
        self.psi = psi
        self.V = V
        self.rad = rad  # radius of the entity, used for collision detection

        self.Nv = self.V * math.cos(self.psi)
        self.Ev = self.V * math.sin(self.psi)
        self.dt = dt

        np.testing.assert_almost_equal(
            self.psi, math.atan2(self.Ev, self.Nv), decimal=2
        )
        self._state = np.array(
            [self.N, self.E, self.psi, self.Nv, self.Ev], dtype=np.float32
        )
        self.done = False

    @property
    def state(self):
        self._state = np.array(
            [self.N, self.E, self.psi, self.Nv, self.Ev], dtype=np.float32
        )
        return self._state

    @property
    def pos(self):
        return np.array([self.N, self.E], dtype=np.float32)

    @property
    def vel(self):
        return np.array([self.Nv, self.Ev], dtype=np.float32)

    def step(self, action=0):
        self.psi += (action / self.V) * self.dt

        # wraparound psi to -pi and pi
        self.psi = (self.psi + np.pi) % (2 * np.pi) - np.pi

        self.Nv = self.V * math.cos(self.psi)
        self.Ev = self.V * math.sin(self.psi)

        self.N += self.Nv * self.dt
        self.E += self.Ev * self.dt


class MissileSim(gym.Env):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30,
    }

    def __init__(self, env_config={}):
        super().__init__()
        self.max_sim_time = env_config.get("max_sim_time", 20)  # seconds
        # --- Engagement area: set to max missile range (20 km) in 2D ---
        self.max_engagement_range = 20_000  # 20,000 meters
        self.max_speed = 1700  # m/s
        self.min_speed = self.max_speed * 0.7  # m/s
        self.max_t_speed = 120  # m/s
        self.min_t_speed = 80  # m/s
        self.max_gamma = np.deg2rad(80)
        self.min_gamma = np.deg2rad(10)

        self.dt = env_config.get("dt", 0.1)
        self.collision_rad = env_config.get("collision_rad", 100)  # m
        self._seed = env_config.get("seed", None)
        self.render_mode = env_config.get("render_mode", "human")
        self.fig = None
        self.screen = None
        self.clock = None
        self.max_num_iteration = self.max_sim_time / self.dt
        self.norm_env = env_config.get("norm_env", False)

        self.use_lidar = env_config.get("use_lidar", True)

        self.num_beams = env_config.get("num_beams", 16)
        self.lidar_range = env_config.get("lidar_range", 10000)  # meters

        self.lidar_fov = env_config.get("lidar_fov", np.pi)  # 180 degrees

        # if self.norm_env:
        # print('using normalize env')

        self.g = env_config.get("g", 9.81)  # Gravity, not used in this simulation

        # R-37 missile max normal acceleration (22g)
        # TODO: this is not realistic
        self.amax = env_config.get("max_accel", 200 * self.g)  # ≈ 216 m/s^2
        self.amin = -self.amax

        # Action space is now normalized: [-1, 1]
        self.action_space = spaces.Box(
            low=self.amin, high=self.amax, shape=(1,), dtype=np.float32
        )
        self.observation_space = self._get_observation_space()

        self.reset(seed=self._seed)

    def unnormalize_action(self, norm_action):
        """
        Convert normalized action in [-1, 1] to real normal acceleration command.
        """
        # Clip for safety
        norm_action = np.clip(norm_action, -1.0, 1.0)
        return norm_action * self.amax

    def get_random_pos(self):
        """
        Get a random position within the engagement area.
        """
        N = self.np_random.uniform(
            -self.max_engagement_range / 2, self.max_engagement_range / 2
        )
        E = self.np_random.uniform(
            -self.max_engagement_range / 2, self.max_engagement_range / 2
        )
        return N, E

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        # --- R-37 missile parameters (approximate) ---
        r37_mass = 510.0  # kg
        r37_length = 4.2  # m
        r37_diameter = 0.38  # m
        r37_max_speed = 1700.0  # m/s (Mach 5)
        r37_max_g = 22.0  # g
        r37_max_accel = r37_max_g * 9.81  # m/s^2
        r37_max_range = 400000.0  # m (400 km)
        # r37_max_range = 35000.0  # m (35 km) Keeping sim at 35 km
        r37_warhead_mass = 60.0  # kg
        r37_rad = r37_diameter / 2

        # --- Target UAV parameters (example: MQ-9 Reaper) ---
        uav_mass = 2223.0  # kg
        uav_length = 11.0  # m
        uav_wingspan = 20.0  # m
        uav_max_speed = 120.0  # m/s
        uav_max_speed = self.max_t_speed
        uav_cruise_speed = 80.0  # m/s
        uav_cruise_speed = self.min_t_speed
        uav_rad = 2.0  # m (approximate)

        # initialize the target
        # Target (UAV) initial position and velocity within engagement area
        # target_N = np.random.uniform(
        #     -self.max_engagement_range / 2, self.max_engagement_range / 2
        # )
        # target_E = np.random.uniform(
        #     -self.max_engagement_range / 2, self.max_engagement_range / 2
        # )

        target_N, target_E = self.get_random_pos()

        target_psi = self.np_random.uniform(-self.min_gamma, self.max_gamma)
        target_V = self.np_random.uniform(uav_cruise_speed, uav_max_speed)
        self.target = Entity(
            N=target_N,
            E=target_E,
            psi=target_psi,
            V=target_V,
            rad=uav_rad,
            dt=self.dt,
        )

        self.obstacles = []
        self.obstacles = [
            (2000.0, 2000.0, 2000.0),
            (1000.0, -5000.0, 1000.0),
            (-5000.0, 7000.0, 2000.0),
            (5000.0, -7000.0, 2000.0),
        ]  # (x, y, radius)

        # Randomize pursuer (missile) start position and velocity within engagement area
        # pursuer_N = np.random.uniform(
        #     -self.max_engagement_range / 2, self.max_engagement_range / 2
        # )
        # pursuer_E = np.random.uniform(
        #     -self.max_engagement_range / 2, self.max_engagement_range / 2
        # )
        pursuer_N, pursuer_E = self.get_random_pos()
        pursuer_psi = self.np_random.uniform(-self.min_gamma, self.max_gamma)
        pursuer_V = self.np_random.uniform(self.max_speed * 0.7, self.max_speed)
        self.pursuer = Entity(
            N=pursuer_N,
            E=pursuer_E,
            psi=pursuer_psi,
            V=pursuer_V,
            rad=r37_rad,
            dt=self.dt,
        )

        self.total_reward = 0
        self.collision_count = 0
        self.action = 0.0

        self.rtp_last = np.array(
            [self.target.N - self.pursuer.N, self.target.E - self.pursuer.E]
        )
        self.num_iteration = 0
        obs = self._get_obs(init=True)
        self.last_r = self.r
        info = self._get_info()
        return obs, info

    def check_bounds(self, entity):
        entity.N = min(
            max(-self.max_engagement_range / 2, entity.N), self.max_engagement_range / 2
        )
        entity.E = min(
            max(-self.max_engagement_range / 2, entity.E), self.max_engagement_range / 2
        )
        return entity

    def step(self, action):
        if isinstance(action, np.ndarray):
            action = action.squeeze()

        if self.norm_env:
            # Unnormalize action from [-1, 1] to [-amax, amax]
            action = self.unnormalize_action(action)

        action = min(max(action, self.amin), self.amax)
        self.action = action
        self.pursuer.step(action)
        self.pursuer = self.check_bounds(self.pursuer)

        self.target.step()
        self.target = self.check_bounds(self.target)

        self.num_iteration += 1

        obs = self._get_obs()
        reward = self._get_reward()
        terminated = bool(self._get_done())
        info = self._get_info()
        truncated = bool(
            self.num_iteration >= self.max_num_iteration
            or self.r >= self.max_engagement_range
        )
        return obs, reward, terminated, truncated, info

    def _get_observation_space(self):
        # Define realistic bounds for normalization
        # These should reflect the largest expected values in the scenario
        # max_range = 400_000.0  # 400 km
        max_range = self.max_engagement_range / 2
        # max_speed = 1700.0  # m/s (missile max speed)
        max_speed = self.max_speed
        max_angle = np.pi
        max_omega = self.amax / self.max_speed * 5  # rad /s
        max_omega = np.pi
        max_rad = 20.0  # max entity radius (meters)

        max_val = np.array(
            [
                # pursuer state: N, E, psi, Nv, Ev
                max_range,
                max_range,
                max_angle,
                max_speed,
                max_speed,
                max_rad,
                # target state: N, E, psi, Nv, Ev
                max_range,
                max_range,
                max_angle,
                self.max_t_speed,
                self.max_t_speed,
                max_rad,
                # pursuer.V, r, r_dot, l, l_dot
                max_speed,
                self.max_engagement_range * 2,
                # TODO: confirm this is accurate
                max_speed * 4,
                max_angle,
                max_omega,
                # max_speed / max_range,
            ],
            dtype=np.float32,
        )
        lidar_obs_dim = self.num_beams if self.use_lidar else 0

        if self.norm_env:
            max_val = np.array([max_speed, max_range, max_speed, max_angle, max_omega])
            max_val = np.array([max_range, max_speed, max_angle, max_omega])

        min_val = -max_val

        if self.use_lidar:
            max_val = np.concatenate(
                [
                    max_val,
                    np.ones(lidar_obs_dim, dtype=np.float32) * self.lidar_range,
                ],
            )
            min_val = np.concatenate(
                [
                    min_val,
                    np.zeros(lidar_obs_dim, dtype=np.float32),
                ],
            )
        # min_val[[3, 4, 12]] = self.min_speed
        # min_val[[9, 10]] = self.min_t_speed

        self.num_states = len(max_val)
        self._obs_max = max_val
        self._obs_min = -max_val
        return spaces.Box(
            # low=-np.ones_like(max_val, dtype=np.float32),
            # high=np.ones_like(max_val, dtype=np.float32),
            low=min_val,
            high=max_val,
            shape=(self.num_states,),
            dtype=np.float32,
        )

    def _get_obs(self, init=False):
        """
        https://unacademy.com/content/jee/study-material/physics/relative-angular-velocity/
        https://arc.aiaa.org/doi/pdf/10.2514/1.G000082?casa_token=axBX6NooUM0AAAAA:Ta93x5ueJBpgQhmySyHG1HyC-2irJ7PpgmuOGzKBu_O2JL-73tj07ko3N5QRStDsxGNVS1_W2Bt_RA
        Get the observation for the pursuer and target."""
        rel_pos = self.target.pos - self.pursuer.pos
        rel_vel = self.target.vel - self.pursuer.vel

        self.r = np.linalg.norm(rel_pos)

        # self.r = np.clip(self.r, 0, self.max_engagement_range / 2)
        if self.r == 0:
            range_rate = 0.0
            los_angle = 0.0
            los_unit = np.zeros_like(rel_pos)
            los_rate = 0.0
        else:
            los_unit = rel_pos / self.r
            # closing_vel = -np.dot(rel_pos, rel_vel / self.r)# / self.r
            # closing_vel = -np.dot(rel_vel, los_unit)# / self.r
            range_rate = np.dot(rel_vel, rel_pos) / self.r
            los_angle = math.atan2(rel_pos[1], rel_pos[0]) - self.pursuer.psi
            los_angle = (los_angle + np.pi) % (2 * np.pi) - np.pi

            # LOS rate (approximate derivative of los angle)
            los_rate_vec = (rel_vel - np.dot(rel_vel, los_unit) * los_unit) / self.r

            # Normal acceleration is perpendicular to velocity, so take the cross product (2D)
            # In 2D, the z-component of the cross product gives the normal acceleration direction
            # los_rate = los_rate_vec[0] * los_unit[1] - los_rate_vec[1] * los_unit[0]

            los_rate = np.cross(rel_pos, rel_vel) / self.r**2
            los_rate = (los_rate + np.pi) % (2 * np.pi) - np.pi

        # closing velocity
        self.r_dot = -range_rate
        self.l = los_angle
        self.l_dot = los_rate
        # self.r_dot = self.target.V * math.cos(
        #     self.target.psi - self.l
        # ) - self.pursuer.V * math.cos(self.pursuer.psi - self.l)

        # self.l_dot = (
        #     self.target.V * math.sin(self.target.psi - self.l)
        #     - self.pursuer.V * math.sin(self.pursuer.psi - self.l) / self.r
        # )
        # # print(f"rtp shape: {rtp}, {rtp.shape}")
        # rtp_dot = (rtp - self.rtp_last) / self.dt
        # # print(f"rtp_dot shape: {rtp_dot}, {rtp_dot.shape}")

        # self.r_dot = np.linalg.norm(rtp_dot)

        # self.l_dot = ((rtp[0] * rtp_dot[1]) - (rtp[1] * rtp_dot[0])) / self.r**2

        # self.rtp_last = rtp
        if init:
            self.r0 = self.r
            self.l0 = self.l
            self.r_dot0 = self.r_dot
            self.l_dot0 = self.l_dot
            self._z0 = self._cal_z()
            self.min_distance_to_target = self.r

        self.min_distance_to_target = min(self.r, self.min_distance_to_target)
        obs = np.concatenate(
            [
                self.pursuer.state,
                np.array([self.pursuer.rad], dtype=np.float32),
                self.target.state,
                np.array([self.target.rad], dtype=np.float32),
                np.array(
                    [
                        self.pursuer.V,
                        self.r,
                        self.r_dot,
                        self.l,
                        self.l_dot,
                    ],
                    dtype=np.float32,
                ),
            ],
            dtype=np.float32,
        )

        if self.use_lidar:
            lidar = self._simulate_lidar()
        # if self.normalize_obs:
        # lidar = lidar / self.lidar_range
        obs = np.concatenate([obs, lidar.astype(np.float32)])

        if self.norm_env:
            # Normalize observation to [-1, 1]
            obs = np.clip(obs[-4:] / self._obs_max, -1.0, 1.0, dtype=np.float32)
        return obs

    def _simulate_lidar(self):
        readings = np.ones(self.num_beams, dtype=np.float32) * self.lidar_range
        pursuer_state = self.pursuer.state
        origin = pursuer_state[:2]  # N, E
        # origin = self.m_pos
        pursuer_heading = pursuer_state[2]  # psi
        angle_start = pursuer_heading - self.lidar_fov / 2

        angles = angle_start + np.linspace(0, self.lidar_fov, self.num_beams)

        for i, angle in enumerate(angles):
            ray_dir = np.array([np.cos(angle), np.sin(angle)])
            for ox, oy, r in self.obstacles:
                center = np.array([ox, oy])
                oc = center - origin

                # Ray-circle intersection (quadratic)
                proj = np.dot(oc, ray_dir)
                closest = origin + proj * ray_dir
                dist_sq = np.sum((closest - center) ** 2)
                if dist_sq < r**2 and proj > 0:
                    offset = np.sqrt(r**2 - dist_sq)
                    hit_dist = proj - offset
                    if 0 < hit_dist < readings[i]:
                        readings[i] = hit_dist

        return readings

    def _get_info(self):
        info = {
            "t": self.num_iteration * self.dt,
            # "pN": self.pursuer.N,
            # "pE": self.pursuer.E,
            # "pNv": self.pursuer.Nv,
            # "pEv": self.pursuer.Ev,
            # "p_psi": self.pursuer.psi,
            # "tN": self.target.N,
            # "tE": self.target.E,
            # "tNv": self.target.Nv,
            # "tEv": self.target.Ev,
            "r": self.r,
            "l": self.l,
            "r_dot": self.r_dot,
            "l_dot": self.l_dot,
            "total_reward": self.total_reward,
            "min_distance_to_target": self.min_distance_to_target,
            "collision_count": self.collision_count,
        }

        return info

    def _get_done(self):
        return (
            (self.num_iteration >= self.max_num_iteration)
            or (self.r <= self.collision_rad)
            or self.pursuer.done
        )

    def _cal_z(self):
        return (
            self.r**2
            * self.l_dot
            / math.sqrt(self.r_dot**2 + (self.r * self.l_dot) ** 2)
        )

    def _get_reward(self):
        reward = 0
        z = self._cal_z()
        # reward += -0.5 * (z / abs(self._z0)) ** 2
        reward += 10 if self.r <= self.collision_rad else 0
        # reward += float(min(0, 0.03 * np.sign(self.last_r - self.r)))
        reward += -0.03 * (self.r / self.max_engagement_range) ** 2
        # reward += -0.03 * np.exp(-self.r / self.max_engagement_range)
        reward += -0.02 * (self.action / (self.amax)) ** 2
        # reward += -0.02 * np.exp(-((self.action / self.amax) ** 2))
        reward += -0.01 if self.r_dot < 0 else 0

        for ox, oy, r in self.obstacles:
            # Check if pursuer is within obstacle radius
            if np.linalg.norm([self.pursuer.N - ox, self.pursuer.E - oy]) < (
                r + self.pursuer.rad
            ):
                self.pursuer.done = True
                self.collision_count += 1
                reward -= 10
                # break

        self.last_r = self.r
        reward = float(reward)
        self.total_reward += reward
        return reward

    def render(self):
        if self.render_mode is None:
            env_id = self.spec.id if self.spec is not None else self.__class__.__name__
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gymnasium.make("{env_id}", render_mode="rgb_array")'
            )
            return

        try:
            import pygame
        except ImportError:
            raise DependencyNotInstalled(
                "pygame is not installed; install it with `uv sync --extra render`"
            )

        width, height = 1100, 720
        map_rect = pygame.Rect(24, 24, 672, 672)
        sidebar_x = 728
        sidebar_width = width - sidebar_x - 24
        scale = map_rect.width / self.max_engagement_range
        half_range = self.max_engagement_range / 2
        colors = {
            "background": (9, 15, 27),
            "map": (13, 23, 37),
            "grid": (29, 45, 62),
            "axis": (49, 73, 92),
            "panel": (17, 28, 43),
            "border": (36, 54, 72),
            "text": (225, 235, 245),
            "muted": (133, 153, 173),
            "cyan": (70, 202, 224),
            "missile": (255, 105, 91),
            "target": (92, 221, 157),
            "warning": (255, 190, 92),
            "danger": (255, 103, 111),
        }

        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                pygame.display.set_caption("Missile Simulation | Engagement Monitor")
                self.screen = pygame.display.set_mode((width, height))
            else:
                self.screen = pygame.Surface((width, height))

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

        if self.clock is None:
            self.clock = pygame.time.Clock()

        self.surf = pygame.Surface((width, height))
        self.surf.fill(colors["background"])
        title_font = pygame.font.SysFont("segoeui", 23, bold=True)
        section_font = pygame.font.SysFont("segoeui", 12, bold=True)
        body_font = pygame.font.SysFont("segoeui", 15)
        small_font = pygame.font.SysFont("segoeui", 12)
        metric_font = pygame.font.SysFont("segoeui", 27, bold=True)

        def draw_text(text, font, color, position):
            self.surf.blit(font.render(str(text), True, color), position)

        def to_screen(north, east):
            x = int(map_rect.left + (north + half_range) * scale)
            y = int(map_rect.top + (half_range - east) * scale)
            return x, y

        pygame.draw.rect(self.surf, colors["map"], map_rect, border_radius=8)
        self.surf.set_clip(map_rect)
        for coordinate in range(-int(half_range), int(half_range) + 1, 2500):
            x, _ = to_screen(coordinate, 0)
            _, y = to_screen(0, coordinate)
            grid_color = colors["axis"] if coordinate == 0 else colors["grid"]
            pygame.draw.line(
                self.surf, grid_color, (x, map_rect.top), (x, map_rect.bottom), 1
            )
            pygame.draw.line(
                self.surf, grid_color, (map_rect.left, y), (map_rect.right, y), 1
            )

        draw_text(
            "NORTH / m",
            small_font,
            colors["muted"],
            (map_rect.left + 8, map_rect.top + 8),
        )
        draw_text(
            "EAST / m",
            small_font,
            colors["muted"],
            (map_rect.left + 8, map_rect.bottom - 24),
        )
        pursuer_x, pursuer_y = to_screen(self.pursuer.N, self.pursuer.E)
        target_x, target_y = to_screen(self.target.N, self.target.E)

        if self.use_lidar:
            lidar_readings = self._simulate_lidar()
            denominator = max(1, self.num_beams - 1)
            for index, reading in enumerate(lidar_readings):
                beam_angle = (
                    self.pursuer.psi
                    - self.lidar_fov / 2
                    + (index / denominator) * self.lidar_fov
                )
                endpoint = to_screen(
                    self.pursuer.N + reading * math.cos(beam_angle),
                    self.pursuer.E + reading * math.sin(beam_angle),
                )
                pygame.draw.line(
                    self.surf,
                    (30, 105, 123),
                    (pursuer_x, pursuer_y),
                    endpoint,
                    1,
                )
                if reading < self.lidar_range:
                    pygame.draw.circle(self.surf, colors["warning"], endpoint, 3)

        for north, east, radius in self.obstacles:
            x, y = to_screen(north, east)
            radius_px = max(4, int(radius * scale))
            pygame.draw.circle(self.surf, (57, 47, 39), (x, y), radius_px)
            pygame.draw.circle(self.surf, colors["warning"], (x, y), radius_px, 1)
            pygame.draw.circle(self.surf, (185, 133, 74), (x, y), 3)

        collision_radius_px = max(5, int(self.collision_rad * scale))
        pygame.draw.circle(
            self.surf, colors["target"], (target_x, target_y), collision_radius_px, 1
        )

        for entity_x, entity_y, heading, color, size in (
            (target_x, target_y, self.target.psi, colors["target"], 15),
            (pursuer_x, pursuer_y, self.pursuer.psi, colors["missile"], 19),
        ):
            angle = -heading
            forward = (math.cos(angle), math.sin(angle))
            side = (-forward[1], forward[0])
            tip = (
                entity_x + int(forward[0] * size),
                entity_y + int(forward[1] * size),
            )
            rear_left = (
                entity_x - int(forward[0] * size * 0.65) + int(side[0] * size * 0.55),
                entity_y - int(forward[1] * size * 0.65) + int(side[1] * size * 0.55),
            )
            rear_right = (
                entity_x - int(forward[0] * size * 0.65) - int(side[0] * size * 0.55),
                entity_y - int(forward[1] * size * 0.65) - int(side[1] * size * 0.55),
            )
            pygame.draw.circle(
                self.surf, (19, 31, 43), (entity_x, entity_y), size + 5
            )
            pygame.draw.polygon(self.surf, color, (tip, rear_left, rear_right))
            pygame.draw.circle(self.surf, colors["text"], (entity_x, entity_y), 2)

        if self.r <= self.collision_rad or self.pursuer.done:
            pygame.draw.circle(self.surf, colors["danger"], (pursuer_x, pursuer_y), 29, 2)
            pygame.draw.circle(self.surf, colors["warning"], (pursuer_x, pursuer_y), 38, 1)

        self.surf.set_clip(None)
        pygame.draw.rect(
            self.surf, colors["border"], map_rect, width=1, border_radius=8
        )

        # Telemetry is kept in a dedicated sidebar to leave the map unobstructed.
        draw_text("ENGAGEMENT MONITOR", title_font, colors["text"], (24, 2))
        fps = self.clock.get_fps()

        if self.r <= self.collision_rad:
            status, status_color = "INTERCEPT", colors["target"]
        elif self.pursuer.done:
            status, status_color = "MISSION ENDED", colors["danger"]
        else:
            status, status_color = "TRACKING", colors["cyan"]
        status_surface = section_font.render(status, True, status_color)
        status_rect = status_surface.get_rect(topright=(width - 25, 9))
        pygame.draw.rect(
            self.surf, colors["panel"], status_rect.inflate(20, 12), border_radius=9
        )
        self.surf.blit(status_surface, status_rect)

        def draw_panel(top, panel_height):
            rect = pygame.Rect(sidebar_x, top, sidebar_width, panel_height)
            pygame.draw.rect(self.surf, colors["panel"], rect, border_radius=10)
            pygame.draw.rect(
                self.surf, colors["border"], rect, width=1, border_radius=10
            )

        def draw_metric(label, value, top, color, suffix=""):
            draw_text(label.upper(), section_font, colors["muted"], (sidebar_x + 16, top))
            draw_text(
                f"{value}{suffix}",
                metric_font,
                color,
                (sidebar_x + 16, top + 19),
            )

        draw_panel(24, 140)
        draw_metric("Range to target", f"{self.r / 1000:.2f}", 40, colors["text"], " km")
        range_bar = pygame.Rect(sidebar_x + 16, 125, sidebar_width - 32, 5)
        pygame.draw.rect(self.surf, colors["grid"], range_bar, border_radius=3)
        range_fraction = min(1.0, max(0.0, self.r / self.max_engagement_range))
        pygame.draw.rect(
            self.surf,
            colors["cyan"],
            (
                range_bar.left,
                range_bar.top,
                int(range_bar.width * range_fraction),
                range_bar.height,
            ),
            border_radius=3,
        )

        draw_panel(176, 142)
        closing_color = colors["target"] if self.r_dot > 0 else colors["warning"]
        draw_metric("Closing velocity", f"{self.r_dot:+.1f}", 192, closing_color, " m/s")
        draw_text(
            f"LOS angle   {math.degrees(self.l):+.1f} deg",
            body_font,
            colors["text"],
            (sidebar_x + 16, 253),
        )
        draw_text(
            f"LOS rate      {math.degrees(self.l_dot):+.2f} deg/s",
            body_font,
            colors["text"],
            (sidebar_x + 16, 279),
        )

        draw_panel(330, 148)
        draw_text("VEHICLE STATE", section_font, colors["muted"], (sidebar_x + 16, 346))
        draw_text("PURSUER", section_font, colors["missile"], (sidebar_x + 16, 372))
        draw_text(
            f"{self.pursuer.V:.0f} m/s    heading {math.degrees(self.pursuer.psi):+.1f} deg",
            body_font,
            colors["text"],
            (sidebar_x + 16, 392),
        )
        draw_text("TARGET", section_font, colors["target"], (sidebar_x + 16, 422))
        draw_text(
            f"{self.target.V:.0f} m/s    heading {math.degrees(self.target.psi):+.1f} deg",
            body_font,
            colors["text"],
            (sidebar_x + 16, 442),
        )

        draw_panel(490, 105)
        draw_text("GUIDANCE COMMAND", section_font, colors["muted"], (sidebar_x + 16, 506))
        draw_text(
            f"{self.action:+.1f} m/s^2",
            metric_font,
            colors["missile"],
            (sidebar_x + 16, 525),
        )
        action_bar = pygame.Rect(sidebar_x + 16, 577, sidebar_width - 32, 5)
        pygame.draw.rect(self.surf, colors["grid"], action_bar, border_radius=3)
        center_x = action_bar.centerx
        pygame.draw.line(
            self.surf,
            colors["muted"],
            (center_x, action_bar.top - 3),
            (center_x, action_bar.bottom + 3),
        )
        action_fraction = min(1.0, max(-1.0, self.action / self.amax))
        action_end = center_x + int(action_fraction * (action_bar.width / 2))
        pygame.draw.line(
            self.surf,
            colors["missile"],
            (center_x, action_bar.centery),
            (action_end, action_bar.centery),
            5,
        )

        draw_panel(608, 88)
        draw_text(
            f"STEP  {self.num_iteration:04d}     FPS  {int(fps):02d}",
            section_font,
            colors["text"],
            (sidebar_x + 16, 624),
        )
        draw_text(
            f"SIM TIME  {self.num_iteration * self.dt:.1f} s",
            body_font,
            colors["muted"],
            (sidebar_x + 16, 650),
        )

        for x, color, label in (
            (28, colors["missile"], "PURSUER"),
            (164, colors["target"], "TARGET"),
            (278, colors["warning"], "OBSTACLE"),
            (408, (30, 105, 123), "LIDAR"),
        ):
            pygame.draw.circle(self.surf, color, (x, height - 16), 4)
            draw_text(label, small_font, colors["muted"], (x + 9, height - 24))

        self.screen.blit(self.surf, (0, 0))
        if self.render_mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()
        elif self.render_mode == "rgb_array":
            return np.transpose(
                pygame.surfarray.array3d(self.surf), axes=(1, 0, 2)
            )

    def close(self):
        if hasattr(self, "screen") and self.screen is not None:
            import pygame

            pygame.quit()
        self.screen = None
