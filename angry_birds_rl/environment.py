import math
from typing import Any, Dict, Tuple

import gymnasium as gym
from gymnasium import spaces
import numpy as np


class AngryBirdsEnv(gym.Env):
    """
    간단한 Angry Birds 스타일 투사체 발사 환경.

    - state: [pig_x, pig_y, nominal_max_range]
    - action: angle x power 이산 조합
    - reward: 탄착점과 타겟 사이 거리 기반
    """

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(
        self,
        pig_x_range: Tuple[float, float] = (5.0, 20.0),
        pig_y_range: Tuple[float, float] = (0.0, 5.0),
        gravity: float = 9.81,
        hit_radius: float = 1.0,
        n_angle_bins: int = 12,
        n_power_bins: int = 8,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()

        self.pig_x_range = pig_x_range
        self.pig_y_range = pig_y_range
        self.g = gravity
        self.hit_radius = hit_radius
        self.n_angle_bins = n_angle_bins
        self.n_power_bins = n_power_bins
        self.render_mode = render_mode

        # state: [pig_x, pig_y, max_range_rough]
        low = np.array(
            [pig_x_range[0], pig_y_range[0], 0.0],
            dtype=np.float32,
        )
        high = np.array(
            [pig_x_range[1], pig_y_range[1], 50.0],
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # action: Discrete(n_angle_bins * n_power_bins)
        self.action_space = spaces.Discrete(self.n_angle_bins * self.n_power_bins)

        self._rng = np.random.default_rng()
        self._target: Tuple[float, float] | None = None
        self._last_landing: Tuple[float, float] | None = None

    # Gymnasium API
    def reset(
        self,
        seed: int | None = None,
        options: Dict[str, Any] | None = None,
    ):
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        pig_x = self._rng.uniform(*self.pig_x_range)
        pig_y = self._rng.uniform(*self.pig_y_range)
        self._target = (pig_x, pig_y)
        self._last_landing = None

        obs = np.array([pig_x, pig_y, 25.0], dtype=np.float32)
        info: Dict[str, Any] = {}
        return obs, info

    def _decode_action(self, action: int) -> Tuple[float, float]:
        angle_idx = action // self.n_power_bins
        power_idx = action % self.n_power_bins

        angle_min_deg, angle_max_deg = 15.0, 75.0
        power_min, power_max = 0.4, 1.0

        if self.n_angle_bins > 1:
            angle_deg = angle_min_deg + (angle_max_deg - angle_min_deg) * (
                angle_idx / (self.n_angle_bins - 1)
            )
        else:
            angle_deg = (angle_min_deg + angle_max_deg) / 2.0

        if self.n_power_bins > 1:
            power = power_min + (power_max - power_min) * (
                power_idx / (self.n_power_bins - 1)
            )
        else:
            power = (power_min + power_max) / 2.0

        return math.radians(angle_deg), float(power)

    def _simulate_projectile(self, theta: float, power: float) -> Tuple[float, float]:
        """
        단순 포물선 운동 시뮬레이션.
        새는 (0, 0)에서 발사되고 지면(y=0)에 떨어질 때까지의 탄착점 (x_land, 0)을 계산.
        """
        v0 = 10.0 * power
        vx = v0 * math.cos(theta)
        vy = v0 * math.sin(theta)

        if self.g <= 0 or vy <= 0:
            return 0.0, 0.0

        # y(t) = vy * t - 0.5 * g * t^2 = 0 -> t = 2 * vy / g
        t_flight = 2.0 * vy / self.g
        x_land = vx * t_flight
        y_land = 0.0
        return x_land, y_land

    def step(self, action: int):
        assert self._target is not None, "Call reset() before step()."

        theta, power = self._decode_action(action)
        x_land, y_land = self._simulate_projectile(theta, power)
        self._last_landing = (x_land, y_land)

        pig_x, pig_y = self._target
        dx = x_land - pig_x
        dy = y_land - pig_y
        dist = math.sqrt(dx * dx + dy * dy)

        # reward: hit_radius 이내면 +1, 아니면 거리-based penalty
        if dist <= self.hit_radius:
            reward = 1.0
        else:
            max_dist = 30.0
            reward = -min(dist / max_dist, 1.0)

        terminated = True  # 한 발 쏘면 종료
        truncated = False
        obs = np.array([pig_x, pig_y, 25.0], dtype=np.float32)
        info = {
            "target": (pig_x, pig_y),
            "landing": (x_land, y_land),
            "distance": dist,
        }
        return obs, reward, terminated, truncated, info

    def render(self):
        if self._target is None:
            print("Environment not reset.")
            return

        pig_x, pig_y = self._target
        if self._last_landing is None:
            print(f"Target at ({pig_x:.2f}, {pig_y:.2f}), not fired yet.")
        else:
            lx, ly = self._last_landing
            print(
                f"Target at ({pig_x:.2f}, {pig_y:.2f}), "
                f"landing at ({lx:.2f}, {ly:.2f})"
            )