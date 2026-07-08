"""The chase environment: pure physics, no rendering.

A single *chaser* pursues a single *runner* on a wrap-around (toroidal) square
world. Each agent picks a 2D acceleration in ``[-1, 1]^2`` every step. The
chaser is rewarded for closing the gap and gets a bonus on capture; the runner
is rewarded for opening the gap. Rewards are otherwise zero-sum.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from chaser_runner.config import EnvConfig

STATE_DIM = 8
ACTION_DIM = 2


class ChaseEnv:
    """Toroidal pursuit-evasion environment with continuous acceleration control."""

    def __init__(self, config: EnvConfig | None = None):
        self.cfg = config or EnvConfig()
        self.steps = 0
        self.rng = np.random.default_rng()

        self.ch_pos = np.zeros(2)
        self.ch_vel = np.zeros(2)
        self.rn_pos = np.zeros(2)
        self.rn_vel = np.zeros(2)
        self._prev_dist: float | None = None

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    def reset(self, *, randomize: bool = True, seed: int | None = None) -> np.ndarray:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.steps = 0
        self._prev_dist = None

        size = self.cfg.world_size
        if randomize:
            self.ch_pos = self.rng.uniform(0, size, size=2)
            # Keep the runner far enough away that the episode is not trivial.
            while True:
                self.rn_pos = self.rng.uniform(0, size, size=2)
                if np.linalg.norm(self._torus_vector(self.ch_pos, self.rn_pos)) > size * 0.2:
                    break
            self.ch_vel = self.rng.uniform(-10, 10, size=2)
            self.rn_vel = self.rng.uniform(-10, 10, size=2)
        else:
            self.ch_pos = np.array([size * 0.25, size * 0.5])
            self.rn_pos = np.array([size * 0.75, size * 0.5])
            self.ch_vel = np.zeros(2)
            self.rn_vel = np.zeros(2)

        self._prev_dist = float(np.linalg.norm(self._torus_vector(self.ch_pos, self.rn_pos)))
        return self._get_state()

    def step(
        self, action_chaser: np.ndarray, action_runner: np.ndarray
    ) -> tuple[np.ndarray, tuple[float, float], bool, dict[str, Any]]:
        self.steps += 1
        cfg = self.cfg

        acc_ch = np.clip(action_chaser, -1.0, 1.0).astype(float) * cfg.max_acc
        acc_rn = np.clip(action_runner, -1.0, 1.0).astype(float) * cfg.max_acc

        self.ch_vel = self.ch_vel + acc_ch * cfg.dt
        self.rn_vel = self.rn_vel + acc_rn * cfg.dt

        self.ch_vel *= 1.0 - cfg.friction * cfg.dt
        self.rn_vel *= 1.0 - cfg.friction * cfg.dt

        self.ch_vel = self._clip_speed(self.ch_vel)
        self.rn_vel = self._clip_speed(self.rn_vel)

        self.ch_pos = self._wrap(self.ch_pos + self.ch_vel * cfg.dt)
        self.rn_pos = self._wrap(self.rn_pos + self.rn_vel * cfg.dt)

        disp = self._torus_vector(self.ch_pos, self.rn_pos)
        dist = float(np.linalg.norm(disp))
        prev = self._prev_dist if self._prev_dist is not None else dist

        # Shaped, zero-sum reward: chaser gains what the runner loses.
        r_ch = (prev - dist) / cfg.world_size
        r_rn = -r_ch

        done = False
        info: dict[str, Any] = {}
        if dist <= cfg.capture_radius:
            done = True
            r_ch += 1.0
            r_rn -= 1.0
            info["reason"] = "captured"
        elif self.steps >= cfg.timeout_steps:
            done = True
            info["reason"] = "timeout"

        self._prev_dist = dist
        info["distance"] = dist
        return self._get_state(), (r_ch, r_rn), done, info

    # ------------------------------------------------------------------ #
    # Geometry helpers
    # ------------------------------------------------------------------ #
    def _wrap(self, pos: np.ndarray) -> np.ndarray:
        return np.mod(pos, self.cfg.world_size)

    def _torus_vector(self, from_pos: np.ndarray, to_pos: np.ndarray) -> np.ndarray:
        """Shortest displacement from ``from_pos`` to ``to_pos`` across wrap edges."""
        size = self.cfg.world_size
        d = to_pos - from_pos
        d = (d + size / 2.0) % size - size / 2.0
        return d

    def _clip_speed(self, vel: np.ndarray) -> np.ndarray:
        speed = float(np.linalg.norm(vel))
        if speed > self.cfg.max_speed:
            return vel * (self.cfg.max_speed / speed)
        return vel

    def _get_state(self) -> np.ndarray:
        return np.concatenate([self.ch_pos, self.ch_vel, self.rn_pos, self.rn_vel]).astype(
            np.float32
        )

    @property
    def distance(self) -> float:
        return float(np.linalg.norm(self._torus_vector(self.ch_pos, self.rn_pos)))
