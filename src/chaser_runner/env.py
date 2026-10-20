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

# State: [rel_disp(2), chaser_vel(2), runner_vel(2)], all normalized. See
# ChaseEnv._get_state for the rationale behind the relative encoding.
STATE_DIM = 6
ACTION_DIM = 2


class ChaseEnv:
    """Toroidal pursuit-evasion environment with continuous acceleration control."""

    def __init__(self, config: EnvConfig | None = None):
        self.cfg = config or EnvConfig()
        self.steps = 0
        self._done = False
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
        self._done = False
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
        """Hold actions for several ticks; time limits truncate, captures terminate."""
        if self._done:
            raise RuntimeError("Reset the environment after an episode ends")
        for action in (action_chaser, action_runner):
            if np.shape(action) != (ACTION_DIM,) or not np.isfinite(action).all():
                raise ValueError("Actions must be finite vectors of shape (2,)")
        total_ch = total_rn = 0.0
        for _ in range(self.cfg.action_repeat):
            state, (rch, rrn), done, info = self._physics_step(action_chaser, action_runner)
            total_ch += rch
            total_rn += rrn
            if done:
                break
        self._done = done
        return state, (total_ch, total_rn), done, info

    def _physics_step(
        self, action_chaser: np.ndarray, action_runner: np.ndarray
    ) -> tuple[np.ndarray, tuple[float, float], bool, dict[str, Any]]:
        self.steps += 1
        cfg = self.cfg

        adv = cfg.chaser_advantage
        acc_ch = np.clip(action_chaser, -1.0, 1.0).astype(float) * cfg.max_acc * adv
        acc_rn = np.clip(action_runner, -1.0, 1.0).astype(float) * cfg.max_acc

        self.ch_vel = self.ch_vel + acc_ch * cfg.dt
        self.rn_vel = self.rn_vel + acc_rn * cfg.dt

        self.ch_vel *= 1.0 - cfg.friction * cfg.dt
        self.rn_vel *= 1.0 - cfg.friction * cfg.dt

        self.ch_vel = self._clip_speed(self.ch_vel, cfg.max_speed * adv)
        self.rn_vel = self._clip_speed(self.rn_vel, cfg.max_speed)

        self.ch_pos = self._wrap(self.ch_pos + self.ch_vel * cfg.dt)
        self.rn_pos = self._wrap(self.rn_pos + self.rn_vel * cfg.dt)

        disp = self._torus_vector(self.ch_pos, self.rn_pos)
        dist = float(np.linalg.norm(disp))
        prev = self._prev_dist if self._prev_dist is not None else dist

        # Shaped, zero-sum reward: chaser gains what the runner loses.
        r_ch = (prev - dist) / cfg.world_size * cfg.distance_reward_scale
        r_rn = -r_ch

        done = False
        info: dict[str, Any] = {}
        if dist <= cfg.capture_radius:
            done = True
            r_ch += cfg.capture_reward
            r_rn -= cfg.capture_reward
            info["reason"] = "captured"
        elif self.steps >= cfg.timeout_steps:
            done = True
            info["reason"] = "timeout"

        self._prev_dist = dist
        info["distance"] = dist
        info["terminated"] = info.get("reason") == "captured"
        info["truncated"] = info.get("reason") == "timeout"
        return self._get_state(), (r_ch, r_rn), done, info

    def state_dict(self) -> dict[str, Any]:
        return {
            "steps": self.steps,
            "done": self._done,
            "prev_dist": self._prev_dist,
            "rng": self.rng.bit_generator.state,
            **{
                key: getattr(self, key).tolist() for key in ("ch_pos", "ch_vel", "rn_pos", "rn_vel")
            },
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.steps = state["steps"]
        self._done = state.get("done", False)
        self._prev_dist = state["prev_dist"]
        self.rng.bit_generator.state = state["rng"]
        for key in ("ch_pos", "ch_vel", "rn_pos", "rn_vel"):
            setattr(self, key, np.asarray(state[key], dtype=float))

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

    def _clip_speed(self, vel: np.ndarray, max_speed: float) -> np.ndarray:
        speed = float(np.linalg.norm(vel))
        if speed > max_speed:
            return vel * (max_speed / speed)
        return vel

    def _get_state(self) -> np.ndarray:
        """Relative, normalized state (all components roughly in ``[-1, 1]``).

        On a homogeneous wrap-around world only *relative* geometry matters, so
        we feed the shortest chaser->runner displacement rather than absolute
        positions. Everything is normalized so the network sees well-scaled
        inputs, which is what makes the problem learnable.
        """
        half = self.cfg.world_size / 2.0
        ms = self.cfg.max_speed
        disp = self._torus_vector(self.ch_pos, self.rn_pos)
        return np.concatenate(
            [
                disp / half,  # where the runner is, relative to the chaser
                self.ch_vel / ms,  # chaser's own velocity
                self.rn_vel / ms,  # runner's own velocity
            ]
        ).astype(np.float32)

    @property
    def distance(self) -> float:
        return float(np.linalg.norm(self._torus_vector(self.ch_pos, self.rn_pos)))
