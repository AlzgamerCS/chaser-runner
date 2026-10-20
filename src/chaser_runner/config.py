"""Typed configuration objects, loadable from YAML.

The whole project is driven by a single :class:`Config` tree so that training,
evaluation and the interactive showcase all share the same source of truth.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass
class EnvConfig:
    """Physics of the toroidal chase world."""

    world_size: float = 800.0
    tickrate: int = 60
    max_acc: float = 300.0
    max_speed: float = 250.0
    # Small pursuit advantage; relative skill still needs fixed-opponent evaluation.
    chaser_advantage: float = 1.2
    friction: float = 0.1
    capture_radius: float = 25.0
    timeout_steps: int = 1000
    action_repeat: int = 4
    distance_reward_scale: float = 100.0
    capture_reward: float = 10.0

    @property
    def dt(self) -> float:
        return 1.0 / self.tickrate


@dataclass
class AgentConfig:
    """Learner settings shared by both roles; sampling fields apply only to DQN."""

    algorithm: str = "td3"
    hidden: int = 128
    lr: float = 3e-4
    gamma: float = 0.99
    sample_actions: int = 128
    batch_size: int = 128
    buffer_capacity: int = 50_000
    target_update_freq: int = 400
    target_samples: int = 32
    grad_clip: float = 10.0
    min_buffer: int = 1000
    start_steps: int = 1000
    tau: float = 0.005
    policy_delay: int = 2
    policy_noise: float = 0.2
    noise_clip: float = 0.5


@dataclass
class TrainConfig:
    """Outer training loop and exploration schedule."""

    total_steps: int = 200_000
    train_iters_per_step: int = 1
    eps_start: float = 0.3
    eps_final: float = 0.05
    eps_decay_steps: int = 100_000
    log_interval: int = 100
    save_interval: int = 5_000
    seed: int = 0
    eval_interval: int = 10_000
    eval_episodes: int = 20
    eval_seed: int = 10_000
    snapshot_interval: int = 2_000
    opponent_pool_size: int = 8
    random_opponent_prob: float = 0.2
    scripted_opponent_prob: float = 0.3
    keep_checkpoints: int = 3
    torch_threads: int = 1


@dataclass
class Config:
    env: EnvConfig = field(default_factory=EnvConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def __post_init__(self) -> None:
        for section in (self.env, self.agent, self.train):
            for item in fields(section):
                value, expected = getattr(section, item.name), type(item.default)
                valid = (
                    type(value) in {int, float} if expected is float else type(value) is expected
                )
                if not valid:
                    raise ValueError(f"{item.name} must be {expected.__name__}")
                if isinstance(value, (int, float)) and not math.isfinite(value):
                    raise ValueError(f"{item.name} must be finite")
        positive = {
            "world_size": self.env.world_size,
            "tickrate": self.env.tickrate,
            "max_speed": self.env.max_speed,
            "max_acc": self.env.max_acc,
            "chaser_advantage": self.env.chaser_advantage,
            "timeout_steps": self.env.timeout_steps,
            "action_repeat": self.env.action_repeat,
            "hidden": self.agent.hidden,
            "lr": self.agent.lr,
            "batch_size": self.agent.batch_size,
            "buffer_capacity": self.agent.buffer_capacity,
            "target_update_freq": self.agent.target_update_freq,
            "sample_actions": self.agent.sample_actions,
            "target_samples": self.agent.target_samples,
            "policy_delay": self.agent.policy_delay,
            "grad_clip": self.agent.grad_clip,
            "train_iters_per_step": self.train.train_iters_per_step,
            "log_interval": self.train.log_interval,
            "save_interval": self.train.save_interval,
            "eval_episodes": self.train.eval_episodes,
            "snapshot_interval": self.train.snapshot_interval,
            "opponent_pool_size": self.train.opponent_pool_size,
            "keep_checkpoints": self.train.keep_checkpoints,
            "torch_threads": self.train.torch_threads,
        }
        if any(value <= 0 for value in positive.values()):
            raise ValueError(f"Values must be positive: {positive}")
        if not 0 <= self.env.capture_radius < self.env.world_size * 0.2:
            raise ValueError("capture_radius must be in [0, world_size * 0.2)")
        if not 0 <= self.env.friction * self.env.dt < 1:
            raise ValueError("friction * dt must be in [0, 1)")
        if self.agent.algorithm not in {"dqn", "td3"}:
            raise ValueError("algorithm must be dqn or td3")
        if not 0 <= self.agent.gamma < 1 or not 0 < self.agent.tau <= 1:
            raise ValueError("gamma must be in [0, 1), tau in (0, 1]")
        if self.agent.buffer_capacity < max(self.agent.batch_size, self.agent.min_buffer):
            raise ValueError("buffer_capacity must cover batch_size and min_buffer")
        if not 0 <= self.train.eps_final <= self.train.eps_start <= 1:
            raise ValueError("exploration must satisfy 0 <= final <= start <= 1")
        if not (
            0 <= self.train.random_opponent_prob <= 1
            and 0 <= self.train.scripted_opponent_prob <= 1
            and self.train.random_opponent_prob + self.train.scripted_opponent_prob <= 1
        ):
            raise ValueError("opponent probabilities must be nonnegative and sum to at most 1")
        for value in (
            self.train.total_steps,
            self.train.eval_interval,
            self.train.eps_decay_steps,
            self.agent.min_buffer,
            self.agent.start_steps,
            self.agent.policy_noise,
            self.agent.noise_clip,
            self.env.distance_reward_scale,
            self.env.capture_reward,
            self.train.seed,
            self.train.eval_seed,
        ):
            if value < 0:
                raise ValueError("Step counts, seeds, noise and reward scales must be nonnegative")

    @classmethod
    def from_yaml(cls, path: str | Path) -> Config:
        with open(path, encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Config:
        sections = {"env": EnvConfig, "agent": AgentConfig, "train": TrainConfig}
        if not isinstance(raw, dict) or set(raw) - sections.keys():
            raise ValueError("Config must contain only env, agent and train sections")
        kwargs = {}
        for name, section_type in sections.items():
            values = raw.get(name, {})
            if not isinstance(values, dict):
                raise ValueError(f"{name} must be a mapping")
            unknown = set(values) - {item.name for item in fields(section_type)}
            if unknown:
                raise ValueError(f"Unknown {name} settings: {sorted(unknown)}")
            kwargs[name] = section_type(**values)
        return cls(**kwargs)

    def to_dict(self) -> dict[str, Any]:
        return {"env": asdict(self.env), "agent": asdict(self.agent), "train": asdict(self.train)}

    def flat(self) -> dict[str, Any]:
        """Flattened ``section.key`` view, handy for logging hyperparameters."""
        out: dict[str, Any] = {}
        for section, values in self.to_dict().items():
            for key, value in values.items():
                out[f"{section}.{key}"] = value
        return out


def load_config(path: str | Path | None) -> Config:
    """Load a config from ``path``, or return defaults when ``path`` is ``None``."""
    if path is None:
        return Config()
    return Config.from_yaml(path)
