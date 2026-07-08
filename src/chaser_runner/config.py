"""Typed configuration objects, loadable from YAML.

The whole project is driven by a single :class:`Config` tree so that training,
evaluation and the interactive showcase all share the same source of truth.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
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
    friction: float = 0.1
    capture_radius: float = 20.0
    timeout_steps: int = 1000

    @property
    def dt(self) -> float:
        return 1.0 / self.tickrate


@dataclass
class AgentConfig:
    """Q-network learner shared by both the chaser and the runner."""

    hidden: int = 256
    lr: float = 1e-4
    gamma: float = 0.99
    sample_actions: int = 128
    batch_size: int = 128
    buffer_capacity: int = 200_000
    target_update_freq: int = 400
    target_samples: int = 32
    grad_clip: float = 10.0
    min_buffer: int = 1000


@dataclass
class TrainConfig:
    """Outer training loop and exploration schedule."""

    total_steps: int = 200_000
    train_iters_per_step: int = 2
    eps_start: float = 0.3
    eps_final: float = 0.02
    eps_decay_steps: int = 20_000
    log_interval: int = 100
    save_interval: int = 5_000
    seed: int = 0


@dataclass
class Config:
    env: EnvConfig = field(default_factory=EnvConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> Config:
        with open(path, encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Config:
        sections = {"env": EnvConfig, "agent": AgentConfig, "train": TrainConfig}
        kwargs = {name: cls_(**(raw.get(name) or {})) for name, cls_ in sections.items()}
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
