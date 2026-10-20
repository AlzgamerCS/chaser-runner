"""Deterministic benchmark opponents and reproducible, isolated rollouts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Protocol

import numpy as np

from chaser_runner.config import Config, EnvConfig
from chaser_runner.env import ChaseEnv
from chaser_runner.persistence import isolated_rng, seed_all


class Policy(Protocol):
    def select_action(self, state: np.ndarray, explore_eps: float = 0.0) -> np.ndarray: ...


class BaselinePolicy:
    def __init__(self, role: str, env: EnvConfig, kind: str = "scripted"):
        self.role, self.env, self.kind = role, env, kind

    def select_action(self, state, explore_eps=0.0):
        if self.kind == "random":
            return np.random.uniform(-1, 1, 2).astype(np.float32)
        if self.kind == "stationary":
            return np.zeros(2, dtype=np.float32)
        displacement = state[:2].astype(float)
        direction = displacement / max(float(np.linalg.norm(displacement)), 1e-8)
        is_chaser = self.role == "chaser"
        advantage = self.env.chaser_advantage if is_chaser else 1.0
        # State displacement points from chaser to runner. Both pursuit and
        # evasion accelerate along it; they differ in whose velocity is steered.
        desired = direction * self.env.max_speed * advantage
        velocity = state[2:4] if is_chaser else state[4:6]
        # A velocity controller gives a stable, reproducible pursue/evade anchor.
        return np.clip(
            (desired - velocity * self.env.max_speed) / (self.env.max_acc * advantage * 0.25), -1, 1
        ).astype(np.float32)


@dataclass
class EvalResult:
    episodes: int
    capture_rate: float
    avg_length: float
    avg_reward_chaser: float
    avg_reward_runner: float
    avg_distance: float
    capture_ci_low: float
    capture_ci_high: float

    @property
    def survival_rate(self) -> float:
        return 1 - self.capture_rate

    def to_dict(self) -> dict:
        return {**asdict(self), "survival_rate": self.survival_rate}


def evaluate(
    config: Config, chaser: Policy, runner: Policy, *, episodes: int = 50, seed: int = 123
) -> EvalResult:
    """No exploration; seed both resets and action sampling; preserve caller RNGs."""
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    captures = 0
    lengths, rewards_ch, rewards_rn, distances = [], [], [], []
    with isolated_rng(seed):
        env = ChaseEnv(config.env)
        for ep in range(episodes):
            seed_all(seed + ep)
            state = env.reset(seed=seed + ep)
            total_ch = total_rn = distance = 0.0
            decisions = 0
            done = False
            while not done:
                ach = chaser.select_action(state, explore_eps=0.0)
                arn = runner.select_action(state, explore_eps=0.0)
                state, (rch, rrn), done, info = env.step(ach, arn)
                total_ch += rch
                total_rn += rrn
                distance += info["distance"]
                decisions += 1
            captures += int(info["terminated"])
            lengths.append(env.steps)
            rewards_ch.append(total_ch)
            rewards_rn.append(total_rn)
            distances.append(distance / decisions)
    rate = captures / episodes
    z = 1.96
    denominator = 1 + z * z / episodes
    center = (rate + z * z / (2 * episodes)) / denominator
    margin = (
        z * np.sqrt(rate * (1 - rate) / episodes + z * z / (4 * episodes * episodes)) / denominator
    )
    return EvalResult(
        episodes,
        rate,
        float(np.mean(lengths)),
        float(np.mean(rewards_ch)),
        float(np.mean(rewards_rn)),
        float(np.mean(distances)),
        float(0 if captures == 0 else max(0, center - margin)),
        float(1 if captures == episodes else min(1, center + margin)),
    )


def benchmark(
    config: Config, chaser: Policy, runner: Policy, *, episodes: int = 20, seed: int = 10_000
) -> dict[str, dict]:
    matchups = {
        "paired": (chaser, runner),
        "chaser_vs_scripted": (chaser, BaselinePolicy("runner", config.env)),
        "scripted_vs_runner": (BaselinePolicy("chaser", config.env), runner),
        "chaser_vs_random": (chaser, BaselinePolicy("runner", config.env, "random")),
        "random_vs_runner": (BaselinePolicy("chaser", config.env, "random"), runner),
    }
    return {
        name: evaluate(config, ch, rn, episodes=episodes, seed=seed).to_dict()
        for name, (ch, rn) in matchups.items()
    }
