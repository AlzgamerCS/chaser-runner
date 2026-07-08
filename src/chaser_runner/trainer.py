"""Self-play training loop, evaluation, and rollout helpers."""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from chaser_runner.agent import DQNAgent
from chaser_runner.config import Config
from chaser_runner.env import ChaseEnv
from chaser_runner.logging_utils import MetricLogger, NullLogger


def resolve_device(device: str = "auto") -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_agents(config: Config, device: str) -> tuple[DQNAgent, DQNAgent]:
    chaser = DQNAgent(config.agent, device=device, name="chaser")
    runner = DQNAgent(config.agent, device=device, name="runner")
    return chaser, runner


class Trainer:
    """Trains the chaser and runner simultaneously against each other."""

    def __init__(
        self,
        config: Config,
        *,
        device: str = "auto",
        logger: MetricLogger | None = None,
        model_dir: str | Path = "models",
    ):
        self.config = config
        self.device = resolve_device(device)
        self.logger = logger or NullLogger()
        self.model_dir = Path(model_dir)

        set_seed(config.train.seed)
        self.env = ChaseEnv(config.env)
        self.chaser, self.runner = make_agents(config, self.device)

    def load(self, chaser_path: str | Path | None, runner_path: str | Path | None) -> None:
        if chaser_path:
            self.chaser.load(chaser_path)
        if runner_path:
            self.runner.load(runner_path)

    def _epsilon(self, step: int) -> float:
        tc = self.config.train
        frac = step / max(1, tc.eps_decay_steps)
        return max(tc.eps_final, tc.eps_start - (tc.eps_start - tc.eps_final) * frac)

    def save(self) -> tuple[Path, Path]:
        ch_path = self.model_dir / "chaser_q.pth"
        rn_path = self.model_dir / "runner_q.pth"
        self.chaser.save(ch_path)
        self.runner.save(rn_path)
        return ch_path, rn_path

    def train(self, on_step=None) -> None:
        """Run the training loop for ``config.train.total_steps`` steps.

        ``on_step(env, stats)`` is called once per step if provided, which the
        CLI uses to render a live window during training.
        """
        cfg = self.config
        self.logger.log_hparams(cfg.flat())

        state = self.env.reset(seed=cfg.train.seed)
        losses = {"chaser": deque(maxlen=200), "runner": deque(maxlen=200)}
        ep_reward = {"chaser": 0.0, "runner": 0.0}
        episodes = 0

        try:
            for step in range(1, cfg.train.total_steps + 1):
                eps = self._epsilon(step)
                act_ch = self.chaser.select_action(state, explore_eps=eps)
                act_rn = self.runner.select_action(state, explore_eps=eps)

                next_state, (r_ch, r_rn), done, info = self.env.step(act_ch, act_rn)
                self.chaser.store(state, act_ch, r_ch, next_state, done)
                self.runner.store(state, act_rn, r_rn, next_state, done)

                for _ in range(cfg.train.train_iters_per_step):
                    lch = self.chaser.train_once()
                    lrn = self.runner.train_once()
                    if lch is not None:
                        losses["chaser"].append(lch)
                    if lrn is not None:
                        losses["runner"].append(lrn)

                ep_reward["chaser"] += r_ch
                ep_reward["runner"] += r_rn
                state = next_state

                if on_step is not None:
                    on_step(
                        self.env,
                        {
                            "eps": f"{eps:.3f}",
                            "loss_c": f"{_mean(losses['chaser']):.4f}",
                            "loss_r": f"{_mean(losses['runner']):.4f}",
                        },
                    )

                if step % cfg.train.log_interval == 0:
                    self.logger.log_scalars(
                        "train",
                        {
                            "epsilon": eps,
                            "loss_chaser": _mean(losses["chaser"]),
                            "loss_runner": _mean(losses["runner"]),
                            "distance": info["distance"],
                        },
                        step,
                    )

                if step % cfg.train.save_interval == 0:
                    self.save()

                if done:
                    episodes += 1
                    self.logger.log_scalars(
                        "episode",
                        {
                            "reward_chaser": ep_reward["chaser"],
                            "reward_runner": ep_reward["runner"],
                            "length": self.env.steps,
                            "captured": 1.0 if info.get("reason") == "captured" else 0.0,
                        },
                        step,
                    )
                    print(
                        f"[step {step}] episode {episodes} {info.get('reason')} "
                        f"len={self.env.steps} "
                        f"R_ch={ep_reward['chaser']:.3f} R_rn={ep_reward['runner']:.3f}"
                    )
                    state = self.env.reset()
                    ep_reward = {"chaser": 0.0, "runner": 0.0}
        finally:
            self.save()
            self.logger.close()


@dataclass
class EvalResult:
    episodes: int
    capture_rate: float
    avg_length: float
    avg_reward_chaser: float
    avg_reward_runner: float


def evaluate(
    config: Config,
    chaser: DQNAgent,
    runner: DQNAgent,
    *,
    episodes: int = 50,
    seed: int = 123,
) -> EvalResult:
    """Roll out greedy (no-exploration) policies and summarise the outcome."""
    env = ChaseEnv(config.env)
    captures = 0
    lengths: list[int] = []
    rewards_ch: list[float] = []
    rewards_rn: list[float] = []

    for ep in range(episodes):
        state = env.reset(seed=seed + ep)
        r_ch_total = r_rn_total = 0.0
        done = False
        info: dict = {}
        while not done:
            act_ch = chaser.select_action(state, explore_eps=0.0)
            act_rn = runner.select_action(state, explore_eps=0.0)
            state, (r_ch, r_rn), done, info = env.step(act_ch, act_rn)
            r_ch_total += r_ch
            r_rn_total += r_rn
        captures += int(info.get("reason") == "captured")
        lengths.append(env.steps)
        rewards_ch.append(r_ch_total)
        rewards_rn.append(r_rn_total)

    return EvalResult(
        episodes=episodes,
        capture_rate=captures / episodes,
        avg_length=float(np.mean(lengths)),
        avg_reward_chaser=float(np.mean(rewards_ch)),
        avg_reward_runner=float(np.mean(rewards_rn)),
    )


def _mean(values) -> float:
    return float(np.mean(values)) if len(values) else 0.0
