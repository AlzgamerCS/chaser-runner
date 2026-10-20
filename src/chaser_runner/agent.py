"""DQN-style agent for continuous actions via action sampling."""

from __future__ import annotations

import random
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim

from chaser_runner.config import AgentConfig
from chaser_runner.env import ACTION_DIM, STATE_DIM
from chaser_runner.networks import Actor, QNetwork
from chaser_runner.persistence import atomic_save


class AgentState:
    """Shared complete training-state serialization for both learners."""

    module_names: tuple[str, ...]
    optimizer_names: tuple[str, ...]

    def training_state(self) -> dict:
        return {
            "modules": {k: getattr(self, k).state_dict() for k in self.module_names},
            "optimizers": {k: getattr(self, k).state_dict() for k in self.optimizer_names},
            "replay": self.replay.state_dict(),
            "train_steps": self.train_steps,
            "env_steps": self.env_steps,
        }

    def restore_training_state(self, state: dict) -> None:
        for k, v in state["modules"].items():
            getattr(self, k).load_state_dict(v)
        for k, v in state["optimizers"].items():
            getattr(self, k).load_state_dict(v)
        self.replay.load_state_dict(state["replay"])
        self.train_steps, self.env_steps = state["train_steps"], state["env_steps"]

    def policy_state(self) -> dict:
        net = self.actor if isinstance(self, TD3Agent) else self.q
        return {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}


class DQNAgent(AgentState):
    """A continuous-action Q-learner.

    Action selection samples ``sample_actions`` random actions and keeps the one
    with the highest predicted Q-value. The Bellman target's ``max`` over next
    actions is estimated the same way, by sampling ``target_samples`` actions.
    """

    module_names = ("q", "q_target")
    optimizer_names = ("opt",)

    def __init__(
        self,
        config: AgentConfig | None = None,
        *,
        state_dim: int = STATE_DIM,
        action_dim: int = ACTION_DIM,
        device: str | torch.device = "cpu",
        name: str = "agent",
    ):
        self.cfg = config or AgentConfig()
        self.device = torch.device(device)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.name = name

        self.q = QNetwork(state_dim, action_dim, self.cfg.hidden).to(self.device)
        self.q_target = QNetwork(state_dim, action_dim, self.cfg.hidden).to(self.device)
        self.q_target.load_state_dict(self.q.state_dict())
        self.opt = optim.Adam(self.q.parameters(), lr=self.cfg.lr)

        from chaser_runner.replay import ReplayBuffer

        self.replay = ReplayBuffer(self.cfg.buffer_capacity)
        self.train_steps = 0
        self.env_steps = 0

    # ------------------------------------------------------------------ #
    # Acting
    # ------------------------------------------------------------------ #
    def select_action(self, state: np.ndarray, explore_eps: float = 0.0) -> np.ndarray:
        if explore_eps > 0 and random.random() < explore_eps:
            return np.random.uniform(-1, 1, size=(self.action_dim,)).astype(np.float32)

        n = self.cfg.sample_actions
        states = np.repeat(state.reshape(1, -1), n, axis=0)
        actions = np.random.uniform(-1, 1, size=(n, self.action_dim)).astype(np.float32)

        with torch.no_grad():
            s_t = torch.as_tensor(states, dtype=torch.float32, device=self.device)
            a_t = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
            qvals = self.q(s_t, a_t).cpu().numpy().reshape(-1)
        return actions[int(np.argmax(qvals))]

    def store(self, s: np.ndarray, a: np.ndarray, r: float, s2: np.ndarray, done: float) -> None:
        self.replay.add(s, a, r, s2, done)
        self.env_steps += 1

    # ------------------------------------------------------------------ #
    # Learning
    # ------------------------------------------------------------------ #
    def train_once(self) -> float | None:
        cfg = self.cfg
        if len(self.replay) < max(cfg.min_buffer, cfg.batch_size):
            return None

        s, a, r, s2, done = self.replay.sample(cfg.batch_size)
        s = torch.as_tensor(s, device=self.device)
        a = torch.as_tensor(a, device=self.device)
        r = torch.as_tensor(r, device=self.device)
        s2 = torch.as_tensor(s2, device=self.device)
        done = torch.as_tensor(done, device=self.device)

        q_pred = self.q(s, a)

        # Estimate max_a' Q_target(s2, a') by sampling K candidate actions.
        k = cfg.target_samples
        batch_k = cfg.batch_size * k
        cand = torch.rand(batch_k, self.action_dim, device=self.device) * 2 - 1
        s2_rep = s2.unsqueeze(1).repeat(1, k, 1).view(batch_k, -1)
        with torch.no_grad():
            # Double DQN: select with the online net, evaluate with the target.
            indices = self.q(s2_rep, cand).view(cfg.batch_size, k).argmax(dim=1, keepdim=True)
            q_next = self.q_target(s2_rep, cand).view(cfg.batch_size, k)
            q_next_max = q_next.gather(1, indices)
            target = r + cfg.gamma * (1.0 - done) * q_next_max

        loss = F.smooth_l1_loss(q_pred, target)
        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q.parameters(), cfg.grad_clip)
        self.opt.step()

        self.train_steps += 1
        if self.train_steps % cfg.target_update_freq == 0:
            self.q_target.load_state_dict(self.q.state_dict())
        return float(loss.item())

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_save(self.q.state_dict(), path)

    def load(self, path: str | Path) -> None:
        state = torch.load(path, map_location=self.device, weights_only=True)
        self.q.load_state_dict(state)
        self.q_target.load_state_dict(self.q.state_dict())


class TD3Agent(AgentState):
    """Twin critics, delayed deterministic actor and smoothed target actions."""

    module_names = ("actor", "actor_target", "q", "q2", "q_target", "q2_target")
    optimizer_names = ("opt", "actor_opt")

    def __init__(self, config: AgentConfig, *, device="cpu", name="agent"):
        self.cfg, self.device, self.name = config, torch.device(device), name
        self.actor = Actor(hidden=config.hidden).to(self.device)
        self.q = QNetwork(hidden=config.hidden).to(self.device)
        self.q2 = QNetwork(hidden=config.hidden).to(self.device)
        self.actor_target = deepcopy(self.actor).requires_grad_(False)
        self.q_target = deepcopy(self.q).requires_grad_(False)
        self.q2_target = deepcopy(self.q2).requires_grad_(False)
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=config.lr)
        self.opt = optim.Adam(list(self.q.parameters()) + list(self.q2.parameters()), lr=config.lr)
        from chaser_runner.replay import ReplayBuffer

        self.replay = ReplayBuffer(config.buffer_capacity)
        self.train_steps = self.env_steps = 0

    def select_action(self, state: np.ndarray, explore_eps: float = 0.0) -> np.ndarray:
        with torch.no_grad():
            action = self.actor(torch.as_tensor(state, device=self.device)).cpu().numpy()
        if explore_eps:
            action = action + np.random.normal(0, explore_eps, size=ACTION_DIM)
        return np.clip(action, -1, 1).astype(np.float32)

    def store(self, s, a, r, s2, done) -> None:
        self.replay.add(s, a, r, s2, done)
        self.env_steps += 1

    def train_once(self) -> float | None:
        cfg = self.cfg
        if len(self.replay) < max(cfg.min_buffer, cfg.batch_size):
            return None
        s, a, r, s2, terminal = [
            torch.as_tensor(v, device=self.device) for v in self.replay.sample(cfg.batch_size)
        ]
        with torch.no_grad():
            noise = (torch.randn_like(a) * cfg.policy_noise).clamp(-cfg.noise_clip, cfg.noise_clip)
            next_action = (self.actor_target(s2) + noise).clamp(-1, 1)
            next_q = torch.minimum(self.q_target(s2, next_action), self.q2_target(s2, next_action))
            target = r + cfg.gamma * (1 - terminal) * next_q
        loss = F.mse_loss(self.q(s, a), target) + F.mse_loss(self.q2(s, a), target)
        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.q.parameters()) + list(self.q2.parameters()), cfg.grad_clip
        )
        self.opt.step()
        self.train_steps += 1
        if self.train_steps % cfg.policy_delay == 0:
            self.q.requires_grad_(False)
            actor_loss = -self.q(s, self.actor(s)).mean()
            self.actor_opt.zero_grad()
            actor_loss.backward()
            self.actor_opt.step()
            self.q.requires_grad_(True)
            with torch.no_grad():
                for source, target_net in (
                    (self.actor, self.actor_target),
                    (self.q, self.q_target),
                    (self.q2, self.q2_target),
                ):
                    for p, tp in zip(source.parameters(), target_net.parameters(), strict=True):
                        tp.lerp_(p, cfg.tau)
        return float(loss.item())

    def save(self, path) -> None:
        atomic_save(self.policy_state(), path)

    def load(self, path) -> None:
        self.actor.load_state_dict(torch.load(path, map_location=self.device, weights_only=True))
        self.actor_target.load_state_dict(self.actor.state_dict())


Agent = DQNAgent | TD3Agent


def make_agent(config: AgentConfig, *, device="cpu", name="agent") -> Agent:
    cls = TD3Agent if config.algorithm == "td3" else DQNAgent
    return cls(config, device=device, name=name)


class FrozenPolicy:
    """Only the inference network, for a historical opponent or saved policy."""

    def __init__(self, config: AgentConfig, state: dict, device="cpu"):
        self.config, self.device = config, torch.device(device)
        self.net = (
            Actor(hidden=config.hidden)
            if config.algorithm == "td3"
            else QNetwork(hidden=config.hidden)
        ).to(self.device)
        self.net.load_state_dict(state)
        self.net.eval().requires_grad_(False)

    def select_action(self, state, explore_eps=0.0):
        with torch.no_grad():
            if self.config.algorithm == "td3":
                return self.net(torch.as_tensor(state, device=self.device)).cpu().numpy()
            actions = np.random.uniform(-1, 1, (self.config.sample_actions, ACTION_DIM)).astype(
                np.float32
            )
            states = np.repeat(state[None], len(actions), axis=0)
            values = self.net(
                torch.as_tensor(states, device=self.device),
                torch.as_tensor(actions, device=self.device),
            ).flatten()
            return actions[int(values.argmax())]
