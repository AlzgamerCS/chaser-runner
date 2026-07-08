"""DQN-style agent for continuous actions via action sampling."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim

from chaser_runner.config import AgentConfig
from chaser_runner.env import ACTION_DIM, STATE_DIM
from chaser_runner.networks import QNetwork


class DQNAgent:
    """A continuous-action Q-learner.

    Action selection samples ``sample_actions`` random actions and keeps the one
    with the highest predicted Q-value. The Bellman target's ``max`` over next
    actions is estimated the same way, by sampling ``target_samples`` actions.
    """

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
            q_next = self.q_target(s2_rep, cand).view(cfg.batch_size, k)
            q_next_max = q_next.max(dim=1, keepdim=True)[0]
            target = r + cfg.gamma * (1.0 - done) * q_next_max

        loss = F.mse_loss(q_pred, target)
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
        torch.save(self.q.state_dict(), path)

    def load(self, path: str | Path) -> None:
        state = torch.load(path, map_location=self.device, weights_only=True)
        self.q.load_state_dict(state)
        self.q_target.load_state_dict(self.q.state_dict())
