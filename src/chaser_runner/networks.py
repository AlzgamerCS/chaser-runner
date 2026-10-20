"""Neural network models."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from chaser_runner.env import ACTION_DIM, STATE_DIM


class QNetwork(nn.Module):
    """Action-value network ``Q(s, a)`` for continuous actions.

    Instead of the usual discrete DQN head, this takes the state *and* a
    candidate action and returns a single scalar value, which lets us handle a
    continuous action space by scoring sampled actions (see :class:`DQNAgent`).
    """

    def __init__(self, state_dim: int = STATE_DIM, action_dim: int = ACTION_DIM, hidden: int = 256):
        super().__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, 1)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, action], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class Actor(nn.Module):
    """Deterministic continuous policy with bounded accelerations."""

    def __init__(self, state_dim: int = STATE_DIM, action_dim: int = ACTION_DIM, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim),
            nn.Tanh(),
        )
        nn.init.uniform_(self.net[-2].weight, -0.003, 0.003)
        nn.init.zeros_(self.net[-2].bias)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)
