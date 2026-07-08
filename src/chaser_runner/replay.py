"""Experience replay buffer."""

from __future__ import annotations

import random
from collections import deque

import numpy as np

Batch = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]


class ReplayBuffer:
    """Fixed-capacity ring buffer of ``(s, a, r, s2, done)`` transitions."""

    def __init__(self, capacity: int = 200_000):
        self.capacity = capacity
        self.buffer: deque = deque(maxlen=capacity)

    def add(self, s: np.ndarray, a: np.ndarray, r: float, s2: np.ndarray, done: float) -> None:
        self.buffer.append(
            (
                s.astype(np.float32),
                a.astype(np.float32),
                float(r),
                s2.astype(np.float32),
                float(done),
            )
        )

    def sample(self, batch_size: int) -> Batch:
        batch = random.sample(self.buffer, batch_size)
        s, a, r, s2, d = zip(*batch, strict=True)
        return (
            np.stack(s),
            np.stack(a),
            np.array(r, dtype=np.float32).reshape(-1, 1),
            np.stack(s2),
            np.array(d, dtype=np.float32).reshape(-1, 1),
        )

    def __len__(self) -> int:
        return len(self.buffer)
