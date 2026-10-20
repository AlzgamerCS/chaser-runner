"""Compact, serializable replay with an explicit absorbing-terminal mask."""

from __future__ import annotations

import numpy as np
import torch

Batch = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]


class ReplayBuffer:
    def __init__(self, capacity: int = 50_000):
        self.capacity = capacity
        self.size = 0
        self.position = 0
        self.arrays: list[np.ndarray] = []

    def add(self, s: np.ndarray, a: np.ndarray, r: float, s2: np.ndarray, done: float) -> None:
        # `done` means truly terminated (capture), never just time-limited.
        values = (s, a, np.array([r]), s2, np.array([done]))
        if not self.arrays:
            self.arrays = [np.zeros((self.capacity, *v.shape), dtype=np.float32) for v in values]
        for array, value in zip(self.arrays, values, strict=True):
            array[self.position] = value
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Batch:
        indices = np.random.choice(self.size, size=batch_size, replace=False)
        return tuple(array[indices] for array in self.arrays)

    def state_dict(self) -> dict:
        return {
            "capacity": self.capacity,
            "size": self.size,
            "position": self.position,
            "arrays": [torch.from_numpy(a[: self.size].copy()) for a in self.arrays],
        }

    def load_state_dict(self, state: dict) -> None:
        if state["capacity"] != self.capacity:
            raise ValueError("Replay capacity differs from checkpoint")
        self.size, self.position = state["size"], state["position"]
        self.arrays = []
        for tensor in state["arrays"]:
            array = np.zeros((self.capacity, *tensor.shape[1:]), dtype=np.float32)
            array[: self.size] = tensor.cpu().numpy()
            self.arrays.append(array)

    def __len__(self) -> int:
        return self.size
