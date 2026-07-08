"""Pluggable metric logging.

A tiny abstraction so the training loop does not depend on any one backend.
TensorBoard is the default; a null logger is used when logging is disabled, and
the interface leaves room to add an MLflow backend later without touching the
trainer.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol


class MetricLogger(Protocol):
    def log_scalars(self, tag: str, values: dict[str, float], step: int) -> None: ...
    def log_hparams(self, params: dict[str, Any]) -> None: ...
    def close(self) -> None: ...


class NullLogger:
    """No-op logger."""

    def log_scalars(self, tag: str, values: dict[str, float], step: int) -> None:
        pass

    def log_hparams(self, params: dict[str, Any]) -> None:
        pass

    def close(self) -> None:
        pass


class TensorBoardLogger:
    """Writes scalars and hyperparameters to a TensorBoard run directory."""

    def __init__(self, logdir: str | Path):
        from torch.utils.tensorboard import SummaryWriter

        self.logdir = Path(logdir)
        self.writer = SummaryWriter(str(self.logdir))

    def log_scalars(self, tag: str, values: dict[str, float], step: int) -> None:
        for key, value in values.items():
            self.writer.add_scalar(f"{tag}/{key}", value, step)

    def log_hparams(self, params: dict[str, Any]) -> None:
        text = "\n".join(f"- **{k}**: {v}" for k, v in params.items())
        self.writer.add_text("hparams", text, 0)

    def close(self) -> None:
        self.writer.flush()
        self.writer.close()
