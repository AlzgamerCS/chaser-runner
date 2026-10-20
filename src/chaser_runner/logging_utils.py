"""Pluggable metric logging.

A tiny abstraction so the training loop does not depend on any one backend.
TensorBoard is the default; a null logger is used when logging is disabled, and
the interface leaves room to add an MLflow backend later without touching the
trainer.
"""

from __future__ import annotations

import json
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


class RunLogger:
    """Keep portable metric history, even when TensorBoard is disabled.

    Each invocation appends a session marker. If a crash rolls back to an older
    checkpoint, readers can distinguish discarded steps from resumed progress.
    """

    def __init__(self, path: Path, backend: MetricLogger, step: int):
        self.backend = backend
        self.fh = path.open("a", encoding="utf-8", buffering=1)
        self._write({"event": "session", "start_step": step})

    def _write(self, record: dict) -> None:
        self.fh.write(json.dumps(record, allow_nan=False) + "\n")

    def log_scalars(self, tag: str, values: dict[str, float], step: int) -> None:
        self._write({"event": "metrics", "tag": tag, "step": step, "values": values})
        self.backend.log_scalars(tag, values, step)

    def log_hparams(self, params: dict[str, Any]) -> None:
        self._write({"event": "parameters", "values": params})
        self.backend.log_hparams(params)

    def close(self) -> None:
        try:
            self.backend.close()
        finally:
            self.fh.close()


class TensorBoardLogger:
    """Writes scalars and hyperparameters to a TensorBoard run directory."""

    def __init__(self, logdir: str | Path, purge_step: int | None = None):
        from torch.utils.tensorboard import SummaryWriter

        self.logdir = Path(logdir)
        self.writer = SummaryWriter(str(self.logdir), purge_step=purge_step)

    def log_scalars(self, tag: str, values: dict[str, float], step: int) -> None:
        for key, value in values.items():
            self.writer.add_scalar(f"{tag}/{key}", value, step)

    def log_hparams(self, params: dict[str, Any]) -> None:
        text = "\n".join(f"- **{k}**: {v}" for k, v in params.items())
        self.writer.add_text("hparams", text, 0)

    def close(self) -> None:
        self.writer.flush()
        self.writer.close()
