"""Atomic artifacts, versioned checkpoints, and isolated evaluation randomness."""

from __future__ import annotations

import json
import os
import random
import sys
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4

import numpy as np
import torch

SCHEMA_VERSION = 1


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def rng_state() -> dict:
    state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": [state[0], torch.tensor(state[1].astype(np.int64)), state[2], state[3], state[4]],
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng(state: dict) -> None:
    random.setstate(state["python"])
    n = state["numpy"]
    np.random.set_state((n[0], n[1].cpu().numpy().astype(np.uint32), n[2], n[3], n[4]))
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])


@contextmanager
def isolated_rng(seed: int):
    previous = rng_state()
    try:
        seed_all(seed)
        yield
    finally:
        restore_rng(previous)


def atomic_save(data: dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temp.open("wb") as fh:
            torch.save(data, fh)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)
    return path


def atomic_json(data: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temp.open("w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, allow_nan=False)
            fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def checkpoint_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_dir():
        pointer = json.loads((path / "latest.json").read_text(encoding="utf-8"))
        path = path / pointer["checkpoint"]
    return path


def read_checkpoint(path: str | Path) -> dict:
    data = torch.load(checkpoint_path(path), map_location="cpu", weights_only=True)
    if data.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported checkpoint schema version")
    if data.get("kind") not in {"training", "policy"}:
        raise ValueError("Expected a training checkpoint or a policy bundle")
    return data


@contextmanager
def run_lock(directory: Path):
    """OS lock released on process exit, including crashes (no stale PID lock)."""
    with (directory / ".training.lock").open("a+b") as fh:
        if fh.seek(0, os.SEEK_END) == 0:
            fh.write(b"0")
            fh.flush()
        fh.seek(0)
        try:
            if sys.platform == "win32":
                import msvcrt

                msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise ValueError(f"Run is already being trained: {directory}") from error
        try:
            yield
        finally:
            fh.seek(0)
            if sys.platform == "win32":
                msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(fh, fcntl.LOCK_UN)
