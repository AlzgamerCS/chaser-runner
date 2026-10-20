"""Alternating self-play, fixed benchmarks, and complete training continuation."""

from __future__ import annotations

import platform
import signal
import subprocess
import threading
import zipfile
from collections import deque
from importlib.metadata import distributions
from pathlib import Path
from uuid import uuid4

import numpy as np
import torch
from tqdm import tqdm

from chaser_runner.agent import Agent, FrozenPolicy, make_agent
from chaser_runner.config import Config
from chaser_runner.env import ChaseEnv
from chaser_runner.evaluation import BaselinePolicy, EvalResult, benchmark, evaluate
from chaser_runner.logging_utils import MetricLogger, NullLogger, RunLogger
from chaser_runner.persistence import (
    SCHEMA_VERSION,
    atomic_json,
    atomic_save,
    checkpoint_path,
    isolated_rng,
    read_checkpoint,
    restore_rng,
    rng_state,
    run_lock,
    seed_all,
)

__all__ = ["Trainer", "EvalResult", "evaluate", "make_agents", "resolve_device", "set_seed"]
set_seed = seed_all


def resolve_device(device: str = "auto") -> str:
    return ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device


def make_agents(config: Config, device: str) -> tuple[Agent, Agent]:
    return (
        make_agent(config.agent, device=device, name="chaser"),
        make_agent(config.agent, device=device, name="runner"),
    )


class Trainer:
    """Alternate learner roles against episode-frozen historical or baseline policies.

    Only the active learner stores experience and updates. A time-limit transition
    bootstraps; a capture does not. All step counts are policy decisions, whereas
    episode lengths are physics ticks.
    """

    def __init__(
        self,
        config: Config,
        *,
        device="auto",
        logger: MetricLogger | None = None,
        run_dir: str | Path = "runs/default",
        _restoring=False,
    ):
        config.__post_init__()
        self.config, self.device = config, resolve_device(device)
        self.logger = logger or NullLogger()
        self.run_dir = Path(run_dir)
        if not _restoring:
            try:
                self.run_dir.mkdir(parents=True, exist_ok=False)
            except FileExistsError as error:
                raise ValueError(
                    f"Run already exists: {self.run_dir}. Use --resume or a new run name."
                ) from error
        torch.set_num_threads(config.train.torch_threads)
        seed_all(config.train.seed)
        self.env = ChaseEnv(config.env)
        self.chaser, self.runner = make_agents(config, self.device)
        self.agents = {"chaser": self.chaser, "runner": self.runner}
        self.global_step = self.episodes = 0
        self.ep_reward = {"chaser": 0.0, "runner": 0.0}
        self.losses = {role: deque(maxlen=200) for role in self.agents}
        self.pool: list[dict] = []
        self.best_scores: dict[str, list[float]] = {}
        self.last_evaluation: dict = {}
        self.evaluation_step: int | None = None
        self.stop_requested = False
        if not _restoring:
            self._snapshot()
            self._start_episode()
            self._provenance()

    def _provenance(self, directory: Path | None = None) -> None:
        directory = directory or self.run_dir
        directory.mkdir(parents=True, exist_ok=True)
        atomic_json(self.config.to_dict(), directory / "config.json")
        try:
            revision = subprocess.run(
                ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
            ).stdout.strip()
            dirty = bool(
                subprocess.run(
                    ["git", "status", "--porcelain"], capture_output=True, text=True, check=True
                ).stdout.strip()
            )
        except (OSError, subprocess.CalledProcessError):
            revision, dirty = None, None
        atomic_json(
            {
                "schema_version": SCHEMA_VERSION,
                "git_revision": revision,
                "git_dirty": dirty,
                "python": platform.python_version(),
                "torch": str(torch.__version__),
                "numpy": np.__version__,
                "device": self.device,
                "threads": self.config.train.torch_threads,
                "starting_step": self.global_step,
                "packages": {
                    d.metadata["Name"]: d.version for d in distributions() if d.metadata["Name"]
                },
            },
            directory / "metadata.json",
        )
        # Preserve the actual implementation even when the worktree is dirty.
        package = Path(__file__).parent
        with zipfile.ZipFile(directory / "source.zip", "w", zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(package.glob("*.py")):
                archive.write(path, f"chaser_runner/{path.name}")
            project = package.parent.parent / "pyproject.toml"
            if project.exists():
                archive.write(project, "pyproject.toml")

    def _snapshot(self) -> None:
        self.pool.append(
            {
                "step": self.global_step,
                **{role: a.policy_state() for role, a in self.agents.items()},
            }
        )
        self.pool = self.pool[-self.config.train.opponent_pool_size :]

    def _build_opponent(self) -> None:
        opponent_role = "runner" if self.learner_role == "chaser" else "chaser"
        if self.opponent_spec["kind"] == "snapshot":
            # Network initialization must not consume the learner's RNG stream.
            with isolated_rng(0):
                self.opponent = FrozenPolicy(
                    self.config.agent, self.opponent_spec["state"], self.device
                )
        else:
            self.opponent = BaselinePolicy(
                opponent_role, self.config.env, self.opponent_spec["kind"]
            )

    def _start_episode(self) -> None:
        self.learner_role = "chaser" if self.episodes % 2 == 0 else "runner"
        opposite = "runner" if self.learner_role == "chaser" else "chaser"
        draw = np.random.random()
        tc = self.config.train
        if draw < tc.random_opponent_prob:
            self.opponent_spec = {"kind": "random"}
        elif draw < tc.random_opponent_prob + tc.scripted_opponent_prob:
            self.opponent_spec = {"kind": "scripted"}
        else:
            snapshot = self.pool[np.random.randint(len(self.pool))]
            self.opponent_spec = {
                "kind": "snapshot",
                "step": snapshot["step"],
                "state": snapshot[opposite],
            }
        self._build_opponent()
        self.state = self.env.reset(seed=self.config.train.seed if self.global_step == 0 else None)
        self.ep_reward = {"chaser": 0.0, "runner": 0.0}

    def _epsilon(self, step: int) -> float:
        tc = self.config.train
        fraction = min(1.0, step / max(1, tc.eps_decay_steps))
        return tc.eps_start + (tc.eps_final - tc.eps_start) * fraction

    def request_stop(self) -> None:
        self.stop_requested = True

    def _training_step(self) -> None:
        learner = self.agents[self.learner_role]
        eps = self._epsilon(self.global_step)
        if learner.env_steps < self.config.agent.start_steps:
            action = np.random.uniform(-1, 1, 2).astype(np.float32)
        else:
            action = learner.select_action(self.state, explore_eps=eps)
        opposite_action = self.opponent.select_action(self.state)
        ach, arn = (
            (action, opposite_action)
            if self.learner_role == "chaser"
            else (opposite_action, action)
        )
        next_state, rewards, done, info = self.env.step(ach, arn)
        reward = rewards[0 if self.learner_role == "chaser" else 1]
        learner.store(self.state, action, reward, next_state, float(info["terminated"]))
        for _ in range(self.config.train.train_iters_per_step):
            loss = learner.train_once()
            if loss is not None:
                if not np.isfinite(loss):
                    raise FloatingPointError(
                        "Non-finite learner loss; previous checkpoint preserved"
                    )
                self.losses[self.learner_role].append(loss)
        self.state = next_state
        self.global_step += 1
        for role, reward in zip(("chaser", "runner"), rewards, strict=True):
            self.ep_reward[role] += reward
        if done:
            self.episodes += 1
            self.logger.log_scalars(
                "episode",
                {
                    "reward_chaser": self.ep_reward["chaser"],
                    "reward_runner": self.ep_reward["runner"],
                    "length": self.env.steps,
                    "captured": float(info["terminated"]),
                    "learner_chaser": float(self.learner_role == "chaser"),
                },
                self.global_step,
            )
            self._start_episode()

    def policy_bundle(self) -> dict:
        return {
            "schema_version": SCHEMA_VERSION,
            "kind": "policy",
            "step": self.global_step,
            "config": self.config.to_dict(),
            "evaluation": self.last_evaluation,
            "evaluation_step": self.evaluation_step,
            "policies": {role: a.policy_state() for role, a in self.agents.items()},
        }

    def save(self) -> Path:
        path = self.run_dir / "checkpoints" / f"step_{self.global_step:09d}.pt"
        if self.global_step == 0 and not (self.run_dir / "policies" / "initial.pt").exists():
            atomic_save(self.policy_bundle(), self.run_dir / "policies" / "initial.pt")
        data = {
            "schema_version": SCHEMA_VERSION,
            "kind": "training",
            "config": self.config.to_dict(),
            "device": self.device,
            "step": self.global_step,
            "episodes": self.episodes,
            "env": self.env.state_dict(),
            "agents": {role: a.training_state() for role, a in self.agents.items()},
            "learner_role": self.learner_role,
            "opponent": self.opponent_spec,
            "ep_reward": self.ep_reward,
            "pool": self.pool,
            "best_scores": self.best_scores,
            "evaluation": self.last_evaluation,
            "evaluation_step": self.evaluation_step,
            "losses": {k: list(v) for k, v in self.losses.items()},
            "rng": rng_state(),
        }
        atomic_save(data, path)
        atomic_save(self.policy_bundle(), self.run_dir / "policies" / "latest.pt")
        atomic_json(
            {"checkpoint": path.relative_to(self.run_dir).as_posix(), "step": self.global_step},
            self.run_dir / "latest.json",
        )
        # Prune only this run's numbered training checkpoints, after publishing the new pointer.
        paths = sorted((self.run_dir / "checkpoints").glob("step_*.pt"))
        for old in paths[: -self.config.train.keep_checkpoints]:
            old.unlink()
        return path

    @classmethod
    def resume(cls, path, *, device="auto", logger=None) -> Trainer:
        actual = checkpoint_path(path)
        data = read_checkpoint(actual)
        if data["kind"] != "training":
            raise ValueError(
                "Policy exports cannot resume training; use a full training checkpoint"
            )
        run_dir = actual.parent.parent
        if actual.parent.name != "checkpoints" or not (run_dir / "latest.json").exists():
            raise ValueError(
                "Resume requires RUN/checkpoints/*.pt; preserve the complete run directory"
            )
        pointer = run_dir / "latest.json"
        if pointer.exists() and checkpoint_path(run_dir).resolve() != actual.resolve():
            raise ValueError("Resume the latest checkpoint to avoid overwriting a newer run")
        requested_device = data["device"] if device == "auto" else device
        if requested_device.startswith("cuda") and not torch.cuda.is_available():
            raise ValueError("Checkpoint used CUDA; pass --device cpu to migrate it to CPU")
        trainer = cls(
            Config.from_dict(data["config"]),
            device=requested_device,
            logger=logger,
            run_dir=run_dir,
            _restoring=True,
        )
        for role, state in data["agents"].items():
            trainer.agents[role].restore_training_state(state)
        trainer.global_step, trainer.episodes = data["step"], data["episodes"]
        trainer.env.load_state_dict(data["env"])
        trainer.state = trainer.env._get_state()
        trainer.learner_role, trainer.opponent_spec = data["learner_role"], data["opponent"]
        trainer.pool, trainer.ep_reward = data["pool"], data["ep_reward"]
        trainer.best_scores, trainer.last_evaluation = data["best_scores"], data["evaluation"]
        trainer.evaluation_step = data.get("evaluation_step")
        trainer.losses = {k: deque(v, maxlen=200) for k, v in data["losses"].items()}
        trainer._build_opponent()
        restore_rng(data["rng"])
        trainer._resumed_step = trainer.global_step
        return trainer

    def run_evaluation(self) -> dict:
        tc = self.config.train
        result = benchmark(
            self.config, self.chaser, self.runner, episodes=tc.eval_episodes, seed=tc.eval_seed
        )
        self.last_evaluation = result
        self.evaluation_step = self.global_step
        atomic_json(
            {"step": self.global_step, "seed": tc.eval_seed, "matchups": result},
            self.run_dir / "evaluations" / f"step_{self.global_step:09d}.json",
        )
        for name, metrics in result.items():
            self.logger.log_scalars(f"eval/{name}", metrics, self.global_step)
        ch = result["chaser_vs_scripted"]
        rn = result["scripted_vs_runner"]
        ch_random = result["chaser_vs_random"]
        rn_random = result["random_vs_runner"]
        scores = {
            "chaser": [
                (ch["capture_rate"] + ch_random["capture_rate"]) / 2,
                -(ch["avg_distance"] + ch_random["avg_distance"]) / 2,
            ],
            "runner": [
                (rn["survival_rate"] + rn_random["survival_rate"]) / 2,
                rn["avg_length"],
                rn["avg_distance"],
            ],
        }
        for role, score in scores.items():
            if role not in self.best_scores or score > self.best_scores[role]:
                self.best_scores[role] = score
                atomic_save(self.policy_bundle(), self.run_dir / "policies" / f"best_{role}.pt")
        tqdm.write(
            f"[eval {self.global_step}] chaser vs scripted: {ch['capture_rate']:.0%} capture; "
            f"runner vs scripted: {rn['survival_rate']:.0%} survival"
        )
        return result

    def train(self, on_step=None) -> None:
        with run_lock(self.run_dir):
            if hasattr(self, "_resumed_step"):
                latest = read_checkpoint(self.run_dir)
                if latest["step"] != self._resumed_step:
                    raise ValueError("Run advanced since loading; reload its latest checkpoint")
            backend = self.logger
            self.logger = RunLogger(self.run_dir / "metrics.jsonl", backend, self.global_step)
            try:
                self._provenance(self.run_dir / "sessions" / uuid4().hex)
                self._train(on_step)
                self._resumed_step = self.global_step
            finally:
                self.logger.close()
                self.logger = backend

    def _train(self, on_step=None) -> None:
        self.config.__post_init__()
        if self.config.train.total_steps < self.global_step:
            raise ValueError("Total target steps cannot be smaller than resumed progress")
        self.stop_requested = False
        self.logger.log_hparams(self.config.flat())
        atomic_json(self.config.to_dict(), self.run_dir / "config.json")
        previous_handler = None
        if threading.current_thread() is threading.main_thread():
            previous_handler = signal.signal(signal.SIGINT, lambda *_: self.request_stop())
        pbar = tqdm(
            total=self.config.train.total_steps,
            initial=self.global_step,
            desc="training",
            unit="decision",
            dynamic_ncols=True,
        )
        status = "failed"
        try:
            atomic_json(
                {"status": "running", "step": self.global_step}, self.run_dir / "status.json"
            )
            # A safe recovery point exists even if initial evaluation fails.
            self.save()
            if not self.last_evaluation and self.config.train.eval_interval:
                self.run_evaluation()
                self.save()
            while self.global_step < self.config.train.total_steps and not self.stop_requested:
                self._training_step()
                step, tc = self.global_step, self.config.train
                pbar.update(1)
                if step % tc.snapshot_interval == 0:
                    self._snapshot()
                if step % tc.log_interval == 0:
                    self.logger.log_scalars(
                        "train",
                        {
                            "exploration": self._epsilon(step),
                            **{
                                f"loss_{role}": float(np.mean(v)) if v else 0.0
                                for role, v in self.losses.items()
                            },
                            **{f"updates_{role}": a.train_steps for role, a in self.agents.items()},
                            "distance": self.env.distance,
                        },
                        step,
                    )
                    pbar.set_postfix(episodes=self.episodes, learner=self.learner_role)
                if tc.eval_interval and step % tc.eval_interval == 0:
                    self.run_evaluation()
                if step % tc.save_interval == 0:
                    self.save()
                if on_step is not None:
                    on_step(
                        self.env,
                        {
                            "step": str(step),
                            "learner": self.learner_role,
                            "exploration": f"{self._epsilon(step):.3f}",
                        },
                    )
            final_status = "interrupted" if self.stop_requested else "completed"
            if final_status == "completed" and self.config.train.eval_interval:
                if self.evaluation_step != self.global_step:
                    self.run_evaluation()
            self.save()
            status = final_status
        finally:
            pbar.close()
            if previous_handler is not None:
                signal.signal(signal.SIGINT, previous_handler)
            atomic_json(
                {"status": status, "step": self.global_step, "episodes": self.episodes},
                self.run_dir / "status.json",
            )
