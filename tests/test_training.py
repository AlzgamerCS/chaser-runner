"""Regression tests for continuation, evaluation isolation, and artifact preservation."""

from copy import deepcopy

import numpy as np
import pytest
import torch

from chaser_runner.config import AgentConfig, Config, EnvConfig, TrainConfig
from chaser_runner.evaluation import BaselinePolicy, evaluate
from chaser_runner.persistence import atomic_save, read_checkpoint, restore_rng, rng_state
from chaser_runner.trainer import Trainer


def small_config(algorithm="td3"):
    return Config(
        env=EnvConfig(capture_radius=0, timeout_steps=20),
        agent=AgentConfig(
            algorithm=algorithm,
            hidden=16,
            batch_size=4,
            min_buffer=4,
            start_steps=4,
            buffer_capacity=16,
            sample_actions=4,
            target_samples=4,
        ),
        train=TrainConfig(
            total_steps=47,
            eval_interval=10,
            eval_episodes=2,
            save_interval=13,
            snapshot_interval=7,
            opponent_pool_size=3,
            random_opponent_prob=0,
            scripted_opponent_prob=0,
            keep_checkpoints=2,
        ),
    )


def assert_equal(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, np.ndarray):
        np.testing.assert_array_equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            assert_equal(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for av, bv in zip(a, b, strict=True):
            assert_equal(av, bv)
    else:
        assert a == b


@pytest.mark.parametrize("algorithm", ["td3", "dqn"])
def test_resume_matches_uninterrupted_training_exactly(tmp_path, algorithm):
    cfg = small_config(algorithm)
    full = Trainer(deepcopy(cfg), run_dir=tmp_path / "full")
    full.train()
    expected = read_checkpoint(full.run_dir)
    partial = Trainer(deepcopy(cfg), run_dir=tmp_path / "partial")

    def stop_mid_episode(env, stats):
        if partial.global_step == 17:
            partial.request_stop()

    partial.train(on_step=stop_mid_episode)
    assert partial.global_step == 17
    assert partial.env.steps != 0
    resumed = Trainer.resume(partial.run_dir, device="cpu")
    resumed.train()
    actual = read_checkpoint(resumed.run_dir)
    assert_equal(expected, actual)
    # Replay wrapped, both optimizers updated, snapshots rotated, and eval ran.
    assert all(a.train_steps > 0 for a in resumed.agents.values())
    assert all(len(a.replay) == cfg.agent.buffer_capacity for a in resumed.agents.values())
    assert len(list((resumed.run_dir / "checkpoints").glob("step_*.pt"))) == 2
    assert (resumed.run_dir / "policies" / "best_chaser.pt").exists()
    assert read_checkpoint(resumed.run_dir / "policies" / "initial.pt")["step"] == 0
    assert (resumed.run_dir / "source.zip").exists()
    with pytest.raises(ValueError, match="already exists"):
        Trainer(cfg, run_dir=resumed.run_dir)


@pytest.mark.parametrize("algorithm", ["td3", "dqn"])
def test_evaluation_is_repeatable_and_does_not_change_training_rng(tmp_path, algorithm):
    trainer = Trainer(small_config(algorithm), run_dir=tmp_path / "run")
    state_before = deepcopy(rng_state())
    weights = deepcopy(trainer.chaser.training_state())
    opponent = BaselinePolicy("runner", trainer.config.env, "random")
    first = evaluate(trainer.config, trainer.chaser, opponent, episodes=3, seed=77)
    second = evaluate(trainer.config, trainer.chaser, opponent, episodes=3, seed=77)
    assert first == second
    assert_equal(state_before, rng_state())
    assert_equal(weights, trainer.chaser.training_state())
    assert 0 <= first.capture_ci_low <= first.capture_rate <= first.capture_ci_high <= 1
    with pytest.raises(ValueError, match="positive"):
        evaluate(trainer.config, trainer.chaser, opponent, episodes=0)


def test_time_limit_transition_bootstraps(tmp_path):
    cfg = small_config()
    cfg.env.timeout_steps = 4
    trainer = Trainer(cfg, run_dir=tmp_path / "run")
    trainer._training_step()
    assert trainer.episodes == 1
    assert trainer.chaser.replay.arrays[-1][0, 0] == 0


def test_capture_transition_does_not_bootstrap(tmp_path):
    cfg = small_config()
    trainer = Trainer(cfg, run_dir=tmp_path / "run")
    trainer.env.ch_pos[:] = trainer.env.rn_pos
    trainer.env.ch_vel[:] = trainer.env.rn_vel[:] = 0
    trainer.config.env.capture_radius = 100
    trainer.state = trainer.env._get_state()
    trainer._training_step()
    assert trainer.episodes == 1
    assert trainer.chaser.replay.arrays[-1][0, 0] == 1


def test_failed_atomic_save_keeps_previous_artifact(tmp_path, monkeypatch):
    path = tmp_path / "checkpoint.pt"
    atomic_save({"good": 1}, path)

    def fail(data, fh):
        fh.write(b"incomplete")
        raise OSError("disk error")

    monkeypatch.setattr(torch, "save", fail)
    with pytest.raises(OSError, match="disk error"):
        atomic_save({"bad": 2}, path)
    assert torch.load(path, weights_only=True) == {"good": 1}
    assert list(tmp_path.iterdir()) == [path]


def test_policy_export_cannot_resume_and_old_checkpoint_cannot_overwrite(tmp_path):
    cfg = small_config()
    trainer = Trainer(cfg, run_dir=tmp_path / "run")
    trainer.save()
    older = next((trainer.run_dir / "checkpoints").glob("*.pt"))
    trainer._training_step()
    trainer.save()
    with pytest.raises(ValueError, match="latest"):
        Trainer.resume(older)
    with pytest.raises(ValueError, match="Policy exports"):
        Trainer.resume(trainer.run_dir / "policies" / "latest.pt")


def test_environment_state_restores_trajectory_and_future_resets(tmp_path):
    trainer = Trainer(small_config(), run_dir=tmp_path / "run")
    env = trainer.env
    env.step(np.ones(2), np.zeros(2))
    state = deepcopy(env.state_dict())
    expected = env.step(-np.ones(2), np.ones(2))
    reset = env.reset()
    env.load_state_dict(state)
    assert_equal(expected, env.step(-np.ones(2), np.ones(2)))
    np.testing.assert_array_equal(reset, env.reset())


def test_rng_round_trip():
    state = rng_state()
    expected = (np.random.random(4), torch.rand(4))
    restore_rng(state)
    assert_equal(expected, (np.random.random(4), torch.rand(4)))


def test_run_lock_prevents_simultaneous_training(tmp_path):
    from chaser_runner.persistence import run_lock

    trainer = Trainer(small_config(), run_dir=tmp_path / "run")
    with run_lock(trainer.run_dir):
        with pytest.raises(ValueError, match="already being trained"):
            trainer.train()
    # Releasing the lock allows training and creates a readable metrics history.
    trainer.config.train.total_steps = 1
    trainer.train()
    assert '"event": "session"' in (trainer.run_dir / "metrics.jsonl").read_text()


def test_failure_keeps_recovery_checkpoint_and_reports_failed(tmp_path, monkeypatch):
    import json

    cfg = small_config()
    cfg.train.eval_interval = 0
    trainer = Trainer(cfg, run_dir=tmp_path / "run")

    def fail():
        # Simulate failure after partially changing the environment.
        trainer.env.ch_pos[:] = -123
        raise RuntimeError("training failed")

    monkeypatch.setattr(trainer, "_training_step", fail)
    with pytest.raises(RuntimeError, match="training failed"):
        trainer.train()
    status = json.loads((trainer.run_dir / "status.json").read_text())
    assert status["status"] == "failed"
    saved = read_checkpoint(trainer.run_dir)
    assert saved["step"] == 0
    assert saved["env"]["ch_pos"] != [-123, -123]
    # Failure also released the process lock.
    resumed = Trainer.resume(trainer.run_dir)
    resumed.config.train.total_steps = 1
    resumed.train()
