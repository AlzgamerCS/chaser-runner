from pathlib import Path

from chaser_runner.config import Config, load_config

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"


def test_defaults():
    cfg = Config()
    assert cfg.env.world_size == 800.0
    assert cfg.agent.hidden == 128
    assert cfg.train.total_steps > 0


def test_dt_derived_from_tickrate():
    cfg = Config()
    assert cfg.env.dt == 1.0 / cfg.env.tickrate


def test_from_dict_partial_override():
    cfg = Config.from_dict({"env": {"world_size": 500.0}, "train": {"seed": 7}})
    assert cfg.env.world_size == 500.0
    assert cfg.train.seed == 7
    assert cfg.agent.hidden == 128  # untouched default


def test_default_yaml_loads():
    cfg = load_config(CONFIG_PATH)
    assert isinstance(cfg, Config)
    assert cfg.env.capture_radius == 25.0
    assert cfg.env.chaser_advantage == 1.2


def test_flat_view():
    flat = Config().flat()
    assert flat["env.world_size"] == 800.0
    assert "agent.lr" in flat


def test_yaml_matches_all_defaults():
    assert load_config(CONFIG_PATH) == Config()


def test_invalid_config_fails_early():
    import pytest

    for raw in (
        {"train": {"log_interval": 0}},
        {"agent": {"algorithm": "unknown"}},
        {"env": {"action_repeat": 1.5}},
        {"train": {"seed": True}},
        {"env": {"world_size": float("nan")}},
        {"train": {"random_opponent_prob": 0.8, "scripted_opponent_prob": 0.5}},
        {"env": {"unknown": 1}},
        {"agent": []},
        {"typo": {}},
    ):
        with pytest.raises(ValueError):
            Config.from_dict(raw)
