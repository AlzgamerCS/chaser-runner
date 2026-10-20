import json

import pytest
import yaml

from chaser_runner.cli import main
from chaser_runner.config import AgentConfig, Config, EnvConfig, TrainConfig
from chaser_runner.persistence import read_checkpoint


def test_cli_train_resume_evaluate_and_record(tmp_path):
    cfg = Config(
        env=EnvConfig(capture_radius=0, timeout_steps=12),
        agent=AgentConfig(hidden=16, batch_size=4, min_buffer=4, start_steps=4),
        train=TrainConfig(total_steps=4, eval_interval=0),
    )
    config_path = tmp_path / "input.yaml"
    config_path.write_text(yaml.safe_dump(cfg.to_dict()))
    run = tmp_path / "cli"
    main(
        [
            "train",
            "--config",
            str(config_path),
            "--logdir",
            str(tmp_path),
            "--run-name",
            "cli",
        ]
    )
    assert read_checkpoint(run)["step"] == 4
    main(["train", "--resume", str(run), "--steps", "8"])
    assert read_checkpoint(run)["step"] == 8
    history = (run / "metrics.jsonl").read_text()
    assert history.count('"event": "session"') == 2
    assert list((run / "tensorboard").glob("events.*"))
    output = tmp_path / "result.json"
    main(
        [
            "eval",
            "--checkpoint",
            str(run),
            "--episodes",
            "2",
            "--benchmark",
            "--seed",
            "20000",
            "--out",
            str(output),
        ]
    )
    data = json.loads(output.read_text())
    assert len(data["matchups"]) == 5
    assert data["seed"] == 20000
    assert data["config"]["env"] == cfg.to_dict()["env"]
    gif = tmp_path / "demo.gif"
    main(
        [
            "record",
            "--checkpoint",
            str(run),
            "--episodes",
            "1",
            "--max-frames",
            "2",
            "--out",
            str(gif),
        ]
    )
    assert gif.read_bytes().startswith(b"GIF")
    with pytest.raises(SystemExit) as error:
        main(["train", "--resume", str(run), "--config", str(config_path), "--no-log"])
    assert error.value.code == 2


def test_cli_rejects_ambiguous_policy_loading():
    with pytest.raises(SystemExit) as error:
        main(["eval", "--episodes", "1"])
    assert error.value.code == 2
