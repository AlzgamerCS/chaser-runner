import numpy as np

from chaser_runner.agent import DQNAgent
from chaser_runner.config import AgentConfig
from chaser_runner.env import ACTION_DIM, STATE_DIM


def _small_agent() -> DQNAgent:
    cfg = AgentConfig(hidden=16, batch_size=8, min_buffer=8, target_samples=4)
    return DQNAgent(cfg, device="cpu")


def test_select_action_shape_and_range():
    agent = _small_agent()
    action = agent.select_action(np.zeros(STATE_DIM, dtype=np.float32))
    assert action.shape == (ACTION_DIM,)
    assert (action >= -1).all() and (action <= 1).all()


def test_train_once_returns_none_when_buffer_small():
    agent = _small_agent()
    assert agent.train_once() is None


def test_train_once_runs_after_enough_data():
    agent = _small_agent()
    s = np.zeros(STATE_DIM, dtype=np.float32)
    a = np.zeros(ACTION_DIM, dtype=np.float32)
    for _ in range(32):
        agent.store(s, a, 1.0, s, 0.0)
    loss = agent.train_once()
    assert isinstance(loss, float) and loss >= 0.0


def test_save_and_load_roundtrip(tmp_path):
    agent = _small_agent()
    path = tmp_path / "q.pth"
    agent.save(path)

    other = _small_agent()
    other.load(path)
    for p1, p2 in zip(agent.q.parameters(), other.q.parameters(), strict=True):
        assert (p1 == p2).all()
