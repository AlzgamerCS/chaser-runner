import numpy as np
import torch

from chaser_runner.agent import TD3Agent
from chaser_runner.config import AgentConfig
from chaser_runner.persistence import seed_all


def test_td3_learns_a_known_continuous_control_objective():
    # gamma=0 isolates the continuous-action learner from exploration and self-play.
    torch.set_num_threads(1)
    seed_all(42)
    agent = TD3Agent(
        AgentConfig(
            hidden=32, lr=0.001, gamma=0, batch_size=64, min_buffer=64, buffer_capacity=3000
        )
    )
    states = np.random.uniform(-1, 1, (2000, 6)).astype(np.float32)
    actions = np.random.uniform(-1, 1, (2000, 2)).astype(np.float32)
    rewards = -np.sum((actions - states[:, :2]) ** 2, axis=1)
    for s, a, r in zip(states, actions, rewards, strict=True):
        agent.store(s, a, r, s, 1)
    validation = torch.tensor(states[:100])
    with torch.no_grad():
        initial = torch.mean((agent.actor(validation) - validation[:, :2]) ** 2).item()
    for _ in range(600):
        assert np.isfinite(agent.train_once())
    with torch.no_grad():
        final = torch.mean((agent.actor(validation) - validation[:, :2]) ** 2).item()
    assert final < initial * 0.3, (initial, final)
    action = agent.select_action(states[0])
    np.testing.assert_array_equal(action, agent.select_action(states[0]))
    assert (np.abs(action) <= 1).all()
