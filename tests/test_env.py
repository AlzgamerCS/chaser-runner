import numpy as np

from chaser_runner.config import EnvConfig
from chaser_runner.env import ACTION_DIM, STATE_DIM, ChaseEnv


def test_reset_state_shape():
    env = ChaseEnv()
    state = env.reset(seed=0)
    assert state.shape == (STATE_DIM,)
    assert state.dtype == np.float32


def test_state_is_relative_and_normalized():
    env = ChaseEnv()
    state = env.reset(seed=0)
    # Relative displacement components live in [-1, 1] and velocities are small.
    assert np.all(np.abs(state) <= 1.0 + 1e-6)


def test_state_is_translation_invariant():
    # Shifting both agents by the same offset (mod world) must not change the
    # relative state -- the whole point of the relative encoding.
    cfg = EnvConfig(world_size=100.0)
    env = ChaseEnv(cfg)
    env.reset(seed=0)
    s1 = env._get_state()
    offset = np.array([37.0, 91.0])
    env.ch_pos = env._wrap(env.ch_pos + offset)
    env.rn_pos = env._wrap(env.rn_pos + offset)
    s2 = env._get_state()
    np.testing.assert_allclose(s1, s2, atol=1e-5)


def test_step_shapes_and_types():
    env = ChaseEnv()
    env.reset(seed=1)
    a = np.zeros(ACTION_DIM)
    state, (r_ch, r_rn), done, info = env.step(a, a)
    assert state.shape == (STATE_DIM,)
    assert isinstance(r_ch, float) and isinstance(r_rn, float)
    assert isinstance(done, bool)
    assert "distance" in info


def test_reward_is_zero_sum_before_capture():
    env = ChaseEnv()
    env.reset(seed=2)
    a = np.random.uniform(-1, 1, size=ACTION_DIM)
    _, (r_ch, r_rn), done, _ = env.step(a, a)
    if not done:  # capture adds an asymmetric bonus
        assert r_ch == -r_rn


def test_seed_determinism():
    env1, env2 = ChaseEnv(), ChaseEnv()
    s1, s2 = env1.reset(seed=42), env2.reset(seed=42)
    np.testing.assert_array_equal(s1, s2)


def test_positions_wrap_within_world():
    env = ChaseEnv(EnvConfig(world_size=100.0, max_speed=500.0, max_acc=500.0))
    env.reset(seed=3)
    a = np.ones(ACTION_DIM)
    for _ in range(200):
        env.step(a, a)
    assert (env.ch_pos >= 0).all() and (env.ch_pos < 100.0).all()
    assert (env.rn_pos >= 0).all() and (env.rn_pos < 100.0).all()


def test_torus_vector_takes_short_way_around():
    env = ChaseEnv(EnvConfig(world_size=100.0))
    d = env._torus_vector(np.array([1.0, 1.0]), np.array([99.0, 1.0]))
    # 99 is nearly a full lap away; the short way is -2, not +98.
    assert np.isclose(d[0], -2.0)


def test_capture_terminates_episode():
    env = ChaseEnv(EnvConfig(capture_radius=1e9))  # everything is "captured"
    env.reset(seed=4)
    _, (r_ch, r_rn), done, info = env.step(np.zeros(ACTION_DIM), np.zeros(ACTION_DIM))
    assert done and info["reason"] == "captured"
    assert r_ch > 0 and r_rn < 0


def test_timeout_terminates_episode():
    env = ChaseEnv(EnvConfig(capture_radius=0.0, timeout_steps=3))
    env.reset(seed=5)
    done = False
    for _ in range(3):
        _, _, done, info = env.step(np.zeros(ACTION_DIM), np.zeros(ACTION_DIM))
    assert done and info["reason"] == "timeout"
