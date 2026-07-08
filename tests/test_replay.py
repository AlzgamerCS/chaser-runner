import numpy as np

from chaser_runner.replay import ReplayBuffer


def _add_n(buf: ReplayBuffer, n: int) -> None:
    for i in range(n):
        s = np.full(8, i, dtype=np.float32)
        a = np.zeros(2, dtype=np.float32)
        buf.add(s, a, float(i), s, 0.0)


def test_len_and_capacity():
    buf = ReplayBuffer(capacity=5)
    _add_n(buf, 10)
    assert len(buf) == 5  # oldest transitions evicted


def test_sample_shapes():
    buf = ReplayBuffer(capacity=100)
    _add_n(buf, 50)
    s, a, r, s2, d = buf.sample(16)
    assert s.shape == (16, 8)
    assert a.shape == (16, 2)
    assert r.shape == (16, 1)
    assert s2.shape == (16, 8)
    assert d.shape == (16, 1)
