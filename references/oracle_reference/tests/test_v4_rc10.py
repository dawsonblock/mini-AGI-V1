import torch
from minagi.v4 import TransitionOrientation, summary_from_steps, compose_summaries


def _rand_step(d=4, v=3):
    t = torch.randn(d, d) * 0.1 + torch.eye(d)
    u = torch.randn(d, v) * 0.1
    return t, u


def test_left_affine_summary_matches_direct_recurrence():
    steps = [_rand_step() for _ in range(5)]
    ts, us = zip(*steps)
    s0 = torch.randn(4, 3)
    direct = s0
    for t, u in steps:
        direct = t @ direct + u
    summary = summary_from_steps(ts, us, orientation=TransitionOrientation.LEFT)
    assert torch.allclose(summary.apply(s0), direct, atol=1e-6, rtol=1e-6)


def test_composed_segments_match_direct_recurrence():
    a = [_rand_step() for _ in range(3)]
    b = [_rand_step() for _ in range(4)]
    sa = summary_from_steps(*zip(*a), segment_id="A")
    sb = summary_from_steps(*zip(*b), segment_id="B")
    sab = compose_summaries(sa, sb)
    s0 = torch.randn(4, 3)
    direct = s0
    for t, u in a + b:
        direct = t @ direct + u
    assert torch.allclose(sab.apply(s0), direct, atol=1e-6, rtol=1e-6)


def test_right_orientation_matches_direct():
    d, v = 4, 3
    ts = [torch.randn(d, d) * .1 + torch.eye(d) for _ in range(4)]
    us = [torch.randn(v, d) * .1 for _ in range(4)]
    s0 = torch.randn(v, d)
    direct = s0
    for t, u in zip(ts, us):
        direct = direct @ t + u
    summary = summary_from_steps(ts, us, orientation=TransitionOrientation.RIGHT)
    assert torch.allclose(summary.apply(s0), direct, atol=1e-6, rtol=1e-6)
