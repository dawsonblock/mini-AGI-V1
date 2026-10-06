import numpy as np
import pytest
from kvcontinual.execution.recurrent.affine import AffineSummary, compose_sequence
from kvcontinual.execution.types import TransitionOrientation


def test_left_affine_composition_matches_replay():
    rng = np.random.default_rng(42)
    summaries = []
    for _ in range(5):
        T = np.eye(4) * 0.9 + rng.normal(0, 0.01, (4, 4))
        Z = rng.normal(0, 0.05, (4, 4))
        summaries.append(AffineSummary(T, Z))
    s0 = rng.normal(size=(4, 4))
    exact = s0.copy()
    for s in summaries:
        exact = s.apply(exact)
    composed = compose_sequence(summaries).apply(s0)
    assert np.allclose(exact, composed, rtol=1e-12, atol=1e-12)


def test_right_affine_composition_matches_replay():
    rng = np.random.default_rng(2)
    parts=[]
    for _ in range(3):
        parts.append(AffineSummary(np.eye(3)*.95+rng.normal(0,.01,(3,3)), rng.normal(0,.02,(3,3)), TransitionOrientation.RIGHT_MULTIPLY))
    s0=rng.normal(size=(3,3))
    direct=s0.copy()
    for p in parts: direct=p.apply(direct)
    assert np.allclose(compose_sequence(parts).apply(s0), direct)


def test_orientation_mismatch_rejected():
    a=AffineSummary(np.eye(2),np.zeros((2,2)),TransitionOrientation.LEFT_MULTIPLY)
    b=AffineSummary(np.eye(2),np.zeros((2,2)),TransitionOrientation.RIGHT_MULTIPLY)
    with pytest.raises(ValueError): a.then(b)
