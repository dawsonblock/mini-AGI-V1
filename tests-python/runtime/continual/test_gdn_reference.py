import numpy as np
import pytest

from kvcontinual.continual.recurrent.gdn_reference import (
    GDNToken,
    compose_tokens,
    decay_from_raw,
    readout,
    replay,
    sigmoid,
    softplus,
    state_error,
)


def _tokens(seed: int = 105, dim: int = 8, count: int = 19):
    rng = np.random.default_rng(seed)
    return [
        GDNToken(
            k=rng.normal(0, 0.15, dim),
            v=rng.normal(0, 0.15, dim),
            g=float(rng.uniform(0.80, 0.999)),
            beta=float(rng.uniform(0.05, 0.95)),
        )
        for _ in range(count)
    ]


def test_gdn_block_summary_matches_cuda_algebra_replay():
    tokens = _tokens()
    rng = np.random.default_rng(99)
    s0 = rng.normal(0, 0.1, (8, 8))
    direct = replay(tokens, s0)
    composed = compose_tokens(tokens).apply(s0)
    max_abs, rel_l2 = state_error(composed, direct)
    assert max_abs < 1e-12
    assert rel_l2 < 1e-12


def test_gdn_zero_start_state_is_summary_z():
    tokens = _tokens()
    zero = np.zeros((8, 8))
    summary = compose_tokens(tokens)
    assert np.allclose(summary.Z, replay(tokens, zero), rtol=1e-12, atol=1e-12)


def test_gdn_segment_composition_matches_unsplit_summary():
    tokens = _tokens()
    a = compose_tokens(tokens[:7])
    b = compose_tokens(tokens[7:])
    combined = a.then(b)
    whole = compose_tokens(tokens)
    assert np.allclose(combined.T, whole.T, rtol=1e-12, atol=1e-12)
    assert np.allclose(combined.Z, whole.Z, rtol=1e-12, atol=1e-12)


def test_gdn_gate_preprocessing_and_readout():
    assert sigmoid(1000) == 1.0
    assert sigmoid(-1000) == 0.0
    assert np.isfinite(softplus(1000))
    assert np.isfinite(softplus(-1000))
    g = decay_from_raw(0.25, -0.1, -0.5)
    assert 0.0 < g <= 1.0
    state = replay(_tokens(count=3), np.zeros((8, 8)))
    out = readout(state, np.full(8, 0.25), scale=0.5)
    assert out.shape == (8,)
    assert np.all(np.isfinite(out))


def test_gdn_dimension_mismatch_fails_closed():
    tokens = _tokens(count=3)
    bad = list(tokens)
    bad[-1] = GDNToken(k=np.ones(7), v=np.ones(7), g=0.9, beta=0.5)
    with pytest.raises(ValueError):
        compose_tokens(bad)
