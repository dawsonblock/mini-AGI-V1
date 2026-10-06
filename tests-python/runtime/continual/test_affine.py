import numpy as np
from kvcontinual.continual.recurrent.affine import AffineSummary, compose_sequence


def test_affine_composition_matches_replay():
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
