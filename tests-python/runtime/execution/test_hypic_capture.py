import numpy as np

from kvcontinual.execution.recurrent.hypic_capture import (
    capture_head_tail,
    capture_segment_tail_heads,
    direct_head_recurrence,
    prepare_gdn_gates,
)


def test_prepare_gdn_gates_is_finite_and_bounded():
    alpha=np.array([[-1000.0, 0.0, 1000.0]])
    beta=np.array([[-1000.0, 0.0, 1000.0]])
    dt=np.array([0.1,-0.2,0.3])
    ssm=np.array([-1.0,-0.5,-0.1])
    g,b=prepare_gdn_gates(alpha,beta,dt,ssm)
    assert np.all(np.isfinite(g))
    assert np.all(np.isfinite(b))
    assert np.all(g > 0.0)
    assert np.all(g <= 1.0)
    assert np.all((b >= 0.0) & (b <= 1.0))


def test_rank1_capture_matches_direct_tail_for_arbitrary_state():
    rng=np.random.default_rng(123)
    T,D=23,8
    k=rng.normal(size=(T,D)); k/=np.linalg.norm(k,axis=1,keepdims=True)
    v=rng.normal(size=(T,D))
    g=np.clip(rng.normal(.94,.015,size=T),.75,.999)
    b=np.clip(rng.normal(.45,.1,size=T),.01,.99)
    seam=8
    summary=capture_head_tail(k,v,g,b,seam_width=seam)
    S0=rng.normal(size=(D,D))
    exact=direct_head_recurrence(S0,k[seam:],v[seam:],g[seam:],b[seam:])
    assert np.allclose(summary.apply(S0), exact, rtol=2e-11, atol=2e-11)


def test_multihead_capture_uses_native_v_to_k_head_mapping():
    rng=np.random.default_rng(5)
    T,K,V,D=17,2,4,4
    k=rng.normal(size=(T,K,D)); k/=np.linalg.norm(k,axis=2,keepdims=True)
    v=rng.normal(size=(T,V,D))
    g=np.clip(rng.normal(.92,.02,size=(T,V)),.7,.999)
    b=np.clip(rng.normal(.4,.12,size=(T,V)),.02,.98)
    seam=3
    captured=capture_segment_tail_heads(k,v,g,b,seam_width=seam)
    assert captured.num_k_heads == K and captured.num_v_heads == V
    for vh in range(V):
        S0=rng.normal(size=(D,D))
        kh=vh%K
        exact=direct_head_recurrence(S0,k[seam:,kh],v[seam:,vh],g[seam:,vh],b[seam:,vh])
        assert np.allclose(captured.summary_for_v_head(vh).apply(S0), exact, rtol=2e-11, atol=2e-11)
