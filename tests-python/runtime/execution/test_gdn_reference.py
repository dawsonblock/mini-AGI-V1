import numpy as np
from kvcontinual.execution.recurrent.gdn_reference import capture_segment_tail, direct_recurrence


def test_gdn_affine_tail_matches_direct_same_inputs():
    rng=np.random.default_rng(7)
    T,D=20,4
    k=rng.normal(size=(T,D)); k/=np.linalg.norm(k,axis=1,keepdims=True)
    v=rng.normal(size=(T,D))
    decay=np.clip(rng.normal(.93,.02,size=T),.7,.999)
    beta=np.clip(rng.normal(.4,.1,size=T),.05,.95)
    seam=8
    state=rng.normal(size=(D,D))
    direct=direct_recurrence(state,k[seam:],v[seam:],decay[seam:],beta[seam:])
    summary=capture_segment_tail(k,v,decay,beta,seam)
    assert np.allclose(summary.apply(state),direct,rtol=1e-11,atol=1e-11)
