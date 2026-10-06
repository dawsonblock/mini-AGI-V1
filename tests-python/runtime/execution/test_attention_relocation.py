import numpy as np
from kvcontinual.execution.attention.relocation import relocate_rope_keys, rope_apply, rope_digest


def test_rope_relocation_equals_fresh_rotation():
    rng=np.random.default_rng(8)
    t,h,d=11,3,8
    raw=rng.normal(size=(t,h,d)).astype(np.float64)
    inv=1.0/(10000.0 ** (np.arange(0,d,2)/d))
    old=np.arange(t)
    new=np.array([0,1,5,6,7,20,21,22,40,41,42])
    old_rot=rope_apply(raw,old,inv)
    relocated=relocate_rope_keys(old_rot,old,new,inv)
    fresh=rope_apply(raw,new,inv)
    assert np.allclose(relocated,fresh,rtol=1e-12,atol=1e-12)
    assert rope_digest(inv).startswith("sha256:")
