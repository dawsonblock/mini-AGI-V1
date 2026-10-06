import numpy as np
from kvcontinual.execution.recurrent.conv_boundary import ConvBoundaryState, causal_depthwise_conv_seam


def test_conv_boundary_chunking_matches_one_pass():
    rng=np.random.default_rng(4)
    t,c,k=19,5,4
    x=rng.normal(size=(t,c))
    w=rng.normal(size=(c,k))
    h=np.zeros((c,k-1))
    full,_=causal_depthwise_conv_seam(x,w,ConvBoundaryState(h.copy(),k))
    a,st=causal_depthwise_conv_seam(x[:7],w,ConvBoundaryState(h.copy(),k))
    b,st2=causal_depthwise_conv_seam(x[7:],w,st)
    assert np.allclose(np.concatenate([a,b],axis=0),full,rtol=1e-12,atol=1e-12)
    assert st2.history.shape == (c,k-1)


def test_conv_boundary_matches_manual_current_and_history_weights():
    x=np.array([[4.0],[5.0]])
    w=np.array([[1.0,2.0,3.0]])
    st=ConvBoundaryState(np.array([[2.0,3.0]]),3)
    out,final=causal_depthwise_conv_seam(x,w,st,apply_silu=False)
    assert np.allclose(out[:,0],[1*2+2*3+3*4,1*3+2*4+3*5])
    assert np.allclose(final.history,[ [4.0,5.0] ])
