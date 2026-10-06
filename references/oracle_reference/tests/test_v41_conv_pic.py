import torch
from minagi.v4 import causal_depthwise_conv, capture_boundary, FAPICSegment, relocate_fa_pic, append_fa_pic


def test_causal_conv_boundary_continuation_matches_full_sequence():
    torch.manual_seed(5)
    x=torch.randn(2,19,4)
    w=torch.randn(4,5)*.1
    b=torch.randn(4)*.01
    full,_=causal_depthwise_conv(x,w,b)
    first,state=causal_depthwise_conv(x[:,:11],w,b)
    second,state2=causal_depthwise_conv(x[:,11:],w,b,boundary=state)
    joined=torch.cat([first,second],dim=1)
    assert torch.allclose(joined,full,atol=1e-6,rtol=1e-6)
    assert state2.history.shape==(2,4,4)


def test_capture_boundary_zero_pads_short_history():
    x=torch.arange(6,dtype=torch.float32).reshape(1,2,3)
    state=capture_boundary(x,4)
    assert state.history.shape==(1,3,3)
    assert torch.equal(state.history[...,0],torch.zeros(1,3))


def test_fa_pic_relocation_and_skip_are_contiguous():
    k=torch.randn(2,7,4); v=torch.randn(2,7,4)
    a=FAPICSegment(k,v,tuple(range(100,107)))
    ra=relocate_fa_pic(a,20,skip=2)
    assert ra.logical_positions==(20,21,22,23,24)
    assert ra.source_positions==(102,103,104,105,106)
    b=FAPICSegment(torch.randn(2,3,4),torch.randn(2,3,4),(9,10,11))
    rb=relocate_fa_pic(b,25)
    joined=append_fa_pic([ra,rb])
    assert joined.logical_positions==tuple(range(20,28))
    assert joined.key.shape[-2]==8
