import torch
from minagi.v4 import (
    GDNInputs, TransitionOrientation, affine_step, delta_step,
    compile_gdn_summary, zero_state_for, SegmentExecution, qualify_rc10_reference,
)


def _steps(n=12, heads=3, dk=4, dv=5):
    torch.manual_seed(13)
    out=[]
    for _ in range(n):
        k=torch.randn(heads,dk)*0.25
        k=k/(k.norm(dim=-1,keepdim=True)+1e-6)
        v=torch.randn(heads,dv)*0.2
        beta=torch.sigmoid(torch.randn(heads))*.8
        decay=torch.sigmoid(torch.randn(heads))*.4+.55
        out.append(GDNInputs(k,v,beta,decay))
    return out


def test_affine_step_matches_direct_delta_left_multihead():
    step=_steps(1)[0]
    state=torch.randn(3,4,5)
    t,u=affine_step(step,orientation=TransitionOrientation.LEFT)
    affine=t@state+u
    direct=delta_step(state,step,orientation=TransitionOrientation.LEFT)
    assert torch.allclose(affine,direct,atol=2e-6,rtol=2e-6)


def test_affine_step_matches_direct_delta_right_multihead():
    step=_steps(1)[0]
    state=torch.randn(3,5,4)
    t,u=affine_step(step,orientation=TransitionOrientation.RIGHT)
    affine=state@t+u
    direct=delta_step(state,step,orientation=TransitionOrientation.RIGHT)
    assert torch.allclose(affine,direct,atol=2e-6,rtol=2e-6)


def test_compiled_suffix_requires_fresh_seam_and_matches_exact():
    steps=_steps(20)
    summary=compile_gdn_summary(steps,seam_width=8,segment_id='A')
    seg=SegmentExecution('A',tuple(steps),summary)
    s0=zero_state_for(steps[0],orientation=TransitionOrientation.LEFT)
    result=qualify_rc10_reference(s0,[seg])
    assert result.comparison.state.relative_l2 < 2e-6
    assert result.comparison.state.max_abs < 2e-6


def test_multiple_reassembled_segments_match_when_layer_inputs_are_fixed():
    a=_steps(18)
    b=_steps(22)
    sa=compile_gdn_summary(a,seam_width=8,segment_id='A')
    sb=compile_gdn_summary(b,seam_width=8,segment_id='B')
    segments=[SegmentExecution('A',tuple(a),sa),SegmentExecution('B',tuple(b),sb)]
    s0=torch.randn(3,4,5)*0.1
    result=qualify_rc10_reference(s0,segments)
    assert result.comparison.state.relative_l2 < 3e-6
