import torch
from minagi.asymspec import (
    js_divergence_from_logits, effective_threshold, asym_spec_decision,
)


def test_asymmetric_context_signal_can_redirect_rejected_token():
    # Verifier on compressed context prefers token 0. Full-context drafter gains
    # a strong preference for token 2 relative to the same drafter on compressed
    # context. A rejected draft is therefore redirected by delta fusion.
    t=torch.tensor([[4.0,0.0,1.0]])
    a=torch.tensor([[0.0,0.0,5.0]])
    b=torch.tensor([[0.0,0.0,0.0]])
    draft=torch.tensor([1])
    d=asym_spec_decision(t,a,b,draft,gamma=1.0,beta=1.0)
    assert not bool(d.accepted[0])
    assert int(d.emitted[0])==2
    assert float(d.divergence[0])>0


def test_jsd_and_threshold_are_bounded():
    a=torch.tensor([[10.0,-10.0]])
    b=torch.tensor([[-10.0,10.0]])
    d=js_divergence_from_logits(a,b)
    assert 0 <= float(d[0]) <= 0.6932
    g=effective_threshold(0.6,d)
    assert 0.3 <= float(g[0]) <= 0.6
