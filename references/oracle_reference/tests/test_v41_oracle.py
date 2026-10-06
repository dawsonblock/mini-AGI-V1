import torch
from minagi.v4 import tensor_divergence, compare_oracle


def test_oracle_metrics_zero_for_identical_state_and_logits():
    x=torch.randn(3,4)
    logits=torch.randn(2,11)
    d=tensor_divergence(x,x)
    assert d.relative_l2==0.0
    assert d.max_abs==0.0
    c=compare_oracle(x,x,approx_logits=logits,exact_logits=logits)
    assert c.logit_kl is not None and abs(c.logit_kl)<1e-7
    assert c.top1_agree is True


def test_oracle_detects_perturbation():
    x=torch.ones(4)
    y=x.clone(); y[0]+=0.5
    d=tensor_divergence(y,x)
    assert d.relative_l2>0.2
    assert d.angle_degrees>0
