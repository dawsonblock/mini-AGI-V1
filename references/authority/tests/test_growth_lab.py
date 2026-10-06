from minagi.growth_lab import GrowthArm, decide


def test_paired_growth_requires_causal_gain():
    # both improve equally: temporal correlation is not evidence of growth gain
    d=decide(GrowthArm(1.0,.9),GrowthArm(1.0,.9),min_causal_gain=.01)
    assert not d.promote and abs(d.causal_gain)<1e-9
    d=decide(GrowthArm(1.0,.98),GrowthArm(1.0,.90),min_causal_gain=.05)
    assert d.promote and d.causal_gain > .05


def test_growth_rejects_old_domain_regression():
    d=decide(GrowthArm(1,.99),GrowthArm(1,.8,.2),max_old_domain_regression=.05)
    assert not d.promote
