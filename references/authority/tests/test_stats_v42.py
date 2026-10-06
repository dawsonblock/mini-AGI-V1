import pytest
from minagi.stats import paired_bootstrap
from minagi.growth_lab import decide_paired


def test_paired_bootstrap_is_deterministic_and_detects_gain():
    control=[0.0]*12
    candidate=[0.2,0.3,0.25,0.21,0.19,0.28,0.24,0.22,0.27,0.23,0.26,0.2]
    a=paired_bootstrap(candidate,control,seed=7,resamples=1000)
    b=paired_bootstrap(candidate,control,seed=7,resamples=1000)
    assert a == b
    assert a.n == 12 and a.mean_delta > 0.2 and a.lcb > 0


def test_paired_growth_requires_enough_evidence_and_lcb():
    r=decide_paired([0]*8,[0.1]*8,min_causal_gain=.05,min_n=8,seed=1)
    assert r.promote and r.paired_evidence['lcb'] >= .05
    r2=decide_paired([0]*3,[0.1]*3,min_causal_gain=.05,min_n=8,seed=1)
    assert not r2.promote and 'below required' in r2.reason


def test_paired_bootstrap_rejects_mismatched_pairs():
    with pytest.raises(ValueError):
        paired_bootstrap([1,2],[1])
