import torch
from minagi.v4 import (
    RiskFeatures, RiskObservation, RiskTolerance, fit_risk_model,
    compare_oracle, RiskModelArtifact,
)


def _f(x):
    return RiskFeatures(x,x*.5,x*.2,x*.1,x*.4,2,512,1 if x>.5 else 0)


def test_risk_model_fits_oracle_labels_and_roundtrips(tmp_path):
    tol=RiskTolerance(max_state_relative_l2=.05,max_state_angle_degrees=10,require_top1_agreement=False)
    obs=[]
    exact=torch.ones(8)
    for x in (0.0,.05,.1,.2,.8,1.0,1.2,1.5):
        approx=exact.clone(); approx[0]+=x
        cmp=compare_oracle(approx,exact)
        obs.append(RiskObservation.from_oracle(_f(x),cmp,tol))
    model=fit_risk_model(obs,tolerance=tol,steps=1200,lr=.2)
    assert model.router().probability(_f(1.5)) > model.router().probability(_f(.01))
    path=tmp_path/'risk.json'; model.save(path)
    loaded=RiskModelArtifact.load(path)
    assert loaded.observation_count==len(obs)
    assert loaded.router().probability(_f(1.0)) == model.router().probability(_f(1.0))
