import numpy as np
from kvcontinual.execution.qualification import angle_deg, categorical_kl, relative_l2


def test_oracle_metrics_helpers():
    a=np.array([1.,0.]); b=np.array([1.,0.])
    assert relative_l2(a,b)==0
    assert angle_deg(a,b) < 1e-4
    assert categorical_kl(np.array([1.,0.]),np.array([1.,0.])) < 1e-12
