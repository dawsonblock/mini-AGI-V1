import numpy as np
import pytest
from kvcontinual.execution.cache.block import ExecutionArtifact, RecurrentTailArtifact
from kvcontinual.execution.recurrent.affine import AffineSummary
from kvcontinual.execution.types import ExecutionIdentity, ModelIdentity


def test_artifact_rejects_mixed_seam_contracts():
    i=ExecutionIdentity(ModelIdentity("b","a","t","x","r"),"abi","rec","lay","bf16","fp16")
    tail=RecurrentTailArtifact(0,0,32,AffineSummary(np.eye(2),np.zeros((2,2))))
    a=ExecutionArtifact("s",i,8,{(0,0):tail})
    with pytest.raises(ValueError): a.validate()
