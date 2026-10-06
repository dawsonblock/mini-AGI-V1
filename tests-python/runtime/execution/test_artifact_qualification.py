import numpy as np
from kvcontinual.execution.cache.block import ArtifactQualification, ExecutionArtifact, RecurrentTailArtifact
from kvcontinual.execution.cache.store import ExecutionArtifactStore
from kvcontinual.execution.recurrent.affine import AffineSummary
from kvcontinual.execution.recurrent.interfaces import ApproximateExecution, BackendCapabilities
from kvcontinual.execution.runtime import ReconstructionRuntime
from kvcontinual.execution.source import SourceSegment, SourceSegmentStore
from kvcontinual.execution.types import ExecutionIdentity, ModelIdentity, ReconstructionAction, RuntimeRiskSignals


def ident():
    return ExecutionIdentity(ModelIdentity("b","a","t","arch","rope"),"abi","gdn","layout","fp16","fp16")


class StrictBackend:
    def capabilities(self):
        return BackendCapabilities(hypic_seam8=True, exact_prefix_checkpoint=False, exact_selected_replay=True,
            causal_conv_seam_repair=True, full_attention_relocation=True, requires_artifact_qualification=True)
    def materialize(self,ids,identity):
        out=[]
        for sid in ids:
            r=RecurrentTailArtifact(0,0,8,AffineSummary(np.eye(2),np.zeros((2,2))))
            out.append(ExecutionArtifact(sid,identity,8,{(0,0):r}))
        return out
    def hypic_seam8(self,arts):
        return ApproximateExecution("hypic",RuntimeRiskSignals(),{})
    def single_state_init(self,arts): raise AssertionError
    def exact_prefix_replay(self,ids,checkpoint): raise AssertionError
    def exact_selected_replay(self,ids): return "exact"
    def compare_to_oracle(self,a,b): raise AssertionError


def test_strict_backend_refuses_unqualified_artifacts():
    s=SourceSegmentStore(":memory:")
    a=SourceSegment([1],"x",canonical_stream="c",canonical_start=0,canonical_end=1)
    d=SourceSegment([4],"x",canonical_stream="c",canonical_start=3,canonical_end=4)
    s.put(a); s.put(d)
    r=ReconstructionRuntime(s,ExecutionArtifactStore(),StrictBackend()).reconstruct([a.id,d.id],ident())
    assert r.action == ReconstructionAction.EXACT_SELECTED_REPLAY
    assert "qualification receipt" in r.reason


def test_complete_receipt_is_reusable():
    q=ArtifactQualification(True,True,True,True,0.001,1.0,"apple-m3","sha256:manifest","sha256:receipt")
    assert q.reusable_for_hypic
