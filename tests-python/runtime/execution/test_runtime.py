import numpy as np
from kvcontinual.execution.cache.block import ExecutionArtifact, RecurrentTailArtifact
from kvcontinual.execution.cache.store import ExecutionArtifactStore
from kvcontinual.execution.recurrent.affine import AffineSummary
from kvcontinual.execution.recurrent.interfaces import ApproximateExecution
from kvcontinual.execution.runtime import ReconstructionRuntime
from kvcontinual.execution.source import SourceSegment, SourceSegmentStore
from kvcontinual.execution.types import ExecutionIdentity, ModelIdentity, OracleMetrics, ReconstructionAction, RuntimeRiskSignals


def ident():
    return ExecutionIdentity(ModelIdentity("base","adapter","tok","arch","rope"),"abi","rec","layout","bf16","fp16")


class Backend:
    def __init__(self, risky=False): self.risky=risky
    def capabilities(self):
        from kvcontinual.execution.recurrent.interfaces import BackendCapabilities
        return BackendCapabilities(
            hypic_seam8=True, single_state_init=True, exact_prefix_checkpoint=True,
            exact_selected_replay=True, causal_conv_seam_repair=True,
            full_attention_relocation=True, platform="test", device="cpu"
        )
    def materialize(self, ids, identity):
        out=[]
        for sid in ids:
            r=RecurrentTailArtifact(0,0,8,AffineSummary(np.eye(2),np.zeros((2,2))))
            out.append(ExecutionArtifact(sid,identity,8,{(0,0):r}))
        return out
    def hypic_seam8(self, artifacts):
        x=.2 if self.risky else .01
        return ApproximateExecution({"kind":"hypic","ids":[a.source_segment_id for a in artifacts]}, RuntimeRiskSignals(seam_hidden_rel_l2=x,seam_attention_rel_l2=x,join_count=max(0,len(artifacts)-1)), {})
    def single_state_init(self, artifacts):
        return ApproximateExecution({"kind":"linear"}, RuntimeRiskSignals(seam_hidden_rel_l2=.01,seam_attention_rel_l2=.01), {})
    def exact_prefix_replay(self, ids, checkpoint): return {"kind":"prefix","ids":ids}
    def exact_selected_replay(self, ids): return {"kind":"exact","ids":ids}
    def compare_to_oracle(self, approximate, exact): return OracleMetrics(state_rel_l2=.001)


def build_sources():
    s=SourceSegmentStore(":memory:")
    a=SourceSegment([1],"x",canonical_stream="c",canonical_start=0,canonical_end=1)
    b=SourceSegment([2],"x",canonical_stream="c",canonical_start=1,canonical_end=2)
    d=SourceSegment([4],"x",canonical_stream="c",canonical_start=3,canonical_end=4)
    for x in (a,b,d): s.put(x)
    return s,a,b,d


def test_arbitrary_assembly_uses_hypic_seam8():
    s,a,b,d=build_sources(); rt=ReconstructionRuntime(s,ExecutionArtifactStore(),Backend())
    r=rt.reconstruct([a.id,d.id],ident())
    assert r.action == ReconstructionAction.HYPIC_SEAM8


def test_risky_arbitrary_assembly_falls_back_exact():
    s,a,b,d=build_sources(); rt=ReconstructionRuntime(s,ExecutionArtifactStore(),Backend(risky=True))
    r=rt.reconstruct([a.id,d.id],ident())
    assert r.action == ReconstructionAction.EXACT_SELECTED_REPLAY


def test_prefix_does_not_use_hypic():
    s,a,b,d=build_sources(); rt=ReconstructionRuntime(s,ExecutionArtifactStore(),Backend())
    r=rt.reconstruct([a.id,b.id],ident())
    assert r.action == ReconstructionAction.EXACT_PREFIX_REPLAY
