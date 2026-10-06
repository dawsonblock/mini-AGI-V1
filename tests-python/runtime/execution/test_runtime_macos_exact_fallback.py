from kvcontinual.execution.cache.store import ExecutionArtifactStore
from kvcontinual.execution.recurrent.interfaces import BackendCapabilities
from kvcontinual.execution.runtime import ReconstructionRuntime
from kvcontinual.execution.source import SourceSegment, SourceSegmentStore
from kvcontinual.execution.types import ExecutionIdentity, ModelIdentity, ReconstructionAction


def ident():
    return ExecutionIdentity(ModelIdentity("base","adapter","tok","arch","rope"),"mac-abi","pytorch-ref","n/a","fp16","none")


class ExactOnlyBackend:
    def capabilities(self):
        return BackendCapabilities(hypic_seam8=False, single_state_init=False, exact_prefix_checkpoint=False, exact_selected_replay=True, platform="macos", device="mps")
    def exact_selected_replay(self, ids): return {"kind":"exact","ids":ids}
    def exact_prefix_replay(self, ids, checkpoint): raise AssertionError("should not use checkpoint path")
    def materialize(self, ids, identity): raise AssertionError("should not materialize HYPIC")
    def hypic_seam8(self, artifacts): raise AssertionError("should not call HYPIC")
    def single_state_init(self, artifacts): raise AssertionError("should not call LinearKV")
    def compare_to_oracle(self, approximate, exact): raise AssertionError


def test_arbitrary_topology_exact_fallback_when_hypic_unavailable():
    store=SourceSegmentStore(":memory:")
    a=SourceSegment([1],"x",canonical_stream="c",canonical_start=0,canonical_end=1)
    d=SourceSegment([4],"x",canonical_stream="c",canonical_start=3,canonical_end=4)
    store.put(a); store.put(d)
    rt=ReconstructionRuntime(store,ExecutionArtifactStore(),ExactOnlyBackend())
    r=rt.reconstruct([a.id,d.id],ident())
    assert r.action == ReconstructionAction.EXACT_SELECTED_REPLAY
    assert "does not provide qualified HYPIC" in r.reason


def test_prefix_exact_fallback_when_checkpoint_restore_unavailable():
    store=SourceSegmentStore(":memory:")
    a=SourceSegment([1],"x",canonical_stream="c",canonical_start=0,canonical_end=1)
    b=SourceSegment([2],"x",canonical_stream="c",canonical_start=1,canonical_end=2)
    store.put(a); store.put(b)
    rt=ReconstructionRuntime(store,ExecutionArtifactStore(),ExactOnlyBackend())
    r=rt.reconstruct([a.id,b.id],ident())
    assert r.action == ReconstructionAction.EXACT_SELECTED_REPLAY
    assert "prefix-checkpoint" in r.reason
