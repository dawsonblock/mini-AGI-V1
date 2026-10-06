from kvcontinual.execution.memory.store import MemoryStore, MemoryRecord
from kvcontinual.execution.memory.policy import MemoryWritePolicy, MemorySignals, MemoryDisposition


def test_temporal_supersession_and_source_links():
    s=MemoryStore(":memory:")
    old=MemoryRecord("Bob works at A","fact","test","1",valid_from="2026-01-01T00:00:00Z",source_segments=["seg1"])
    s.put(old)
    new=MemoryRecord("Bob works at B","fact","test","2",source_segments=["seg2"])
    s.supersede(old.id,new,"2026-06-01T00:00:00Z")
    assert s.get(old.id).valid_until == "2026-06-01T00:00:00Z"
    assert s.get(new.id).supersedes == old.id
    assert s.get(new.id).source_segments == ["seg2"]


def test_volatile_fact_stays_external_memory():
    p=MemoryWritePolicy()
    x=MemorySignals(1,1,1,1,1,0,volatility=1,procedurality=0,behavioral_failure=0,cache_invalidation_cost=1)
    assert p.decide(x) == MemoryDisposition.SEMANTIC


def test_repeated_stable_behavior_can_be_learning_candidate():
    p=MemoryWritePolicy()
    x=MemorySignals(.7,1,1,1,1,0,volatility=0,procedurality=1,behavioral_failure=1,cache_invalidation_cost=0)
    assert p.decide(x) == MemoryDisposition.LEARNING_CANDIDATE
