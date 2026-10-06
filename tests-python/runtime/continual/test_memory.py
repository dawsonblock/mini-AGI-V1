from kvcontinual.continual.memory.store import MemoryStore, MemoryRecord
from kvcontinual.continual.memory.policy import MemoryWritePolicy, MemorySignals, MemoryDisposition


def test_temporal_supersession():
    s = MemoryStore(":memory:")
    old = MemoryRecord("Bob works at A", "fact", "test", "1", valid_from="2026-01-01T00:00:00Z")
    s.put(old)
    new = MemoryRecord("Bob works at B", "fact", "test", "2")
    s.supersede(old.id, new, "2026-06-01T00:00:00Z")
    assert s.get(old.id).valid_until == "2026-06-01T00:00:00Z"
    assert s.get(new.id).supersedes == old.id


def test_write_policy():
    p = MemoryWritePolicy()
    d = p.decide(MemorySignals(1, 1, 1, 1, 1, 0))
    assert d == MemoryDisposition.LEARNING_CANDIDATE
