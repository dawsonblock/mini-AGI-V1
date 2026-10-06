from kvcontinual.execution.source import SourceSegment, SourceSegmentStore
from kvcontinual.execution.types import AssemblyTopology


def test_source_segments_are_durable_and_topology_detected():
    s=SourceSegmentStore(":memory:")
    a=SourceSegment([1,2],"doc",canonical_stream="c",canonical_start=0,canonical_end=2)
    b=SourceSegment([3,4],"doc",canonical_stream="c",canonical_start=2,canonical_end=4)
    d=SourceSegment([7,8],"doc",canonical_stream="c",canonical_start=6,canonical_end=8)
    for x in (a,b,d): s.put(x)
    assert s.get(a.id).tokens == [1,2]
    assert s.classify_topology([a.id,b.id]) == AssemblyTopology.EXACT_PREFIX
    assert s.classify_topology([a.id,d.id]) == AssemblyTopology.ARBITRARY
