from kvcontinual.continual.learning import CandidateDatasetBuilder, LearningExample
from kvcontinual.continual.replay import ReplayItem, ReplayStore


def test_dataset_builder_uses_verified_only():
    r = ReplayStore()
    r.add(ReplayItem({"x": 1}))
    b = CandidateDatasetBuilder(r, new_fraction=.5)
    ds = b.build([
        LearningExample("p1","t1","e1", verified=True),
        LearningExample("p2","t2","e2", verified=False),
    ])
    assert len(ds.new_examples) == 1
    assert ds.size >= 1
