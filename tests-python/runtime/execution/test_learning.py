import pytest
from kvcontinual.execution.learning import CandidateDatasetBuilder, LearningExample
from kvcontinual.execution.replay import ReplayItem, ReplayStore


def test_dataset_builder_requires_verified_source_backing():
    r=ReplayStore(); r.add(ReplayItem({"x":1})); b=CandidateDatasetBuilder(r,new_fraction=.5)
    ds=b.build([LearningExample("p","t","e",source_segment_ids=["s"],verified=True), LearningExample("bad","bad","e2",verified=False)])
    assert len(ds.new_examples)==1
    with pytest.raises(ValueError): b.build([LearningExample("p","t","e",verified=True)])
