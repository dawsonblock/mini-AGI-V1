import numpy as np
from kvcontinual.execution.memory.retrieval import NumpyVectorIndex, VectorItem


def test_vector_retrieval_prefers_semantic_match():
    idx = NumpyVectorIndex()
    idx.add(VectorItem("a", np.array([1.,0.])))
    idx.add(VectorItem("b", np.array([0.,1.])))
    assert idx.search(np.array([1.,0.]), 1)[0][0] == "a"
