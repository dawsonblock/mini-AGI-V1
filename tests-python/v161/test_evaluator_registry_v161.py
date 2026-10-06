from minagi.v161.evaluator_registry import EvaluatorArtifact, EvaluatorRegistry
from minagi.v161.evaluators import exact_match, retention_score

def test_evaluator_registry_binds_implementation(tmp_path):
    reg=EvaluatorRegistry(tmp_path)
    a=EvaluatorArtifact.from_callable('exact',exact_match)
    assert reg.register(a)==a.digest
    assert reg.resolve(a.digest)('X','x')==1.0
    b=EvaluatorArtifact.from_callable('retention',retention_score)
    assert a.digest!=b.digest
