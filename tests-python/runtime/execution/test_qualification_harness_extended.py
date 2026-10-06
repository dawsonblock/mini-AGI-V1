from kvcontinual.execution.qualification_harness import (
    MacQualificationThresholds, QualificationObservation,
    artifact_qualification_from_summary, summarize_observations,
)


def test_qualification_receipt_requires_enough_cases_and_builds_artifact_receipt():
    obs=[QualificationObservation(str(i),0.001,1.0,True) for i in range(5)]
    s=summarize_observations(obs,MacQualificationThresholds(min_cases=5))
    assert s.passed and s.receipt_digest.startswith("sha256:")
    q=artifact_qualification_from_summary(s,same_hidden_input_verified=True,conv_boundary_complete=True,
        attention_relocation_complete=True,hardware_fingerprint="M3-Max",capture_manifest_digest="sha256:m")
    assert q.reusable_for_hypic
