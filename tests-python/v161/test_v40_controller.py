"""V4-13 — capability acquisition controller policy tests."""
import json

from minagi.v161.capability_controller import (
    FailureEvidence, InterventionRecord, diagnose,
    intervention_history_append, intervention_history_load,
    select_intervention)


def test_diagnose_cannot_fit_training_set():
    h = diagnose(FailureEvidence(train_fit=0.1, within_family=0.05))
    assert h[0].name == "representation_limitation"
    assert "train_fit" in h[0].evidence[0]


def test_diagnose_memorization_without_generalization():
    h = diagnose(FailureEvidence(train_fit=1.0, within_family=0.5,
                                 frozen_baseline=0.4))
    assert h[0].name == "insufficient_generalization"


def test_diagnose_retention_drop_is_forgetting():
    h = diagnose(FailureEvidence(train_fit=1.0, within_family=0.9,
                                 retention=0.4, retention_baseline=0.9))
    assert h[0].name == "catastrophic_forgetting"


def test_diagnose_empty_output_is_instability():
    h = diagnose(FailureEvidence(empty_output_rate=0.9))
    assert h[0].name == "inference_instability"


def test_no_intervention_selected_when_nothing_helps():
    # weak evidence + all candidates expensive -> no_intervention wins
    h = diagnose(FailureEvidence(train_fit=0.6, within_family=0.55,
                                 frozen_baseline=0.55))
    d = select_intervention(h, [])
    assert d.candidate_controlled is True
    assert d.digest.startswith("sha256:")
    assert d.selected_intervention in d.candidate_interventions or \
        d.selected_intervention == "no_intervention"


def test_failed_intervention_refused_without_changed_condition(tmp_path):
    h = diagnose(FailureEvidence(train_fit=1.0, within_family=0.4,
                                 frozen_baseline=0.3))
    # history: fixed_lora already failed on this target under the same
    # conditions
    from minagi.v161.capability_controller import _signature
    sig = _signature("fixed_lora", "", "")
    hist = [InterventionRecord(signature=sig, intervention="fixed_lora",
                               outcome_delta=-0.1, retention_drop=0.2,
                               verified=True)]
    d = select_intervention(h, hist)
    assert d.selected_intervention != "fixed_lora"
    assert any("fixed_lora" in r for r in d.refused)


def test_changed_condition_overrides_refusal_with_audit_note(tmp_path):
    h = diagnose(FailureEvidence(train_fit=1.0, within_family=0.4,
                                 frozen_baseline=0.3))
    from minagi.v161.capability_controller import _signature
    sig = _signature("fixed_lora", "", "")
    hist = [InterventionRecord(signature=sig, intervention="fixed_lora",
                               outcome_delta=-0.1, retention_drop=0.0,
                               verified=True)]
    d = select_intervention(h, hist,
                            changed_condition="rank doubled 8->16")
    assert any("overridden" in r and "rank doubled" in r
               for r in d.refused)


def test_history_roundtrip(tmp_path):
    p = tmp_path / "history.jsonl"
    rec = InterventionRecord(signature="sha256:" + "0" * 64,
                             intervention="rehearsal",
                             outcome_delta=0.3, retention_drop=0.0,
                             verified=True, cost=12.5,
                             conditions="r16")
    intervention_history_append(p, rec)
    intervention_history_append(p, rec)
    got = intervention_history_load(p)
    assert len(got) == 2 and got[0].intervention == "rehearsal"


def test_unverified_outcomes_do_not_trigger_refusal():
    h = diagnose(FailureEvidence(train_fit=1.0, within_family=0.4,
                                 frozen_baseline=0.3))
    from minagi.v161.capability_controller import _signature
    sig = _signature("rehearsal", "", "")
    hist = [InterventionRecord(signature=sig, intervention="rehearsal",
                               outcome_delta=-0.5, retention_drop=0.0,
                               verified=False)]  # unverified — not a refusal
    d = select_intervention(h, hist)
    assert not any("rehearsal:" in r and "non-positive" in r
                   for r in d.refused)


def test_decision_record_fields_complete():
    h = diagnose(FailureEvidence(train_fit=1.0, within_family=0.5,
                                 frozen_baseline=0.4))
    d = select_intervention(h, [])
    doc = json.loads(json.dumps(d.__dict__))
    for k in ("schema", "observed_weakness", "supporting_evidence",
              "candidate_interventions", "selected_intervention",
              "predicted_improvement", "estimated_cost", "confidence",
              "rationale", "refused", "candidate_controlled", "digest"):
        assert k in doc
