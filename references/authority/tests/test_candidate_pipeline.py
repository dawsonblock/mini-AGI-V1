import json
from pathlib import Path
import pytest
from minagi.candidate_pipeline import CandidateLedger, QualificationPolicy, PromotionAuthority


def test_candidate_state_machine_and_signed_promotion(tmp_path):
    led = CandidateLedger(tmp_path / "authority")
    cid = led.propose(base_generation="g123", source="shadow", dataset_digest="abc")
    led.append(cid, "trained", evidence={"artifact": "candidate.bin"})
    led.append(cid, "evaluated", evidence={"suite": "functional-v41"})
    pol = QualificationPolicy(min_functional_accuracy=.8,
                              max_heldout_regression=.01,
                              require_checks=("compile", "cl"))
    rec = led.qualify(cid, {
        "functional_accuracy": .9,
        "heldout_regression": .0,
        "old_domain_regression": .01,
        "checks": {"compile": True, "cl": True},
    }, pol)
    assert rec["state"] == "qualified"
    art = tmp_path / "candidate.bin"; art.write_bytes(b"candidate")
    auth = PromotionAuthority(b"x" * 32, key_id="test")
    receipt = auth.issue(led, cid, artifact_path=str(art),
                         expected_base_generation="g123")
    assert led.state(cid) == "promoted"
    assert auth.verify_receipt(receipt, str(art))
    art.write_bytes(b"changed")
    assert not auth.verify_receipt(receipt, str(art))
    # A fresh reader verifies the whole chain and state machine.
    assert CandidateLedger(tmp_path / "authority").state(cid) == "promoted"


def test_candidate_rejects_bad_transition_and_bad_chain(tmp_path):
    led = CandidateLedger(tmp_path / "authority")
    cid = led.propose(base_generation="g", source="test")
    with pytest.raises(ValueError):
        led.append(cid, "qualified")
    led.append(cid, "rejected", evidence={"why": "bad"})
    lines = led.path.read_text().splitlines()
    d = json.loads(lines[-1]); d["actor"] = "tampered"
    lines[-1] = json.dumps(d)
    led.path.write_text("\n".join(lines) + "\n")
    with pytest.raises(RuntimeError):
        CandidateLedger(tmp_path / "authority")


def test_qualification_rejects_missing_required_evidence(tmp_path):
    led = CandidateLedger(tmp_path / "a")
    cid = led.propose(base_generation="g", source="growth")
    led.append(cid, "trained")
    led.append(cid, "evaluated")
    pol = QualificationPolicy(min_causal_gain=.01, require_checks=("functional",))
    r = led.qualify(cid, {"causal_gain": .001, "checks": {"functional": False}}, pol)
    assert r["state"] == "rejected"
    assert len(r["evidence"]["failures"]) == 2
