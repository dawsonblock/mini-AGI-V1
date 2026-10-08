"""v16.5 — evidence receipt v3 + authority separation + atomic commit.

Every test in this file would pass on the pre-v165 implementation and
must fail against the repaired one. The suite simulates a compromised
worker: anything the worker can touch (predictions, reported metrics,
receipt bodies, keys it holds, directories it writes) is mutated and
must be rejected.
"""
import json
from dataclasses import asdict

import pytest

from egai.common.canonical import digest, sha256_bytes, validate_digest
from egai.common.crypto import Ed25519Signer, Ed25519Verifier
from minagi.v161.authority import (AuthorityLedger, AuthorityRegistry,
                                   AuthorityRecord, provision_role,
                                   write_trust_root, AUTHORITY_ROLES)
from minagi.v161.evidence_receipt_v3 import (
    EvidenceReceiptV3, evaluation_bundle, input_manifest_digest,
    load_verified_seed_result_v3, prediction_record,
    predictions_canonical_bytes, predictions_digest_of,
    read_predictions, write_predictions, RECEIPT_V3_SCHEMA)

D = digest({"anchor": 1})
Z = "sha256:" + "0" * 64


def _preds():
    return [
        prediction_record("t1", "hidden", "in one", "out one", 3, 32),
        prediction_record("t2", "hidden", "in two", "out two", 4, 32),
        prediction_record("r1", "retention", "in r", "out r", 2, 32),
        prediction_record("s1", "security", "in s", "out s", 2, 32),
    ]


def _cell(signer, seed=0, arm="L1", campaign=D, protocol=D,
          preds=None, metrics=None, cell="L1"):
    """Build a fully consistent signed v3 cell; returns mutable parts."""
    preds = _preds() if preds is None else preds
    metrics = {"hidden_exact_match": 0.5, "retention": 0.9,
               "security_regressions": 0} if metrics is None else metrics
    pd = predictions_digest_of(preds)
    im = input_manifest_digest(preds)
    bundle = evaluation_bundle(campaign, protocol, seed, arm, cell,
                               im, pd, metrics)
    r = EvidenceReceiptV3.sign(
        signer=signer, campaign_digest=campaign, protocol_digest=protocol,
        seed=seed, arm=arm, model_digest=D, tokenizer_digest=D,
        adapter_digest=Z, state_digest=Z, environment_digest=D,
        evaluation_dataset_digest=D, evaluator_digest=D,
        input_manifest_digest=im, predictions_digest=pd,
        evaluation_bundle_digest=digest(bundle))
    return {"receipt": r, "bundle": bundle, "preds": preds}


def _verifier(signer):
    v = Ed25519Verifier()
    v.register(signer.key_id, signer.public_bytes())
    return v


def _write_cell(run_dir, name, cell):
    (run_dir / f"{name}.json").write_text(json.dumps(
        {"receipt": asdict(cell["receipt"]), "bundle": cell["bundle"]},
        indent=2, sort_keys=True))
    write_predictions(cell["preds"], run_dir / f"PREDICTIONS-{name}.jsonl")


def _seed_dir(tmp_path, signer, seed=7, campaign=D):
    """A fully committed v3 seed directory (staging + atomic publish)."""
    run_dir = tmp_path / f"seed-{seed}.staging"
    final = tmp_path / f"seed-{seed}"
    run_dir.mkdir()
    cells = {a: _cell(signer, seed=seed, arm=a, campaign=campaign,
                      cell=a)
             for a in ("L1", "L5", "L6")}
    receipt_digests = {}
    for name, c in cells.items():
        _write_cell(run_dir, name, c)
        receipt_digests[name] = c["receipt"].digest
    sd = {"seed": seed, "hidden_exact_match": {"L1": 0.5, "L5": 0.5,
                                               "L6": 0.6},
          "receipt_digests": receipt_digests, "attempt_id": "a1"}
    (run_dir / "SEED_RESULT.json").write_text(json.dumps(sd))
    (run_dir / "COMPLETE").write_text("complete\n")
    man = {"seed": seed, "campaign_digest": campaign, "attempt_id": "a1",
           "files": {p.relative_to(run_dir).as_posix():
                     sha256_bytes(p.read_bytes())
                     for p in sorted(run_dir.rglob("*")) if p.is_file()}}
    (run_dir / "COMMIT_MANIFEST.json").write_text(json.dumps(man))
    run_dir.rename(final)
    return final, sd


# ---------------------------------------------------------------- prediction
# canonical records

def test_prediction_record_is_canonical_and_digested():
    p = prediction_record("t", "hidden", "in", "out", 2, 32)
    assert p["input_sha256"] == sha256_bytes(b"in")
    assert p["decoding"] == {"mode": "greedy", "max_new_tokens": 32}
    b = predictions_canonical_bytes([p])
    assert b == predictions_canonical_bytes(
        [json.loads(b.decode().strip())])


def test_prediction_record_rejects_bad_inputs():
    with pytest.raises(ValueError):
        prediction_record("t", "bogus-block", "i", "o", 1, 8)
    with pytest.raises(ValueError):
        prediction_record("t", "hidden", "i", "o", -1, 8)
    with pytest.raises(ValueError):
        prediction_record("t", "hidden", "i", "o", 1, 0)
    with pytest.raises(ValueError):
        prediction_record("", "hidden", "i", "o", 1, 8)


def test_canonical_encoding_rejects_nan_and_infinity():
    p = prediction_record("t", "hidden", "i", "o", 1, 8)
    p["generated_tokens"] = float("nan")
    with pytest.raises(ValueError):
        predictions_canonical_bytes([p])
    p["generated_tokens"] = float("inf")
    with pytest.raises(ValueError):
        predictions_canonical_bytes([p])


def test_receipt_requires_signed_payload_and_valid_digests():
    with pytest.raises(ValueError):
        EvidenceReceiptV3(D, D, 0, "L1", D, D, Z, Z, D, D, D, D, D, D,
                          "", "c2ln")
    with pytest.raises(ValueError):
        EvidenceReceiptV3("not-a-digest", D, 0, "L1", D, D, Z, Z, D, D,
                          D, D, D, D, "k", "c2ln")


# ------------------------------------------------------------------- receipt
# signature mutations

def test_receipt_verifies_and_detects_field_tamper():
    s = Ed25519Signer.generate()
    c = _cell(s)
    assert c["receipt"].verify(_verifier(s))
    # every receipt field participates in the signature — swap one
    for field, bad in (("seed", 9), ("arm", "L6"),
                       ("model_digest", Z), ("predictions_digest", Z),
                       ("input_manifest_digest", Z),
                       ("evaluation_bundle_digest", Z),
                       ("protocol_digest", Z),
                       ("campaign_digest", Z),
                       ("evaluator_digest", Z),
                       ("evaluation_dataset_digest", Z)):
        forged = EvidenceReceiptV3(**{**asdict(c["receipt"]), field: bad})
        assert not forged.verify(_verifier(s)), field


def test_receipt_from_other_seed_or_campaign_rejected():
    s = Ed25519Signer.generate()
    good = _cell(s, seed=0)
    other_seed = _cell(s, seed=1)
    assert not EvidenceReceiptV3(
        **{**asdict(other_seed["receipt"]),
           "seed": 0}).verify(_verifier(s))
    other_campaign = _cell(s, campaign=digest({"other": "campaign"}))
    assert good["receipt"].campaign_digest != \
        other_campaign["receipt"].campaign_digest


def test_wrong_signer_rejected():
    s1, s2 = Ed25519Signer.generate(), Ed25519Signer.generate()
    c = _cell(s1)
    assert not c["receipt"].verify(_verifier(s2))


# -------------------------------------------------------------- predictions
# artifact mutations (the load-time checks in load_verified_seed_result_v3
# mirror exactly what the qualifier recomputes)

def test_single_prediction_change_breaks_digest():
    preds = _preds()
    d0 = predictions_digest_of(preds)
    preds[0] = dict(preds[0], output="out CHANGED")
    assert predictions_digest_of(preds) != d0


def test_task_id_change_breaks_digests():
    preds = _preds()
    assert predictions_digest_of(
        [dict(p, id="OTHER") for p in preds]) != \
        predictions_digest_of(preds)
    assert input_manifest_digest(
        [dict(p, id="OTHER") for p in preds]) != \
        input_manifest_digest(preds)


def test_reorder_duplicate_remove_detected():
    preds = _preds()
    d0, m0 = predictions_digest_of(preds), input_manifest_digest(preds)
    assert predictions_digest_of(list(reversed(preds))) != d0
    assert predictions_digest_of(preds + [preds[0]]) != d0
    assert predictions_digest_of(preds[1:]) != d0
    assert input_manifest_digest(list(reversed(preds))) != m0
    assert input_manifest_digest(preds[1:]) != m0


def test_metric_only_edit_breaks_bundle_digest():
    s = Ed25519Signer.generate()
    c = _cell(s)
    forged = dict(c["bundle"], metrics_reported={"hidden_exact_match": 1.0})
    assert digest(forged) != c["receipt"].evaluation_bundle_digest


# ------------------------------------------------------------- atomic commit

def test_verified_seed_result_round_trip(tmp_path):
    s = Ed25519Signer.generate()
    final, sd = _seed_dir(tmp_path, s)
    got = load_verified_seed_result_v3(
        final, 7, D, ("L1", "L5", "L6"), _verifier(s))
    assert got == sd


def test_torn_seed_dir_rejected(tmp_path):
    """Crash-sim: staging never publishes -> no evidence is adopted."""
    s = Ed25519Signer.generate()
    final, _ = _seed_dir(tmp_path, s)
    for victim in ("COMPLETE", "SEED_RESULT.json", "COMMIT_MANIFEST.json",
                   "L1.json", "PREDICTIONS-L1.jsonl"):
        p = final / victim
        p.rename(final / f"{victim}.bak")
        assert load_verified_seed_result_v3(
            final, 7, D, ("L1", "L5", "L6"), _verifier(s)) is None, victim
        (final / f"{victim}.bak").rename(p)


def test_mutated_committed_file_rejected(tmp_path):
    s = Ed25519Signer.generate()
    final, _ = _seed_dir(tmp_path, s)
    p = final / "PREDICTIONS-L6.jsonl"
    recs = read_predictions(p)
    recs[1]["output"] = "fabricated"
    p.write_text("\n".join(json.dumps(r) for r in recs) + "\n")
    assert load_verified_seed_result_v3(
        final, 7, D, ("L1", "L5", "L6"), _verifier(s)) is None


def test_replaced_receipt_from_other_seed_rejected(tmp_path):
    s = Ed25519Signer.generate()
    final, _ = _seed_dir(tmp_path, s, seed=7)
    other, _ = _seed_dir(tmp_path, s, seed=8)
    other.mkdir(exist_ok=True)
    other_final = other.parent / "seed-8"
    # graft seed-8's L6 cell into seed-7 and rebuild the manifest
    (final / "L6.json").write_text((other_final / "L6.json").read_text())
    man = json.loads((final / "COMMIT_MANIFEST.json").read_text())
    man["files"]["L6.json"] = sha256_bytes(
        (final / "L6.json").read_bytes())
    (final / "COMMIT_MANIFEST.json").write_text(json.dumps(man))
    assert load_verified_seed_result_v3(
        final, 7, D, ("L1", "L5", "L6"), _verifier(s)) is None


def test_wrong_witness_key_rejected(tmp_path):
    s, attacker = Ed25519Signer.generate(), Ed25519Signer.generate()
    final, _ = _seed_dir(tmp_path, s)
    assert load_verified_seed_result_v3(
        final, 7, D, ("L1", "L5", "L6"), _verifier(attacker)) is None


def test_missing_planned_arm_rejected(tmp_path):
    s = Ed25519Signer.generate()
    final, _ = _seed_dir(tmp_path, s)
    (final / "L5.json").unlink()
    (final / "PREDICTIONS-L5.jsonl").unlink()
    man = json.loads((final / "COMMIT_MANIFEST.json").read_text())
    del man["files"]["L5.json"], man["files"]["PREDICTIONS-L5.jsonl"]
    (final / "COMMIT_MANIFEST.json").write_text(json.dumps(man))
    assert load_verified_seed_result_v3(
        final, 7, D, ("L1", "L5", "L6"), _verifier(s)) is None


# -------------------------------------------------------------------- roles

def _provisioned(tmp_path):
    keys = tmp_path / ".keys"
    registry = write_trust_root(keys, tmp_path / "trust_root.json")
    signers = {r: Ed25519Signer.from_private_bytes(
        (keys / f"{r}.pem").read_bytes()) for r in AUTHORITY_ROLES}
    return registry, signers


def test_role_registry_separates_signers(tmp_path):
    registry, signers = _provisioned(tmp_path)
    for role, s in signers.items():
        assert registry.is_authorized(role, s.key_id)
        for other in AUTHORITY_ROLES:
            if other != role:
                assert not registry.is_authorized(other, s.key_id)


def test_unregistered_and_revoked_keys_rejected(tmp_path):
    registry, _ = _provisioned(tmp_path)
    rogue = Ed25519Signer.generate()
    assert not registry.is_authorized("plan", rogue.key_id)
    with pytest.raises(PermissionError):
        registry.assert_authorized("plan", rogue.key_id)
    rec = next(iter(registry._by_key.values()))
    revoked = AuthorityRecord(rec.role, rec.key_id, rec.public_key_b64,
                              status="revoked")
    reg2 = AuthorityRegistry([revoked])
    assert not reg2.is_authorized(rec.role, rec.key_id)


def test_same_key_collapse_rejected(tmp_path):
    s = Ed25519Signer.generate()
    pub = s.public_bytes()
    import base64
    b = base64.b64encode(pub).decode()
    with pytest.raises(ValueError):
        AuthorityRegistry([
            AuthorityRecord("plan", s.key_id, b),
            AuthorityRecord("promotion", s.key_id, b)])


def test_reloaded_role_key_keeps_fingerprint_identity(tmp_path):
    keys = tmp_path / ".keys"
    s1 = provision_role(keys, "plan")
    s2 = provision_role(keys, "plan")
    assert s1.key_id == s2.key_id


# ------------------------------------------------------------------- ledger

def test_ledger_chain_detects_tamper_and_wrong_role(tmp_path):
    registry, signers = _provisioned(tmp_path)
    led = AuthorityLedger(tmp_path / "AUTHORITY_LEDGER.jsonl")
    led.append(signers["plan"], "experiment_preregistration",
               {"plan": "x"})
    led.append(signers["qualification"], "qualification_record",
               {"decision": "QUALIFIED"})
    assert led.verify(registry) == []
    # wrong role cannot append a kind it does not own — verifier flags it
    led.append(signers["execution_witness"], "qualification_record",
               {"decision": "QUALIFIED"})
    fails = led.verify(registry)
    assert any("not authorized" in f for f in fails)


def test_ledger_requires_preregistration_first(tmp_path):
    registry, signers = _provisioned(tmp_path)
    led = AuthorityLedger(tmp_path / "AUTHORITY_LEDGER.jsonl")
    led.append(signers["qualification"], "qualification_record", {"d": 1})
    assert any("first record" in f for f in led.verify(registry))


def test_ledger_reorder_and_field_tamper_detected(tmp_path):
    registry, signers = _provisioned(tmp_path)
    p = tmp_path / "L.jsonl"
    led = AuthorityLedger(p)
    led.append(signers["plan"], "experiment_preregistration", {"a": 1})
    led.append(signers["evaluation"], "evaluation_bundle", {"b": 2})
    led.append(signers["qualification"], "qualification_record", {"c": 3})
    lines = p.read_text().splitlines()
    # swap records 1 and 2: the prev_record_digest chain must catch it
    p.write_text("\n".join([lines[0], lines[2], lines[1]]) + "\n")
    fails = led.verify(registry)
    assert any("prev_record_digest" in f or "seq" in f for f in fails)
    # editing a signed field must break record_digest + signature
    p.write_text("\n".join(lines) + "\n")
    rec = json.loads(lines[1])
    rec["kind"] = "promotion_decision"
    lines[1] = json.dumps(rec, sort_keys=True)
    p.write_text("\n".join(lines) + "\n")
    assert led.verify(registry)


def test_trust_root_roundtrip(tmp_path):
    registry, signers = _provisioned(tmp_path)
    loaded = AuthorityRegistry.load(tmp_path / "trust_root.json")
    for s in signers.values():
        assert loaded.is_authorized(
            next(r for r in AUTHORITY_ROLES
                 if registry.is_authorized(r, s.key_id)), s.key_id)
