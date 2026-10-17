"""V4-28 — activation-time enforcement.

Registration admissibility is not enough: the binding

    qualification evidence -> adapter digest -> model revision
        -> registry authorization -> activation

is re-verified at the moment the adapter would serve traffic. These
tests pin the failure modes the plan calls out: modified weights,
missing model identity, and forged family hints.
"""
from egai.common.canonical import digest
from egai.common.crypto import Ed25519Signer, Ed25519Verifier
from minagi.v161.runtime_closure3 import sha256_path
from minagi.v161.verified_routing import (QualifiedAdapterManifest,
                                          VerifiedAdapterRouter)

Z = "sha256:" + "0" * 64


def _manifest(aid="a1", families=("tr-reverse",), terms=("reverse",),
              **kw):
    return QualifiedAdapterManifest(
        adapter_id=aid,
        adapter_digest=kw.pop("adapter_digest", digest({"a": aid})),
        model_id=kw.pop("model_id", "m"),
        model_revision=kw.pop("model_revision", "rev1"),
        qualified_families=families,
        qualification_evidence_digest=kw.pop("evidence", digest({"e": aid})),
        scorer_artifact_digest=kw.pop("scorer", Z),
        family_terms=terms,
        **kw)


def test_weights_modified_between_register_and_route_are_rejected(tmp_path):
    """The adapter's bytes are re-checked at activation — a manifest
    that passed registration cannot serve tampered weights."""
    adir = tmp_path / "adapter-a1"
    adir.mkdir()
    (adir / "w.bin").write_bytes(b"weights-v1")
    r = VerifiedAdapterRouter(
        [_manifest(adapter_digest=sha256_path(adir))],
        artifact_resolver=lambda aid: tmp_path / f"adapter-{aid}")
    # tamper AFTER registration, BEFORE routing
    (adir / "w.bin").write_bytes(b"weights-TAMPERED")
    d = r.route("reverse this", family="tr-reverse",
                family_authenticated=True, model_revision="rev1")
    assert d.selected is None
    assert any("digest mismatch at activation" in x for x in d.refused)
    assert "a1" in r.ineligible()
    assert any(e["kind"] == "activation_recheck_failed"
               for e in r.events)


def test_activation_recheck_falls_through_to_next_adapter(tmp_path):
    """A recheck failure refuses THAT adapter — it does not crash the
    route; the next qualified candidate can still serve."""
    a1 = tmp_path / "adapter-a1"
    a2 = tmp_path / "adapter-a2"
    for d in (a1, a2):
        d.mkdir()
    (a1 / "w.bin").write_bytes(b"w1")
    (a2 / "w.bin").write_bytes(b"w2")
    m1 = _manifest("a1", adapter_digest=sha256_path(a1), priority=0.9)
    m2 = _manifest("a2", adapter_digest=sha256_path(a2), priority=0.1)
    r = VerifiedAdapterRouter(
        [m1, m2],
        artifact_resolver=lambda aid: tmp_path / f"adapter-{aid}")
    (a1 / "w.bin").write_bytes(b"TAMPERED")   # top candidate is dirty
    d = r.route("reverse this", family="tr-reverse",
                family_authenticated=True, model_revision="rev1")
    assert d.selected == "a2"
    assert "a1" in r.ineligible()


def test_forged_family_hint_cannot_force_activation():
    """An attacker-controlled family label narrows the candidate set
    but cannot bypass the lexical/OOD evidence gate."""
    r = VerifiedAdapterRouter([_manifest()])
    # query shares NO terms with the adapter's family vocabulary —
    # an unauthenticated hint claiming tr-reverse must not activate
    d = r.route("photosynthesis essay question", family="tr-reverse",
                family_authenticated=False, model_revision="rev1")
    assert d.selected is None
    assert any("family not qualified" in x for x in d.refused)
    # the same hint marked as authenticated dispatch metadata DOES
    # route — that path is for trusted internal dispatchers only
    d2 = r.route("photosynthesis essay question", family="tr-reverse",
                 family_authenticated=True, model_revision="rev1")
    assert d2.selected == "a1"
    # and the routing event records which kind of hint it was
    assert r.events[-2]["family_hint_authenticated"] is False
    assert r.events[-1]["family_hint_authenticated"] is True


def test_missing_model_identity_fails_closed():
    r = VerifiedAdapterRouter([_manifest()])
    d = r.route("reverse this", family="tr-reverse",
                family_authenticated=True)   # no model_revision
    assert d.selected is None
    assert any("model revision identity required" in x
               for x in d.refused)


def test_manifest_without_model_revision_is_ineligible_at_route():
    r = VerifiedAdapterRouter([_manifest(model_revision="")])
    d = r.route("reverse this", family="tr-reverse",
                family_authenticated=True, model_revision="rev1")
    assert d.selected is None
    assert any("no model revision" in x for x in d.refused)


def test_model_identity_can_be_disabled_only_explicitly():
    """Research isolation may drop the binding — but only by opting
    out, never by omission."""
    r = VerifiedAdapterRouter([_manifest()],
                              require_model_identity=False)
    d = r.route("reverse this", family="tr-reverse",
                family_authenticated=True)
    assert d.selected == "a1"
    assert r.events[-1]["controls"]["model_identity_required"] is False


def test_revoked_after_registration_never_activates(tmp_path):
    """Revocation + activation recheck compose: a revoked adapter is
    refused before bytes are even read."""
    adir = tmp_path / "adapter-a1"
    adir.mkdir()
    (adir / "w.bin").write_bytes(b"w")
    r = VerifiedAdapterRouter(
        [_manifest(adapter_digest=sha256_path(adir))],
        artifact_resolver=lambda aid: tmp_path / f"adapter-{aid}")
    r.revoke("a1")
    d = r.route("reverse this", family="tr-reverse",
                family_authenticated=True, model_revision="rev1")
    assert d.selected is None
    assert any("revoked" in x for x in d.refused)


# ---- V4-31 — outcome receipts: a flag is not verification -------------

def test_capability_outcome_receipt_verification(tmp_path):
    """--verified requires evidence that checks out on its own.

    v4.0.3: a hash-consistent self-authored run directory is
    INTEGRITY_VERIFIED at best — local file consistency is not an
    independent evaluation, so it cannot enter verified history."""
    from minagi.v161.capability_outcomes import verify_outcome_receipt

    # a self-authored committed run: COMPLETE + matching inventory +
    # METRICS.json proves integrity only — NOT a verified outcome
    run = tmp_path / "run1"
    run.mkdir()
    (run / "METRICS.json").write_text("{}")
    commit = {"run_id": "run1", "status": "COMPLETE",
              "files": {"METRICS.json":
                        __import__("hashlib").sha256(b"{}").hexdigest()}}
    (run / "RUN_COMMIT.json").write_text(__import__("json").dumps(commit))
    res = verify_outcome_receipt(run)
    assert res["kind"] == "committed_run"
    assert not res["verified"]              # integrity only — refused
    assert res["verification_state"] == "INTEGRITY_VERIFIED"

    # a FAILED run is not verification of success
    commit["status"] = "FAILED"
    (run / "RUN_COMMIT.json").write_text(__import__("json").dumps(commit))
    assert not verify_outcome_receipt(run)["verified"]

    # a tampered inventory entry fails
    commit["status"] = "COMPLETE"
    (run / "RUN_COMMIT.json").write_text(__import__("json").dumps(commit))
    (run / "METRICS.json").write_text("{\"x\": 1}")
    assert not verify_outcome_receipt(run)["verified"]

    # a signed receipt document requires a verifier
    s = Ed25519Signer.generate()
    rp = tmp_path / "receipt.json"
    unsigned = {"schema": "mini-agi-v16.1-executed-run-receipt-v1",
                "run_id": "run1"}
    rp.write_text(__import__("json").dumps(unsigned))
    # an unsigned receipt is not verification either
    assert not verify_outcome_receipt(rp)["verified"]
    # a correctly verified signature passes — note signer_key_id is
    # part of the signed body (the verifier checks every field except
    # signature_b64), so it is present before signing
    v = Ed25519Verifier()
    v.register(s.key_id, s.public_bytes())
    good = {"schema": "mini-agi-v16.1-executed-run-receipt-v1",
            "run_id": "run1", "signer_key_id": s.key_id}
    good["signature_b64"] = s.sign(good).signature_b64
    rp.write_text(__import__("json").dumps(good))
    # without a verifier a signed receipt is NOT verification —
    # a signature that cannot be checked proves nothing
    assert not verify_outcome_receipt(rp)["verified"]
    # v4.0.3-r1: a verifier alone is still not enough — the signer's
    # AUTHORITY ROLE must be established. A signature from a key whose
    # role cannot be checked is not independent evaluation evidence.
    assert not verify_outcome_receipt(rp, v)["verified"]
    # a valid signature from an UNAUTHORIZED role is not evaluation
    # evidence either — roles are separated
    import base64
    from minagi.v161.authority import AuthorityRecord, AuthorityRegistry
    plan_only = AuthorityRegistry([AuthorityRecord(
        role="plan", key_id=s.key_id,
        public_key_b64=base64.b64encode(s.public_bytes()).decode())])
    assert not verify_outcome_receipt(rp, v, authority=plan_only)[
        "verified"]
    # signature + evaluation-role authorization verifies
    eval_reg = AuthorityRegistry([AuthorityRecord(
        role="evaluation", key_id=s.key_id,
        public_key_b64=base64.b64encode(s.public_bytes()).decode())])
    res = verify_outcome_receipt(rp, v, authority=eval_reg)
    assert res["verified"] and res["signer_key_id"] == s.key_id
    # and a forged signature fails even under a trust root
    bad = dict(good)
    bad["signature_b64"] = Ed25519Signer.generate().sign(good).signature_b64
    rp.write_text(__import__("json").dumps(bad))
    assert not verify_outcome_receipt(rp, v, authority=eval_reg)[
        "verified"]
