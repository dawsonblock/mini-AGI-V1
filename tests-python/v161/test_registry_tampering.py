"""V4-28 — authenticated registry state: tamper/staleness fail closed.

The registry holds mutable safety state (revocations, suspensions,
health). An adapter manifest signature protects one manifest — it does
nothing for the revocation list. These tests pin: a deleted revocation
cannot restore access, a stale snapshot cannot roll safety state back,
and unsigned state is never production-admissible under strict_state.
"""
import json
from dataclasses import asdict

import pytest

from egai.common.canonical import digest
from egai.common.crypto import Ed25519Signer, Ed25519Verifier
from minagi.v161.authenticated_registry import (RevocationLog, open_state,
                                                seal_state)
from minagi.v161.verified_routing import (QualifiedAdapterManifest,
                                          VerifiedAdapterRouter)

Z = "sha256:" + "0" * 64


def _manifest(aid="a1", families=("tr-reverse",), **kw):
    return QualifiedAdapterManifest(
        adapter_id=aid,
        adapter_digest=kw.pop("adapter_digest", digest({"a": aid})),
        model_id=kw.pop("model_id", "m"),
        model_revision=kw.pop("model_revision", "rev1"),
        qualified_families=families,
        qualification_evidence_digest=kw.pop("evidence", digest({"e": aid})),
        scorer_artifact_digest=kw.pop("scorer", Z),
        family_terms=tuple(t.replace("tr-", "") for t in families),
        **kw)


def _sv():
    s = Ed25519Signer.generate()
    v = Ed25519Verifier()
    v.register(s.key_id, s.public_bytes())
    return s, v


def test_signed_state_roundtrip_preserves_revocation(tmp_path):
    s, v = _sv()
    r = VerifiedAdapterRouter([_manifest("a1"),
                               _manifest("a2", families=("tr-upper",))],
                              state_signer=s)
    r.revoke("a1")
    p = tmp_path / "registry.json"
    p.write_text(r.to_json())
    r2 = VerifiedAdapterRouter.from_json(p, registry_verifier=v)
    assert r2.state_authenticated and r2.state_complete
    assert "a1" in r2._revoked
    d = r2.route("reverse", family="tr-reverse", model_revision="rev1")
    assert d.selected is None
    assert any("revoked" in x for x in d.refused)


def test_deleted_revocation_in_signed_state_fails_closed(tmp_path):
    """THE security property: editing the registry file to remove a
    revocation breaks the signature — load refuses, access is NOT
    restored."""
    s, v = _sv()
    r = VerifiedAdapterRouter([_manifest("a1")], state_signer=s)
    r.revoke("a1")
    p = tmp_path / "registry.json"
    doc = json.loads(r.to_json())
    doc["state"]["revoked"] = []          # attacker deletes the revocation
    p.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="signature|tampered"):
        VerifiedAdapterRouter.from_json(p, registry_verifier=v)


def test_stale_snapshot_cannot_roll_back_revocation(tmp_path):
    """A still-validly-signed OLDER snapshot must not undo a newer
    revocation: min_seq is the high-water mark."""
    s, v = _sv()
    r = VerifiedAdapterRouter([_manifest("a1")], state_signer=s)
    old_doc = json.loads(r.to_json())     # seq before the revocation
    r.revoke("a1")                        # bumps seq
    p = tmp_path / "registry.json"
    p.write_text(json.dumps(old_doc))
    with pytest.raises(ValueError, match="seq|stale"):
        VerifiedAdapterRouter.from_json(p, registry_verifier=v,
                                        min_seq=r._seq)


def test_revocation_log_survives_registry_file_tampering(tmp_path):
    """The authoritative log is independent custody: even an UNSIGNED
    registry file with the revocation scrubbed still routes revoked
    adapters to refusal — the log unions them back in."""
    s, v = _sv()
    log = RevocationLog(tmp_path / "revocations.jsonl")
    r = VerifiedAdapterRouter([_manifest("a1")], state_signer=s,
                              revocation_log=log)
    r.revoke("a1")                        # appended to the authoritative log
    # write an unsigned v3 doc with the revocation erased
    doc = seal_state({"adapters": [asdict(_manifest("a1"))],
                      "revoked": [], "suspended": [], "health": {}},
                     seq=1)
    p = tmp_path / "registry.json"
    p.write_text(json.dumps(doc))
    r2 = VerifiedAdapterRouter.from_json(p, revocation_log=log)
    assert "a1" in r2._revoked            # log restores it
    assert r2.route("reverse", family="tr-reverse",
                    model_revision="rev1").selected is None


def test_revocation_log_rejects_forged_entries(tmp_path):
    s, v = _sv()
    forger = Ed25519Signer.generate()
    log = RevocationLog(tmp_path / "revocations.jsonl")
    log.append(forger, "a1", seq=1)       # signed, but not by OUR key
    revoked, failures, _ = log.load(v)
    assert revoked == set() and failures  # unverifiable -> never trusted
    # and a properly-signed entry verifies
    log2 = RevocationLog(tmp_path / "good.jsonl")
    log2.append(s, "a1", seq=2)
    revoked, failures, seq = log2.load(v)
    assert revoked == {"a1"} and not failures and seq == 2


def test_strict_state_requires_authentication(tmp_path):
    doc = seal_state({"adapters": [], "revoked": [], "suspended": [],
                      "health": {}}, seq=0)          # unsigned v3
    p = tmp_path / "registry.json"
    p.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="strict_state|authenticated"):
        VerifiedAdapterRouter.from_json(p, strict_state=True)


def test_open_state_rejects_unknown_schema():
    with pytest.raises(ValueError):
        open_state({"schema": "mini-agi-v4.0-adapter-registry-v9"})


def test_unsigned_v3_loads_flagged_not_authenticated(tmp_path):
    """Research mode may load unsigned state — but it must be flagged,
    never silently trusted."""
    doc = seal_state({"adapters": [asdict(_manifest("a1"))],
                      "revoked": ["a1"], "suspended": [], "health": {}},
                     seq=3)
    p = tmp_path / "registry.json"
    p.write_text(json.dumps(doc))
    r = VerifiedAdapterRouter.from_json(p)
    assert not r.state_authenticated
    assert any("unsigned" in w for w in r.registry_warnings)
    # state itself still applies — an unsigned file is a claim we load,
    # not proof; the revocation it declares is honored
    assert "a1" in r._revoked
