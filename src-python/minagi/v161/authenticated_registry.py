"""Authenticated adapter-registry state (V4-28).

An adapter *manifest* signature protects one manifest — it says nothing
about the registry's mutable safety state. The v2 registry file stored
``revoked``/``suspended``/``health`` as unsigned JSON: deleting a
revocation from the file silently restored access, and loading an
older still-valid snapshot could silently undo a newer revocation.

This module provides the two mechanisms that close that gap:

* **Signed, monotonically sequenced state.** ``seal_state`` binds the
  registry body to a monotonic ``seq`` under an authority signature;
  ``open_state`` refuses a state whose signature does not verify
  (tampering — including a deleted revocation — fails closed) and a
  state whose ``seq`` is below the caller's ``min_seq`` (a stale
  snapshot cannot roll safety state backwards).
* **Authoritative revocation log.** An append-only JSONL of signed
  revocation entries, kept under the revocation authority's custody.
  Loading a registry merges the log's verified revocations over the
  file's own list, so removing a revocation from a registry snapshot —
  signed or not — never restores access while the log survives.

The router (``verified_routing.py``) consumes these; nothing here knows
about routing.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from egai.common.canonical import validate_digest  # noqa: F401  (re-export contract)
from egai.common.crypto import Ed25519Signer, Ed25519Verifier, SignedEnvelope

SCHEMA_REGISTRY_V3 = "mini-agi-v4.0-adapter-registry-v3"
SCHEMA_REVOCATION = "mini-agi-v4.0-revocation-entry-v1"


def seal_state(state: dict, seq: int,
               signer: Ed25519Signer | None = None) -> dict:
    """Serialize registry state as a schema-v3 document.

    ``state`` carries the mutable safety fields (adapters, revoked,
    suspended, health). With ``signer`` the document is an authenticated
    authorization snapshot; without one it is a claim about state —
    loadable only in non-strict mode."""
    doc = {"schema": SCHEMA_REGISTRY_V3, "seq": int(seq),
           "state": dict(state)}
    if signer is not None:
        env = signer.sign(doc)
        doc["signer_key_id"] = env.key_id
        doc["signature_b64"] = env.signature_b64
    return doc


def open_state(doc: dict,
               verifier: Ed25519Verifier | None = None, *,
               min_seq: int = 0) -> dict:
    """Open a schema-v3 registry document.

    Returns ``{"state", "seq", "authenticated"}``. With a verifier, a
    missing or invalid signature raises — tampered state fails closed.
    ``min_seq`` is the caller's high-water mark: a snapshot older than
    it cannot silently undo newer safety actions."""
    if doc.get("schema") != SCHEMA_REGISTRY_V3:
        raise ValueError(f"registry schema must be {SCHEMA_REGISTRY_V3}")
    seq = int(doc.get("seq", -1))
    signed = bool(doc.get("signer_key_id"))
    if verifier is not None:
        body = {k: v for k, v in doc.items()
                if k in ("schema", "seq", "state")}
        if not signed or not verifier.verify(
                body, SignedEnvelope(str(doc.get("signer_key_id")),
                                     str(doc.get("signature_b64", "")))):
            raise ValueError(
                "registry state signature invalid — refusing tampered "
                "or unauthenticated state")
    if seq < int(min_seq):
        raise ValueError(
            f"registry state seq {seq} predates required minimum "
            f"{min_seq} — a stale snapshot cannot roll back newer "
            "revocations")
    state = doc.get("state")
    if not isinstance(state, dict):
        raise ValueError("registry state body missing")
    return {"state": state, "seq": seq, "authenticated": signed}


class RevocationLog:
    """Append-only signed revocation ledger.

    Each entry binds ``(adapter_id, seq)`` under the revocation
    authority's signature. ``load`` returns every adapter whose entry
    verifies — entries that fail verification are reported, never
    trusted. Unioned over a loaded registry's own revocation list, the
    log makes revocation deletion ineffective.
    """

    def __init__(self, path):
        self.path = Path(path)

    def append(self, signer: Ed25519Signer, adapter_id: str,
               seq: int) -> dict:
        body = {"schema": SCHEMA_REVOCATION, "adapter_id": str(adapter_id),
                "seq": int(seq),
                "revoked_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                             time.gmtime()),
                "signer_key_id": signer.key_id}
        env = signer.sign(body)
        entry = {**body, "signature_b64": env.signature_b64}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as fh:
            fh.write(json.dumps(entry, sort_keys=True,
                                separators=(",", ":")) + "\n")
        return entry

    def load(self, verifier: Ed25519Verifier | None = None
             ) -> tuple[set[str], list[str], int]:
        """(revoked adapter ids, integrity failures, max verified seq)."""
        revoked: set[str] = set()
        failures: list[str] = []
        max_seq = -1
        if not self.path.is_file():
            return revoked, failures, max_seq
        for i, line in enumerate(self.path.read_text().splitlines()):
            if not line.strip():
                continue
            where = f"revocation_log[{i}]"
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                failures.append(f"{where}: unparseable")
                continue
            if e.get("schema") != SCHEMA_REVOCATION:
                failures.append(f"{where}: bad schema")
                continue
            body = {k: v for k, v in e.items() if k != "signature_b64"}
            kid, sig = str(e.get("signer_key_id", "")), \
                str(e.get("signature_b64", ""))
            if verifier is not None:
                if not kid or not sig or not verifier.verify(
                        body, SignedEnvelope(kid, sig)):
                    failures.append(f"{where}: signature invalid")
                    continue
            elif not kid or not sig:
                failures.append(f"{where}: unsigned entry")
                continue
            aid = e.get("adapter_id")
            if not aid:
                failures.append(f"{where}: adapter_id missing")
                continue
            revoked.add(str(aid))
            max_seq = max(max_seq, int(e.get("seq", -1)))
        return revoked, failures, max_seq
