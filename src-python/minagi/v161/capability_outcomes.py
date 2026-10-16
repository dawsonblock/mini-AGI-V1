"""Independently verifiable intervention-outcome receipts (V4-31).

``capability_control.py record --verified`` previously accepted a bare
flag — an operator could mark any outcome "verified" with nothing
behind it. Verified now requires ``--receipt PATH`` naming evidence that
this module can check on its own:

* a committed run directory (or its RUN_COMMIT.json): status COMPLETE,
  every inventoried file present with matching hash, METRICS.json
  present — the same integrity the campaign requalifier enforces;
* a signed receipt document (ExecutedRunReceiptV161/V162 or
  EvidenceReceiptV3, flat or wrapped as ``{"receipt": ...}``): requires
  ``verifier`` — a signature that cannot be checked is not
  verification.

The returned record carries ``receipt_digest`` so the history entry is
bound to specific evidence bytes rather than to a flag.
"""
from __future__ import annotations

import json
from pathlib import Path

from egai.common.canonical import digest
from egai.common.crypto import Ed25519Verifier, SignedEnvelope

from .evidence_reconciliation import sha256_file

RECEIPT_SCHEMAS = {
    "mini-agi-v16.1-executed-run-receipt-v1",
    "mini-agi-v16.2-executed-run-receipt-v1",
    "mini-agi-v16.5-evidence-receipt-v3",
}


def _verify_run_dir(run_dir: Path) -> dict:
    commit_path = run_dir / "RUN_COMMIT.json"
    if not commit_path.is_file():
        return {"verified": False,
                "reason": "RUN_COMMIT.json missing"}
    commit = json.loads(commit_path.read_text())
    if commit.get("status") != "COMPLETE":
        return {"verified": False,
                "reason": f"commit status is "
                          f"{commit.get('status')!r}, not COMPLETE"}
    bad = []
    for rel, want in (commit.get("files") or {}).items():
        fp = run_dir / rel
        if not fp.is_file():
            bad.append(f"{rel}: missing")
        elif sha256_file(fp) != want:
            bad.append(f"{rel}: hash mismatch")
    if bad:
        return {"verified": False,
                "reason": f"commit inventory mismatch: {bad[:3]}"}
    if not (run_dir / "METRICS.json").is_file():
        return {"verified": False, "reason": "METRICS.json missing"}
    return {"verified": True, "kind": "committed_run",
            "receipt_digest": digest(commit),
            "run_id": commit.get("run_id"),
            "config_digest": commit.get("config_digest")}


def _verify_receipt_doc(doc: dict, verifier) -> dict:
    r = doc.get("receipt") if isinstance(doc.get("receipt"), dict) else doc
    schema = r.get("schema")
    if schema not in RECEIPT_SCHEMAS:
        return {"verified": False,
                "reason": f"unrecognized receipt schema {schema!r}"}
    kid = str(r.get("signer_key_id", ""))
    sig = str(r.get("signature_b64", ""))
    if not kid or not sig:
        return {"verified": False,
                "reason": "receipt carries no signature"}
    if verifier is None:
        return {"verified": False,
                "reason": "signed receipt supplied but no verifier/"
                          "trust root was given — a signature that "
                          "cannot be checked is not verification"}
    body = {k: v for k, v in r.items() if k != "signature_b64"}
    if not verifier.verify(body, SignedEnvelope(kid, sig)):
        return {"verified": False, "reason": "receipt signature invalid"}
    return {"verified": True, "kind": "signed_receipt",
            "receipt_digest": digest(r),
            "receipt_schema": schema,
            "signer_key_id": kid}


def verify_outcome_receipt(path, verifier: Ed25519Verifier | None = None
                           ) -> dict:
    """Verify an outcome receipt independently of the caller's claim.

    ``path`` may be a run directory, a RUN_COMMIT.json, or a signed
    receipt JSON document. Returns a dict with ``verified`` plus —
    on success — ``receipt_digest`` binding the evidence bytes."""
    p = Path(path)
    if p.is_dir():
        return _verify_run_dir(p)
    if p.is_file():
        if p.name == "RUN_COMMIT.json":
            return _verify_run_dir(p.parent)
        try:
            doc = json.loads(p.read_text())
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            return {"verified": False,
                    "reason": f"unparseable receipt: {exc}"}
        return _verify_receipt_doc(doc, verifier)
    return {"verified": False, "reason": f"no such path: {p}"}
