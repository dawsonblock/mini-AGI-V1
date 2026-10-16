"""Independently verifiable intervention-outcome receipts (V4-31, v4.0.3).

``capability_control.py record --verified`` previously accepted a bare
flag — an operator could mark any outcome "verified" with nothing
behind it. The v4.0.2 repair required ``--receipt PATH`` — but the
check was *integrity only*: a committed run whose inventory hashes
matched and which contained a METRICS.json was accepted as verified.
That is evidence of local file consistency, not scientific
qualification: a self-authored run carrying a fabricated METRICS.json
inside a hash-consistent RUN_COMMIT.json still passed.

v4.0.3 separates three verification states (see
``qualified_outcome.py``):

* **INTEGRITY_VERIFIED** — files internally consistent. Never feeds
  the controller's verified history.
* **EVALUATION_VERIFIED** — an independent evaluator recomputed the
  results: every committed prediction file is strict-rescored against
  the reported metrics and the identity bindings hold
  (``evidence_reconciliation.verify_run``), OR the evidence is a
  receipt signed by an evaluation-authorized identity.
* **QUALIFIED** — a signer authorized for the ``qualification`` role
  bound this run_id and evidence digest in a signed decision.

``verify_outcome_receipt`` returns ``verified: True`` ONLY for the two
admissible states. The returned ``receipt_digest`` binds the evidence
bytes (commit digest for run dirs, document digest for receipts), and
``outcome`` carries the derived record for run directories — numbers
come from evidence, never from the caller.
"""
from __future__ import annotations

import json
from pathlib import Path

from egai.common.canonical import digest
from egai.common.crypto import Ed25519Verifier, SignedEnvelope

from .authority import AuthorityRegistry
from .qualified_outcome import (  # noqa: F401  (state constants re-exported)
    CONTROLLER_ADMISSIBLE, SCHEMA as QUALIFIED_OUTCOME_SCHEMA,
    STATE_EVALUATION, STATE_INTEGRITY, STATE_QUALIFIED,
    derive_qualified_outcome)

RECEIPT_SCHEMAS = {
    "mini-agi-v16.1-executed-run-receipt-v1",
    "mini-agi-v16.2-executed-run-receipt-v1",
    "mini-agi-v16.5-evidence-receipt-v3",
    QUALIFIED_OUTCOME_SCHEMA,
}

#: Receipt signer roles accepted as independent evaluation authority.
EVALUATION_ROLES = ("evaluation", "qualification")


def _verify_run_dir(run_dir: Path, *,
                    verifier: Ed25519Verifier | None,
                    authority: AuthorityRegistry | None,
                    repo_root, corpus_rows) -> dict:
    oc = derive_qualified_outcome(run_dir, repo_root=repo_root,
                                  corpus_rows=corpus_rows,
                                  verifier=verifier,
                                  authority=authority)
    verified = oc.admissible
    reason = "" if verified else (
        "run did not reach an admissible verification state "
        f"({oc.verification_state}; "
        f"evaluation={oc.evaluation_status}, "
        f"qualification={oc.qualification_status}"
        + (f"; {oc.reasons[0]}" if oc.reasons else "") + ")")
    return {"verified": verified,
            "verification_state": oc.verification_state,
            "evaluation_status": oc.evaluation_status,
            "qualification_status": oc.qualification_status,
            "kind": "committed_run",
            "receipt_digest": oc.evidence_digest,
            "outcome_digest": oc.digest,
            "run_id": oc.run_id,
            "outcome": oc.to_dict(),
            "reason": reason}


def _verify_receipt_doc(doc: dict, verifier,
                        authority: AuthorityRegistry | None,
                        required_roles: tuple[str, ...]) -> dict:
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
    if verifier is None and authority is None:
        return {"verified": False,
                "reason": "signed receipt supplied but no verifier/"
                          "trust root was given — a signature that "
                          "cannot be checked is not verification"}
    if authority is not None and not any(
            authority.is_authorized(role, kid)
            for role in required_roles):
        return {"verified": False,
                "reason": f"receipt signer {kid} is not authorized for "
                          f"any of {required_roles} — a signature from "
                          "an unauthorized role is not evaluation "
                          "evidence"}
    if verifier is None:
        return {"verified": False,
                "reason": "receipt signature cannot be checked — no "
                          "verifier covering the signer key"}
    body = {k: v for k, v in r.items() if k != "signature_b64"}
    if not verifier.verify(body, SignedEnvelope(kid, sig)):
        return {"verified": False, "reason": "receipt signature invalid"}
    out = {"verified": True, "kind": "signed_receipt",
           "verification_state": STATE_EVALUATION,
           "evaluation_status": STATE_EVALUATION,
           "qualification_status": "NOT_QUALIFIED",
           "receipt_digest": digest(r),
           "receipt_schema": schema,
           "signer_key_id": kid}
    if schema == QUALIFIED_OUTCOME_SCHEMA:
        # An evaluator-signed qualified-outcome document carries the
        # outcome numbers the evaluator attested — evidence-derived,
        # not caller-supplied. The caller's controller consumes the
        # attested fields verbatim.
        out["outcome"] = {k: v for k, v in r.items()
                          if k != "signature_b64"}
        out["outcome_digest"] = digest(out["outcome"])
        if r.get("qualification_status") == "QUALIFIED":
            out["verification_state"] = STATE_QUALIFIED
            out["qualification_status"] = "QUALIFIED"
            out["qualification_digest"] = r.get("qualification_digest")
    return out


def verify_outcome_receipt(path, verifier: Ed25519Verifier | None = None,
                           *, authority: AuthorityRegistry | None = None,
                           repo_root=None, corpus_rows=None,
                           required_roles: tuple[str, ...]
                           = EVALUATION_ROLES) -> dict:
    """Verify an outcome receipt independently of the caller's claim.

    ``path`` may be a run directory, a RUN_COMMIT.json, or a signed
    receipt JSON document. ``verified`` is True only when the evidence
    reaches EVALUATION_VERIFIED or QUALIFIED — local integrity alone
    (a self-authored commit) never verifies. Returns ``receipt_digest``
    binding the evidence bytes plus ``verification_state`` for audit.
    """
    p = Path(path)
    if p.is_dir():
        return _verify_run_dir(p, verifier=verifier, authority=authority,
                               repo_root=repo_root, corpus_rows=corpus_rows)
    if p.is_file():
        if p.name == "RUN_COMMIT.json":
            return _verify_run_dir(p.parent, verifier=verifier,
                                   authority=authority,
                                   repo_root=repo_root,
                                   corpus_rows=corpus_rows)
        try:
            doc = json.loads(p.read_text())
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            return {"verified": False,
                    "reason": f"unparseable receipt: {exc}"}
        return _verify_receipt_doc(doc, verifier, authority,
                                   required_roles)
    return {"verified": False, "reason": f"no such path: {p}"}
