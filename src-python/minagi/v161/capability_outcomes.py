"""Independently verifiable intervention-outcome receipts (v4.0.3-r1).

``capability_control.py record --verified`` previously accepted a bare
flag — an operator could mark any outcome "verified" with nothing
behind it. v4.0.3 separated three states (integrity / local
recomputation / qualification), but the middle grade — a locally
recomputed run labelled EVALUATION_VERIFIED — still let a self-authored
experiment reach trusted history without any evaluator holding
independent authority ever seeing it.

v4.0.3-r1 defines the four-level boundary (see
``qualified_outcome.py``):

* **INTEGRITY_VERIFIED** — files internally consistent. Never feeds
  the controller's verified history.
* **LOCALLY_RECOMPUTED** — every committed prediction file
  strict-rescores and the identity bindings hold
  (``evidence_reconciliation.verify_run``). Local reproduction only —
  never feeds verified history.
* **INDEPENDENTLY_VERIFIED** — a signed ``independent-evaluation-v1``
  receipt from an ``evaluation``-role authority binds the exact
  evidence identities (``independent_eval_receipt``). Detached receipts
  that cannot be bound to the evidence they name are refused for
  trusted import. Verified NEGATIVE outcomes are admissible too — a
  rejected candidate's real measurements are evidence the controller
  must keep, or it learns only from successes.
* **QUALIFIED** — a signer authorized for the ``qualification`` role
  bound this run_id and evidence digest in a signed decision, on top
  of the independent evaluation.

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
from .independent_eval_receipt import (
    RECEIPT_FILE, RECEIPT_SCHEMA, verify_evaluation_receipt)
from .qualified_outcome import (  # noqa: F401  (state constants re-exported)
    CONTROLLER_ADMISSIBLE, SCHEMA as QUALIFIED_OUTCOME_SCHEMA,
    STATE_EVALUATION, STATE_INDEPENDENT, STATE_INTEGRITY, STATE_LOCAL,
    STATE_QUALIFIED, derive_qualified_outcome)

RECEIPT_SCHEMAS = {
    "mini-agi-v16.1-executed-run-receipt-v1",
    "mini-agi-v16.2-executed-run-receipt-v1",
    "mini-agi-v16.5-evidence-receipt-v3",
    QUALIFIED_OUTCOME_SCHEMA,
    RECEIPT_SCHEMA,
}

#: Receipt signer roles accepted as independent evaluation authority.
EVALUATION_ROLES = ("evaluation", "qualification")


def _verify_run_dir(run_dir: Path, *,
                    verifier: Ed25519Verifier | None,
                    authority: AuthorityRegistry | None,
                    repo_root, corpus_rows, security_doc) -> dict:
    oc = derive_qualified_outcome(run_dir, repo_root=repo_root,
                                  corpus_rows=corpus_rows,
                                  verifier=verifier,
                                  authority=authority,
                                  security_doc=security_doc)
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
                        required_roles: tuple[str, ...],
                        bind_run_dir: Path | None = None,
                        repo_root=None, security_doc=None) -> dict:
    r = doc.get("receipt") if isinstance(doc.get("receipt"), dict) else doc
    schema = r.get("schema")
    if schema not in RECEIPT_SCHEMAS:
        return {"verified": False,
                "reason": f"unrecognized receipt schema {schema!r}"}

    if schema == RECEIPT_SCHEMA:
        # The authoritative v4.0.3-r1 receipt: signature + evaluation
        # role + exact evidence-identity binding. A detached receipt
        # that cannot be bound to its run is refused for trusted
        # import — exact evidence identities are mandatory.
        res = verify_evaluation_receipt(
            r, verifier=verifier, authority=authority,
            run_dir=bind_run_dir, repo_root=repo_root,
            security_doc=security_doc)
        if not res.get("verified"):
            return {"verified": False,
                    "verification_state": None,
                    "reason": res.get("reason"),
                    "signer_key_id": res.get("signer_key_id", "")}
        out = {"verified": True, "kind": "signed_receipt",
               "verification_state": STATE_INDEPENDENT,
               "evaluation_status": STATE_INDEPENDENT,
               "qualification_status": "NOT_QUALIFIED",
               "receipt_digest": digest(r),
               "receipt_schema": schema,
               "signer_key_id": res.get("signer_key_id", ""),
               "experiment_id": r.get("experiment_id"),
               "binding_checked": True}
        return out

    # ---- legacy signed receipt schemas -------------------------------
    # A signed attestation by an authorized evaluator/qualifier is
    # INDEPENDENTLY_VERIFIED — the signer is the independent authority.
    # Role authorization is REQUIRED: a signature from an unvetted key
    # proves nothing about evaluation authority.
    kid = str(r.get("signer_key_id", ""))
    sig = str(r.get("signature_b64", ""))
    if not kid or not sig:
        return {"verified": False,
                "reason": "receipt carries no signature"}
    if authority is None:
        return {"verified": False,
                "reason": "no trust root supplied — the signer's "
                          "authority role cannot be established; a "
                          "signature that cannot be role-checked is "
                          "not independent verification"}
    if not any(
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
           "verification_state": STATE_INDEPENDENT,
           "evaluation_status": STATE_INDEPENDENT,
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
                           security_doc=None,
                           required_roles: tuple[str, ...]
                           = EVALUATION_ROLES) -> dict:
    """Verify an outcome receipt independently of the caller's claim.

    ``path`` may be a run directory, a RUN_COMMIT.json, or a signed
    receipt JSON document. ``verified`` is True only when the evidence
    reaches INDEPENDENTLY_VERIFIED or QUALIFIED — local integrity and
    local recomputation alone never verify. Returns ``receipt_digest``
    binding the evidence bytes plus ``verification_state`` for audit.

    ``security_doc`` is the governed security-lane document the
    evaluator bound into ``security_metrics_digest`` at issue time; a
    receipt whose binding includes security evidence cannot verify
    without it (the digest would not recompute).

    For an ``independent-evaluation-v1`` receipt document, the bound
    run is resolved automatically when the receipt file sits inside the
    committed run directory (EVALUATION_RECEIPT.json next to
    RUN_COMMIT.json); otherwise the receipt is refused as unbindable.
    """
    p = Path(path)
    if p.is_dir():
        return _verify_run_dir(p, verifier=verifier, authority=authority,
                               repo_root=repo_root, corpus_rows=corpus_rows,
                               security_doc=security_doc)
    if p.is_file():
        if p.name == "RUN_COMMIT.json":
            return _verify_run_dir(p.parent, verifier=verifier,
                                   authority=authority,
                                   repo_root=repo_root,
                                   corpus_rows=corpus_rows,
                                   security_doc=security_doc)
        try:
            doc = json.loads(p.read_text())
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            return {"verified": False,
                    "reason": f"unparseable receipt: {exc}"}
        bind = None
        r = doc.get("receipt") if isinstance(doc.get("receipt"), dict) \
            else doc
        if r.get("schema") == RECEIPT_SCHEMA:
            # Auto-bind: a receipt committed beside its run binds the
            # run dir it lives in; anywhere else it cannot be checked.
            if p.name == RECEIPT_FILE \
                    and (p.parent / "RUN_COMMIT.json").is_file():
                bind = p.parent
        return _verify_receipt_doc(doc, verifier, authority,
                                   required_roles, bind_run_dir=bind,
                                   repo_root=repo_root,
                                   security_doc=security_doc)
    return {"verified": False, "reason": f"no such path: {p}"}
