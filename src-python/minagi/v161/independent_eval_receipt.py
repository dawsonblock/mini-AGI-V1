"""Independent evaluation receipts (v4.0.3-r1 / Phase 1).

The evidence stack prior to this release had exactly three grades:

* INTEGRITY_VERIFIED — the committed file inventory hashes match;
* EVALUATION_VERIFIED — ``evidence_reconciliation.verify_run``
  recomputed every committed prediction and the reported metrics
  reproduce — but the "evaluator" was the same process reading files
  the training pipeline wrote;
* QUALIFIED — a qualification-authorized signer bound the evidence
  digest in a decision.

The audit finding this module closes: a locally supplied experiment
could be recomputed locally and then labelled EVALUATION_VERIFIED even
when no evaluator with independent authority ever saw it. A model
training process could therefore produce its own authoritative
evaluation — the scientific trust boundary the release must close.

``IndependentEvaluationReceipt`` is the missing layer. An evaluator
holding an ``evaluation``-role key (see ``authority.py``) recomputes a
committed run itself — prediction rescore, metric agreement, identity
bindings — and signs a canonical record that BINDS the exact evidence
it measured:

    experiment_id               which run this attestation covers
    candidate_digest            adapter artifact bytes (the candidate)
    base_model_digest           declared frozen model id+revision
    dataset_digest              corpus bytes the predictions were scored
                                against
    scorer_digest               scoring oracle identity
    protocol_digest             evaluation protocol the run declared
    predictions_digest          every committed prediction file's bytes
    evaluation_metrics_digest   METRICS.json bytes + the rescore summary
    retention_metrics_digest    recomputed retention destruction/
                                acquisition accounting
    security_metrics_digest     governed security-lane evidence, when
                                supplied to the evaluator
    evaluator_authority_id      the signing identity
    evaluation_status           "INDEPENDENTLY_VERIFIED"
    signature                   Ed25519 over the canonical body

Verification of a receipt is not a signature check alone:

1. the schema, status and every required field must be present;
2. ``evaluator_authority_id`` must be authorized for the ``evaluation``
   role under a caller-supplied ``AuthorityRegistry`` — a signature from
   any other role, an expired key, a revoked key or an unregistered key
   is not evaluation evidence;
3. the Ed25519 signature must verify over the canonical body;
4. when the bound run directory is available (always for the trusted
   import path), the verifier re-runs ``verify_run`` — which must still
   be VERIFIED — and recomputes every binding field. A receipt naming
   another run, another model, altered predictions, different metrics
   or a different corpus does not match and fails.

A detached receipt that cannot be bound to evidence is not independent
verification of anything — it is refused for trusted import. Replay is
covered the same way: the receipt binds the run's bytes, so a receipt
issued for run A cannot stand in for run B.

Issuance is deliberately strict: the evaluator signs only when its own
recomputation reaches ``verify_run`` == VERIFIED. The measurements it
attests may themselves be *negative* — a candidate that honestly
destroyed retention still gets a receipt, because a verified negative
outcome is exactly the evidence the learning controller needs to avoid
selection bias.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

from egai.common.canonical import digest, validate_digest
from egai.common.crypto import Ed25519Signer, Ed25519Verifier, SignedEnvelope

from .authority import AuthorityRegistry
from .evidence_reconciliation import (
    AUTHORIZED_SCORER, VERIFIED, adapter_artifact_digest_of,
    sha256_file, verify_run)

RECEIPT_SCHEMA = "independent-evaluation-v1"
RECEIPT_FILE = "EVALUATION_RECEIPT.json"
INDEPENDENT_STATUS = "INDEPENDENTLY_VERIFIED"
EVALUATOR_ROLE = "evaluation"

#: The evidence-identity fields every receipt must carry (key presence
#: is required; a value may be null only when the evidence has no such
#: artifact — e.g. an adapter-free frozen-baseline run).
BINDING_FIELDS = (
    "experiment_id",
    "candidate_digest",
    "base_model_digest",
    "dataset_digest",
    "scorer_digest",
    "protocol_digest",
    "predictions_digest",
    "evaluation_metrics_digest",
    "retention_metrics_digest",
    "security_metrics_digest",
)


def _load_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        doc = json.loads(path.read_text())
    except (json.JSONDecodeError, UnicodeDecodeError):
        return {}
    return doc if isinstance(doc, dict) else {}


def _retention_summary(run_dir: Path) -> dict | None:
    """Recomputed retention accounting from the committed retention
    prediction files — the quantities an evaluator attests, derived the
    same way ``verify_run`` checks them (frozen-correct rows destroyed,
    frozen-wrong rows acquired)."""
    from .evaluators import exact_match_strict, score_row
    f_path = run_dir / "PREDICTIONS-FROZEN-RETENTION.jsonl"
    a_path = run_dir / "PREDICTIONS-ADAPTED-RETENTION.jsonl"
    if not (f_path.is_file() and a_path.is_file()):
        return None

    def _scores(p: Path) -> dict[str, float]:
        rows = [json.loads(ln) for ln in p.read_text().splitlines()
                if ln.strip()]
        return {str(r["id"]): float(score_row(
            {"expected": r.get("expected", ""), "verify": r.get("verify")},
            str(r.get("output", "")), exact_match_strict))
            for r in rows}
    f_sc, a_sc = _scores(f_path), _scores(a_path)
    n_fc = int(sum(v == 1.0 for v in f_sc.values()))
    n_wrong = len(f_sc) - n_fc
    destroyed = sorted(r for r in f_sc
                       if f_sc[r] == 1.0 and a_sc.get(r) == 0.0)
    gained = sorted(r for r in f_sc
                    if f_sc[r] == 0.0 and a_sc.get(r) == 1.0)
    return {"n_frozen_correct": n_fc,
            "destroyed_ids": destroyed, "n_destroyed": len(destroyed),
            "gained_ids": gained, "n_gained": len(gained),
            "destruction_rate": (len(destroyed) / n_fc) if n_fc else None,
            "acquisition_rate": (len(gained) / n_wrong) if n_wrong else None}


def evidence_binding(run_dir, *, repo_root=None, corpus_rows=None,
                     security_doc=None, verification=None) -> dict:
    """Recompute the evidence-identity binding for a committed run.

    Every field is derived from evidence bytes on disk (or the caller's
    supplied corpus/security evidence) — nothing is taken on trust from
    the receipt or the caller. The same function runs at issue time and
    at verify time; any disagreement is a forgery, replay or corruption
    signal.

    ``verification`` is an optional precomputed ``verify_run`` result;
    when absent the function runs the reconciler itself.
    """
    run_dir = Path(run_dir)
    if verification is None:
        verification = verify_run(run_dir, repo_root=repo_root,
                                  corpus_rows=corpus_rows)
    config = _load_json(run_dir / "CONFIG.json")

    pred_files = sorted(run_dir.glob("PREDICTIONS-*.jsonl"))
    predictions_digest = digest(
        {p.name: sha256_file(p) for p in pred_files}) \
        if pred_files else None

    candidate = None
    adir = run_dir / "adapter"
    if adir.is_dir():
        try:
            candidate = adapter_artifact_digest_of(adir)
        except ValueError:
            candidate = None

    dataset_digest = None
    if corpus_rows is not None:
        dataset_digest = digest([dict(r) for r in corpus_rows])
    elif repo_root and config.get("corpus"):
        cpath = Path(repo_root) / str(config["corpus"])
        if cpath.is_file():
            dataset_digest = "sha256:" + sha256_file(cpath)

    protocol_sha = None
    if repo_root and config.get("config"):
        ppath = Path(repo_root) / str(config["config"])
        if ppath.is_file():
            protocol_sha = "sha256:" + sha256_file(ppath)

    metrics_path = run_dir / "METRICS.json"
    evaluation_metrics = {
        "metrics_sha256": ("sha256:" + sha256_file(metrics_path))
        if metrics_path.is_file() else None,
        "rescore": verification.get("rescore") or {},
        "reported": verification.get("reported") or {},
    }

    return {
        "experiment_id": run_dir.name,
        "candidate_digest": candidate,
        "base_model_digest": digest({
            "model_id": config.get("model_id"),
            "model_revision": config.get("model_revision")}),
        "dataset_digest": dataset_digest,
        "scorer_digest": digest({
            "score_oracle": config.get("score_oracle"),
            "authorized_scorer": AUTHORIZED_SCORER}),
        "protocol_digest": digest({
            "protocol": config.get("config"),
            "protocol_sha256": protocol_sha}),
        "predictions_digest": predictions_digest,
        "evaluation_metrics_digest": digest(evaluation_metrics),
        "retention_metrics_digest": (
            digest(_retention_summary(run_dir))
            if (run_dir / "PREDICTIONS-FROZEN-RETENTION.jsonl").is_file()
            or (run_dir / "PREDICTIONS-ADAPTED-RETENTION.jsonl").is_file()
            else None),
        "security_metrics_digest": (
            digest(security_doc) if security_doc is not None else None),
    }


def issue_evaluation_receipt(signer: Ed25519Signer, run_dir, *,
                             repo_root=None, corpus_rows=None,
                             security_doc=None) -> dict:
    """Evaluator-side constructor: measure the run independently and
    sign the binding record.

    The evaluator MUST recompute the evidence itself — this function
    refuses to sign any run whose reconciliation is not VERIFIED, so an
    evaluator cannot attest evidence its own rescore contradicts. The
    private key must be one the research agent cannot reach: that is an
    operational custody requirement (separate user/host/HSM), not a
    property this function can enforce.
    """
    run_dir = Path(run_dir)
    verification = verify_run(run_dir, repo_root=repo_root,
                              corpus_rows=corpus_rows)
    if verification.get("status") != VERIFIED:
        raise ValueError(
            "refusing to issue an evaluation receipt: independent "
            f"recomputation is {verification.get('status')}, not "
            f"VERIFIED (first mismatch: "
            f"{(verification.get('mismatches') or ['none'])[0]})")
    binding = evidence_binding(
        run_dir, repo_root=repo_root, corpus_rows=corpus_rows,
        security_doc=security_doc, verification=verification)
    body = {"schema": RECEIPT_SCHEMA, **binding,
            "evaluator_authority_id": signer.key_id,
            "evaluation_status": INDEPENDENT_STATUS}
    env = signer.sign(body)
    return {**body, "signature": env.signature_b64}


def write_receipt_file(run_dir, receipt: Mapping) -> Path:
    """Persist a receipt beside the committed evidence.

    EVALUATION_RECEIPT.json is post-commit evidence — deliberately not
    part of the run's own inventory, exactly like QUALIFICATION.json:
    the reconciler records it as an uncommitted file, and the receipt
    binds the committed bytes, so a decision over different bytes does
    not cover this run.
    """
    p = Path(run_dir) / RECEIPT_FILE
    p.write_text(json.dumps(dict(receipt), indent=2, sort_keys=True)
                 + "\n")
    return p


def verify_evaluation_receipt(doc, *, verifier: Ed25519Verifier | None,
                              authority: AuthorityRegistry | None,
                              run_dir=None, repo_root=None,
                              corpus_rows=None, security_doc=None,
                              now=None, allow_detached=False) -> dict:
    """Verify an independent evaluation receipt.

    ``verified`` is True only when ALL of the following hold:

    * schema ``independent-evaluation-v1`` and every binding field
      present, ``evaluation_status`` == INDEPENDENTLY_VERIFIED;
    * ``evaluator_authority_id`` is authorized for the ``evaluation``
      role in ``authority`` at ``now`` (a signature from any other role
      is not evaluation evidence);
    * the Ed25519 signature verifies under ``verifier``;
    * the evidence binding recomputes exactly — ``verify_run`` on the
      bound run must be VERIFIED and every binding field must match.

    ``allow_detached`` permits signature+role verification WITHOUT the
    binding check for tooling that cannot reach the evidence; the
    result then reports ``binding_checked: False`` and is NOT suitable
    for trusted learning-history import (exact evidence identities were
    never validated).
    """
    out = {"verified": False, "binding_checked": False,
           "signer_key_id": "", "reason": "", "mismatches": []}
    fail = out["mismatches"].append

    def done(reason):
        out["reason"] = reason
        return out

    if not isinstance(doc, Mapping):
        return done("receipt is not a JSON object")
    if doc.get("schema") != RECEIPT_SCHEMA:
        return done(f"receipt schema must be {RECEIPT_SCHEMA!r}")
    missing = [f for f in BINDING_FIELDS if f not in doc]
    if missing:
        return done(f"receipt lacks binding fields {missing}")
    if doc.get("evaluation_status") != INDEPENDENT_STATUS:
        return done("evaluation_status is "
                    f"{doc.get('evaluation_status')!r}, not "
                    f"{INDEPENDENT_STATUS!r}")
    kid = str(doc.get("evaluator_authority_id") or "")
    sig = str(doc.get("signature") or "")
    out["signer_key_id"] = kid
    if not kid or not sig:
        return done("receipt is unsigned — self-asserted verification "
                    "is not independent evaluation")
    for f in BINDING_FIELDS:
        v = doc.get(f)
        if v is not None and f.endswith("_digest"):
            try:
                validate_digest(v)
            except ValueError:
                return done(f"{f} is not a sha256 digest: {v!r}")
    if authority is None:
        return done("no authority registry supplied — the signer's "
                    "evaluation role cannot be established; a signature "
                    "that cannot be role-checked is not independent "
                    "verification")
    if not authority.is_authorized(EVALUATOR_ROLE, kid, now=now):
        rec = authority.record_for(kid)
        why = ("unregistered signing identity" if rec is None else
               f"key registered for role {rec.role!r}, not "
               f"{EVALUATOR_ROLE!r}")
        return done(f"receipt signer {kid} is not authorized for role "
                    f"'{EVALUATOR_ROLE}': {why}")
    if verifier is None:
        return done("receipt signature cannot be checked — no verifier "
                    "covering the signer key")
    body = {k: v for k, v in doc.items() if k != "signature"}
    if not verifier.verify(body, SignedEnvelope(kid, sig)):
        return done("receipt signature invalid")

    if run_dir is None:
        if allow_detached:
            out["verified"] = True
            return done("detached signature+role verification only — "
                        "exact evidence identities were NOT validated")
        return done("receipt cannot be bound to evidence — exact "
                    "evidence identities unvalidated; a detached "
                    "receipt is not independent verification")

    run_dir = Path(run_dir)
    verification = verify_run(run_dir, repo_root=repo_root,
                              corpus_rows=corpus_rows)
    if verification.get("status") != VERIFIED:
        return done("bound evidence does not reconcile — "
                    f"verify_run is {verification.get('status')}, not "
                    "VERIFIED")
    if str(doc["experiment_id"]) != \
            str((verification.get("run_id") or run_dir.name)):
        return done("receipt experiment_id does not match the run")
    actual = evidence_binding(
        run_dir, repo_root=repo_root, corpus_rows=corpus_rows,
        security_doc=security_doc, verification=verification)
    for f in BINDING_FIELDS:
        if doc.get(f) != actual.get(f):
            fail(f"{f}: receipt {doc.get(f)!r} != evidence "
                 f"{actual.get(f)!r}")
    if out["mismatches"]:
        return done("receipt does not bind this evidence — the "
                    "evaluator measured different bytes")
    out["binding_checked"] = True
    out["verified"] = True
    return done("")
