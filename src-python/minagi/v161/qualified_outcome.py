"""Authoritative qualification-bound outcome records (v4.0.3-r1).

The v4.0.2 receipt check answered "are this run's files internally
consistent?" — commit status COMPLETE, inventory hashes matching,
METRICS.json present. That is *integrity*, and it is not evidence an
experiment produced the numbers it claims: a self-authored directory
with a fabricated METRICS.json and a hash-consistent RUN_COMMIT.json
sailed through as ``verified`` and entered the controller's history.

v4.0.3-r1 separates verification into FOUR levels (the audit's
required evidence states):

* **INTEGRITY_VERIFIED** — the commit and its file inventory are
  internally consistent. Necessary, never sufficient. NOT eligible
  for trusted learning history.
* **LOCALLY_RECOMPUTED** — ``evidence_reconciliation.verify_run``
  recomputed every committed prediction file and the reported metrics
  reproduce. This is *local* reproduction: the "evaluator" is the same
  codebase reading files the training pipeline wrote, so a fabricated
  but self-consistent run reaches this level. NOT eligible for trusted
  learning history.
* **INDEPENDENTLY_VERIFIED** — additionally, a signed
  ``independent-evaluation-v1`` receipt from an ``evaluation``-role
  authority binds this run's exact evidence identities (predictions,
  metrics, model/adapter, dataset, scorer, protocol). A training
  process cannot generate its own receipt — the evaluator's key is a
  different authority. ELIGIBLE for trusted learning history — including
  verified NEGATIVE outcomes (a rejected candidate's real measurements
  are evidence; dropping them would bias the controller toward
  successes).
* **QUALIFIED** — additionally, a signed qualification decision by a
  signer authorized for the ``qualification`` role binds this run_id
  and this evidence digest. ELIGIBLE for promotion decisions.

Only records at INDEPENDENTLY_VERIFIED or QUALIFIED may feed verified
outcome history to the learning controller. The numeric fields are
populated at LOCALLY_RECOMPUTED and above — they are derived from the
rescored prediction records, never from caller-supplied numbers; the
``admissible`` gate is what keeps them out of trusted history until an
authorized evaluator has signed the binding.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping

from egai.common.canonical import digest
from egai.common.crypto import Ed25519Verifier, SignedEnvelope

from .authority import AuthorityRegistry
from .evidence_reconciliation import (
    FAILED_RUN, INCONSISTENT, VERIFIED, verify_run)
from .independent_eval_receipt import (
    RECEIPT_FILE, verify_evaluation_receipt)

SCHEMA = "qualified-outcome-v1"
QUALIFICATION_SCHEMA = "mini-agi-v4.0-qualification-decision-v1"
QUALIFICATION_FILE = "QUALIFICATION.json"

#: Verification states — the effective grade of a derived outcome, in
#: ascending order of assurance.
STATE_INVALID = "INVALID"
STATE_INTEGRITY = "INTEGRITY_VERIFIED"
STATE_LOCAL = "LOCALLY_RECOMPUTED"
STATE_INDEPENDENT = "INDEPENDENTLY_VERIFIED"
STATE_QUALIFIED = "QUALIFIED"

#: Deprecated v4.0.3 name — what that release called
#: EVALUATION_VERIFIED was local recomputation; the name now aliases
#: the level the label was supposed to mean.
STATE_EVALUATION = STATE_INDEPENDENT

QUALIFIED = "QUALIFIED"
NOT_QUALIFIED = "NOT_QUALIFIED"

#: States that may supply verified experimental outcomes to the
#: learning controller. Integrity and local recomputation never
#: qualify — an authorized independent evaluator must have signed the
#: evidence binding.
CONTROLLER_ADMISSIBLE = (STATE_INDEPENDENT, STATE_QUALIFIED)

#: Ascending assurance order, for "at least level X" checks.
STATE_RANK = {STATE_INVALID: 0, STATE_INTEGRITY: 1, STATE_LOCAL: 2,
              STATE_INDEPENDENT: 3, STATE_QUALIFIED: 4}

#: The derivation precedence for the run's primary outcome surface.
#: within_holdout (v4.1.0 fresh-unseen-trained-family surface) wins;
#: then the within-family block of validation; then hidden.
_SURFACE_LABELS = {
    "within_holdout": "within_holdout.within_family",
    "validation_within": "validation.within_family",
    "hidden": "hidden",
}


@dataclass(frozen=True)
class QualifiedOutcome:
    """One experiment's outcome, derived from verifiable evidence —
    never from operator-supplied numbers.

    ``verification_state`` is the effective grade:
    INVALID | INTEGRITY_VERIFIED | LOCALLY_RECOMPUTED |
    INDEPENDENTLY_VERIFIED | QUALIFIED.
    ``admissible`` is True only for the two states that may enter
    verified controller history. Numeric fields populate at
    LOCALLY_RECOMPUTED and above — they are rescored measurements, not
    claims; below that they are None, so a failed verification carries
    no numbers a claimed delta could leak through."""

    run_id: str
    intervention: str
    verification_state: str
    evaluation_status: str
    qualification_status: str
    baseline_accuracy: float | None
    adapted_accuracy: float | None
    outcome_delta: float | None
    cross_family_delta: float | None
    retention_regression: float | None
    training_seconds: float
    evaluation_seconds: float
    outcome_surface: str
    evidence_digest: str
    evaluation_receipt_digest: str | None
    evaluation_signer: str
    qualification_digest: str | None
    qualification_signer: str
    model_id: str
    model_revision: str
    score_oracle: str
    seed: int | None
    reasons: tuple[str, ...] = ()
    schema: str = SCHEMA

    @property
    def admissible(self) -> bool:
        """True iff this record may enter verified controller history."""
        return self.verification_state in CONTROLLER_ADMISSIBLE

    @property
    def digest(self) -> str:
        return digest(self.to_dict())

    def to_dict(self) -> dict:
        return asdict(self)


def _num(v) -> float | None:
    """Finite float or None — absent/null/non-finite stay absent."""
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _intervention_of(config: Mapping) -> str:
    """Derive the intervention identity from the committed run config —
    the evidence, not the caller, decides what was run."""
    if config.get("intervention"):
        return str(config["intervention"])
    if config.get("shuffle_labels"):
        return "shuffled_label_control"
    reh = config.get("rehearsal") or {}
    kl = config.get("kl_preservation") or {}
    reh_on = bool(reh.get("fraction")) if isinstance(reh, Mapping) else bool(reh)
    kl_on = bool(kl.get("weight")) if isinstance(kl, Mapping) else bool(kl)
    if reh_on and kl_on:
        return "lora_rehearsal_kl"
    if reh_on:
        return "lora_rehearsal"
    if kl_on:
        return "lora_kl"
    return "fixed_lora"


def _surface(metrics: Mapping) -> tuple[str, float | None, float | None]:
    """Pick the run's primary outcome surface in precedence order.

    Returns (surface, frozen_acc, adapted_acc); both numbers are None
    when no surface carries a complete frozen/adapted pair."""
    who = metrics.get("within_holdout") or {}
    who_wf = who.get("within_family") or {}
    if _num(who_wf.get("frozen")) is not None \
            and _num(who_wf.get("adapted")) is not None:
        return ("within_holdout", _num(who_wf["frozen"]),
                _num(who_wf["adapted"]))
    val = metrics.get("validation") or {}
    wf = val.get("within_family") or {}
    if _num(wf.get("frozen")) is not None \
            and _num(wf.get("adapted")) is not None:
        return ("validation_within", _num(wf["frozen"]),
                _num(wf["adapted"]))
    frozen = metrics.get("frozen") or {}
    adapted = metrics.get("adapted") or {}
    if _num(frozen.get("hidden")) is not None \
            and _num(adapted.get("hidden")) is not None:
        return ("hidden", _num(frozen["hidden"]), _num(adapted["hidden"]))
    return ("none", None, None)


def _cross_family_delta(metrics: Mapping, primary: str) -> float | None:
    """Transfer delta on untrained-family surfaces — validation's
    untrained block first, then hidden (whose families are disjoint
    from the trained set by corpus construction)."""
    uf = (metrics.get("validation") or {}).get("untrained_family") or {}
    if _num(uf.get("frozen")) is not None \
            and _num(uf.get("adapted")) is not None:
        return _num(uf["adapted"]) - _num(uf["frozen"])
    if primary != "hidden":
        f = _num((metrics.get("frozen") or {}).get("hidden"))
        a = _num((metrics.get("adapted") or {}).get("hidden"))
        if f is not None and a is not None:
            return a - f
    return None


def _retention_regression(metrics: Mapping) -> float | None:
    """Fraction of previously-correct behavior destroyed — the
    retention surface's destruction rate first, else the hidden
    capability-destruction ratio."""
    ret = metrics.get("retention") or {}
    if _num(ret.get("destruction_rate")) is not None:
        return _num(ret["destruction_rate"])
    cd = metrics.get("capability_destruction") or {}
    n_fc = _num(cd.get("n_frozen_correct_hidden"))
    n_d = _num(cd.get("n_destroyed"))
    if n_fc is not None and n_d is not None and n_fc > 0:
        return n_d / n_fc
    return None


def _seconds(metrics: Mapping, receipt: Mapping | None) -> tuple[float, float]:
    """(training_seconds, evaluation_seconds) — measured, never
    asserted. Sources: METRICS.training.train_seconds /
    TRAINING_RECEIPT.train_seconds; eval_seconds when the run records
    it. Missing measurements are 0.0, not fabricated."""
    train = _num((metrics.get("training") or {}).get("train_seconds"))
    if train is None and receipt:
        train = _num(receipt.get("train_seconds"))
    ev = _num(metrics.get("eval_seconds"))
    if ev is None:
        ev = _num((metrics.get("timing") or {}).get("eval_seconds"))
    if ev is None:
        ev = _num((receipt or {}).get("eval_seconds"))
    return (train if train is not None else 0.0,
            ev if ev is not None else 0.0)


def verify_qualification_document(
        doc: Mapping, *, verifier: Ed25519Verifier | None,
        authority: AuthorityRegistry | None,
        run_id: str, evidence_digest: str) -> tuple[str, str | None, str]:
    """Check a signed qualification decision against this run.

    Returns ``(qualification_status, qualification_digest, reason)``.
    A qualifying decision must (a) carry the decision schema, (b) decide
    ``QUALIFIED``, (c) bind THIS run_id and THIS evidence digest (a
    qualification for different bytes is not a qualification for this
    run), (d) verify under the caller's verifier, and (e) — when an
    authority registry is supplied — be signed by a key authorized for
    the ``qualification`` role.
    """
    if not isinstance(doc, Mapping):
        return NOT_QUALIFIED, None, "qualification document is not an object"
    if doc.get("schema") != QUALIFICATION_SCHEMA:
        return NOT_QUALIFIED, None, \
            f"qualification schema must be {QUALIFICATION_SCHEMA!r}"
    if doc.get("decision") != "QUALIFIED":
        return NOT_QUALIFIED, None, \
            f"decision is {doc.get('decision')!r}, not QUALIFIED"
    if str(doc.get("run_id")) != str(run_id):
        return NOT_QUALIFIED, None, \
            "qualification run_id does not match the run"
    if str(doc.get("evidence_digest")) != str(evidence_digest):
        return NOT_QUALIFIED, None, \
            "qualification evidence_digest does not match this run's " \
            "committed evidence — a decision over different bytes " \
            "does not qualify this run"
    kid = str(doc.get("signer_key_id") or "")
    sig = str(doc.get("signature_b64") or "")
    if not kid or not sig:
        return NOT_QUALIFIED, None, \
            "qualification decision is unsigned — self-asserted " \
            "success is not qualification"
    if authority is not None and not authority.is_authorized(
            "qualification", kid):
        return NOT_QUALIFIED, None, \
            f"signer {kid} is not authorized for role 'qualification'"
    if verifier is None:
        return NOT_QUALIFIED, None, \
            "signed qualification supplied but no verifier/trust root " \
            "was given — a signature that cannot be checked is not " \
            "verification"
    body = {k: v for k, v in doc.items() if k != "signature_b64"}
    if not verifier.verify(body, SignedEnvelope(kid, sig)):
        return NOT_QUALIFIED, None, "qualification signature invalid"
    return QUALIFIED, digest(dict(doc)), ""


def _integrity_of(run_dir: Path, res: Mapping) -> bool:
    """True iff the commit and inventory are internally consistent —
    the *integrity* layer beneath evaluation."""
    checks = res.get("checks") or {}
    return bool(checks.get("inventory") == "ok"
                and checks.get("commit_status") in ("COMPLETE", "FAILED")
                and (run_dir / "RUN_COMMIT.json").is_file())


def derive_qualified_outcome(run_ref, *, repo_root=None,
                             corpus_rows=None,
                             verifier: Ed25519Verifier | None = None,
                             authority: AuthorityRegistry | None = None,
                             qualification_doc: Mapping | None = None,
                             receipt_doc: Mapping | None = None,
                             security_doc: Mapping | None = None
                             ) -> QualifiedOutcome:
    """Derive an immutable outcome record from verifiable experiment
    evidence.

    ``run_ref`` is a committed run directory (or its RUN_COMMIT.json).
    The reconciler re-verifies integrity, re-scores every committed
    prediction file, and checks identity bindings — that is local
    recomputation. INDEPENDENTLY_VERIFIED additionally requires a
    signed ``independent-evaluation-v1`` receipt (the run's own
    EVALUATION_RECEIPT.json, or ``receipt_doc``) whose binding
    recomputes exactly and whose signer is authorized for the
    ``evaluation`` role under ``authority``. QUALIFIED then requires a
    qualification-authorized signed decision on top of the independent
    evaluation — a qualification cannot stand on local evidence.
    """
    run_dir = Path(run_ref)
    if run_dir.is_file() and run_dir.name == "RUN_COMMIT.json":
        run_dir = run_dir.parent
    run_id = run_dir.name

    res = verify_run(run_dir, repo_root=repo_root,
                     corpus_rows=corpus_rows)
    status = res.get("status")
    integrity_ok = _integrity_of(run_dir, res)
    reasons: list[str] = list(res.get("mismatches") or [])

    commit_path = run_dir / "RUN_COMMIT.json"
    commit = json.loads(commit_path.read_text()) \
        if commit_path.is_file() else {}
    evidence_digest = digest(commit) if commit else ""

    base: dict = dict(
        run_id=str(commit.get("run_id") or run_id),
        intervention="",
        verification_state=STATE_INVALID,
        evaluation_status="NOT_EVALUATED",
        qualification_status=NOT_QUALIFIED,
        baseline_accuracy=None, adapted_accuracy=None,
        outcome_delta=None, cross_family_delta=None,
        retention_regression=None,
        training_seconds=0.0, evaluation_seconds=0.0,
        outcome_surface="none",
        evidence_digest=evidence_digest,
        evaluation_receipt_digest=None, evaluation_signer="",
        qualification_digest=None, qualification_signer="",
        model_id="", model_revision="", score_oracle="",
        seed=None, reasons=())

    if status != VERIFIED:
        base["evaluation_status"] = \
            "FAILED_RUN" if status == FAILED_RUN else (
                "EVALUATION_FAILED" if status == INCONSISTENT
                else "NOT_EVALUATED")
        base["verification_state"] = (STATE_INTEGRITY if integrity_ok
                                      else STATE_INVALID)
        if status == FAILED_RUN:
            reasons = ["run status FAILED — no outcome to qualify",
                       *reasons]
        base["reasons"] = tuple(reasons[:32])
        return QualifiedOutcome(**base)

    # ---- LOCALLY_RECOMPUTED: derive numbers from the rescore ---------
    base["evaluation_status"] = STATE_LOCAL
    base["verification_state"] = STATE_LOCAL
    config_path = run_dir / "CONFIG.json"
    metrics_path = run_dir / "METRICS.json"
    config = json.loads(config_path.read_text())
    metrics = json.loads(metrics_path.read_text())
    receipt = None
    tr_path = run_dir / "TRAINING_RECEIPT.json"
    if tr_path.is_file():
        try:
            receipt = json.loads(tr_path.read_text())
        except (json.JSONDecodeError, UnicodeDecodeError):
            pass

    surface, frozen_acc, adapted_acc = _surface(metrics)
    delta = (adapted_acc - frozen_acc) \
        if (frozen_acc is not None and adapted_acc is not None) else None
    cross = _cross_family_delta(metrics, surface)
    ret = _retention_regression(metrics)
    train_s, eval_s = _seconds(metrics, receipt)

    base.update(
        intervention=_intervention_of(config),
        baseline_accuracy=frozen_acc, adapted_accuracy=adapted_acc,
        outcome_delta=delta, cross_family_delta=cross,
        retention_regression=ret,
        training_seconds=train_s, evaluation_seconds=eval_s,
        outcome_surface=surface,
        model_id=str(config.get("model_id") or ""),
        model_revision=str(config.get("model_revision") or ""),
        score_oracle=str(config.get("score_oracle") or ""),
        seed=config.get("seed"))

    # ---- INDEPENDENTLY_VERIFIED: authorized evaluator receipt --------
    rdoc = receipt_doc
    if rdoc is None:
        rpath = run_dir / RECEIPT_FILE
        if rpath.is_file():
            try:
                rdoc = json.loads(rpath.read_text())
            except (json.JSONDecodeError, UnicodeDecodeError):
                reasons.append("EVALUATION_RECEIPT.json unparseable")
    if rdoc is None:
        reasons.append(
            "no authorized independent evaluation receipt — local "
            "recomputation is not independent verification")
    else:
        rres = verify_evaluation_receipt(
            rdoc, verifier=verifier, authority=authority,
            run_dir=run_dir, repo_root=repo_root,
            corpus_rows=corpus_rows, security_doc=security_doc)
        if rres.get("verified"):
            base["verification_state"] = STATE_INDEPENDENT
            base["evaluation_status"] = STATE_INDEPENDENT
            base["evaluation_receipt_digest"] = digest(dict(rdoc))
            base["evaluation_signer"] = str(
                rdoc.get("evaluator_authority_id") or "")
        else:
            reasons.append("evaluation receipt refused: "
                           + str(rres.get("reason")))

    # ---- QUALIFIED: signed decision over independent evidence --------
    qdoc = qualification_doc
    if qdoc is None:
        qpath = run_dir / QUALIFICATION_FILE
        if qpath.is_file():
            try:
                qdoc = json.loads(qpath.read_text())
            except (json.JSONDecodeError, UnicodeDecodeError):
                reasons.append("QUALIFICATION.json unparseable")
    q_status, q_digest, q_reason = (NOT_QUALIFIED, None,
                                    "no qualification decision evidence")
    if qdoc is not None:
        if base["verification_state"] != STATE_INDEPENDENT:
            q_reason = ("qualification requires an independently "
                        "verified evaluation receipt first — a signed "
                        "decision cannot stand on local evidence")
        else:
            q_status, q_digest, q_reason = verify_qualification_document(
                qdoc, verifier=verifier, authority=authority,
                run_id=base["run_id"], evidence_digest=evidence_digest)
    if q_reason:
        reasons.append(q_reason)
    base["qualification_status"] = q_status
    base["qualification_digest"] = q_digest
    if q_status == QUALIFIED:
        base["qualification_signer"] = str(qdoc.get("signer_key_id"))
        base["verification_state"] = STATE_QUALIFIED
    base["reasons"] = tuple(reasons[:32])
    return QualifiedOutcome(**base)


def qualification_decision(signer, *, run_id: str, evidence_digest: str,
                           decision: str = "QUALIFIED",
                           gates: Mapping | None = None) -> dict:
    """The QUALIFIER-side constructor: produce a signed decision
    document binding ``run_id`` + ``evidence_digest``. Lives here so the
    signing layout and the verification layout can never silently
    diverge."""
    body = {"schema": QUALIFICATION_SCHEMA, "run_id": str(run_id),
            "evidence_digest": str(evidence_digest),
            "decision": str(decision),
            "gates": dict(gates or {}),
            "signer_key_id": signer.key_id}
    env = signer.sign(body)
    return {**body, "signature_b64": env.signature_b64}
