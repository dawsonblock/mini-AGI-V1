"""Trusted evidence resolver (v4.0.3-r2 / Phase 3).

Diagnostics 9's evidence-binding requirement: final Campaign 4A
evaluations must require valid bindings for every identity in the
evidence chain — not just "a receipt exists". This module is the
resolver: it loads the referenced artifacts itself, validates each
mandatory identity, and refuses the set that fails closed.

Mandatory identities for a capability-acquisition evaluation:

    base model           CONFIG.model_id + model_revision (non-null)
    tokenizer            generation config's tokenizer binding
    adapter              adapter artifact bytes — or a DECLARED
                         adapter-free experiment type (frozen baseline)
    dataset              corpus identity bound by the receipt
    scorer               an authorized scoring version
    protocol             the run's committed protocol/config identity
    generation config    prompt/output contract digests
    predictions          committed prediction files
    security evidence    governed sidecar — when the corpus carries
                         security rows
    retention evidence   paired frozen/adapted retention files — when
                         the corpus carries retention rows
    receipts             the authorized evaluation receipt; and, when
                         ``require_execution`` applies, a bound
                         ``model-execution-receipt-v1``

Null is not permitted for required fields; the only declared exception
is an adapter-free experiment type, which must be explicit — absence
of adapter bytes cannot silently reclassify a run as a baseline.

The resolver is deliberately a *resolver*, not a scorer: it returns
which identities resolve, which fail, and the effective verification
state — measurement interpretation stays in the matrix analyzer.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

from egai.common.canonical import digest, validate_digest

from .authority import AuthorityRegistry
from .evidence_reconciliation import (
    VERIFIED, adapter_artifact_digest_of, verify_run)
from .independent_eval_receipt import (
    RECEIPT_FILE, verify_evaluation_receipt)
from .model_execution_receipt import (
    EXECUTION_FILE, MODE_SIMULATED, verify_execution_receipt)
from .qualified_outcome import (
    STATE_INVALID, STATE_RANK, STATE_INTEGRITY, STATE_LOCAL,
    STATE_INDEPENDENT, STATE_QUALIFIED)

SCORING_VERSIONS_SUPPORTED = ("exact-match-v2", "exact_match_v2")

#: Experiment types whose declared form permits a null adapter
#: identity — everything else must bind adapter bytes.
ADAPTER_FREE_EXPERIMENT_TYPES = ("frozen_baseline", "foundation_only")


def _load_json(path: Path):
    if not path.is_file():
        return None
    try:
        doc = json.loads(path.read_text())
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None
    return doc if isinstance(doc, dict) else None


def _identity_ok(value) -> bool:
    return bool(value)


def resolve_identities(run_dir, *, corpus_rows=None,
                       security_doc=None,
                       require_execution=False) -> dict:
    """Resolve every mandatory evidence identity from the run's own
    bytes — the caller's claims are not consulted.

    Returns ``{identity: {"status": "resolved"|"missing"|"unbound"|
    "declared_absent", "detail": str}}`` plus the aggregate
    ``all_resolved``. ``declared_absent`` is a valid resolution ONLY
    for identities the experiment type explicitly excused."""
    run_dir = Path(run_dir)
    config = _load_json(run_dir / "CONFIG.json") or {}
    metrics = _load_json(run_dir / "METRICS.json") or {}
    out: dict[str, dict] = {}

    def rec(name, status, detail=""):
        out[name] = {"status": status, "detail": detail}

    # ---- base model -------------------------------------------------
    mid, mrev = config.get("model_id"), config.get("model_revision")
    rec("base_model", "resolved" if mid and mrev else "missing",
        f"{mid}@{mrev}" if mid and mrev else
        "model_id/model_revision absent from CONFIG.json")

    # ---- tokenizer --------------------------------------------------
    tok_d = (config.get("prompt_contract") or {}).get(
        "tokenizer_digest") or config.get("tokenizer_digest")
    rec("tokenizer", "resolved" if tok_d else "missing",
        tok_d or "no tokenizer binding in generation config")

    # ---- adapter ----------------------------------------------------
    exp_type = str(config.get("experiment_type") or "")
    adapter_free = exp_type in ADAPTER_FREE_EXPERIMENT_TYPES
    adir = run_dir / "adapter"
    if adir.is_dir():
        try:
            rec("adapter", "resolved", adapter_artifact_digest_of(adir))
        except ValueError as exc:
            rec("adapter", "unbound", str(exc))
    elif adapter_free:
        rec("adapter", "declared_absent",
            f"experiment_type={exp_type!r} is adapter-free")
    else:
        rec("adapter", "missing",
            "no adapter/ directory and no adapter-free experiment type "
            "declared")

    # ---- dataset ----------------------------------------------------
    if corpus_rows is not None:
        declared = {str(r["id"]) for r in corpus_rows}
        rec("dataset", "resolved" if declared else "missing",
            f"{len(declared)} corpus rows")
    else:
        rec("dataset", "missing", "no corpus context supplied")

    # ---- scorer -----------------------------------------------------
    so = str(config.get("score_oracle") or "")
    rec("scorer",
        "resolved" if so in SCORING_VERSIONS_SUPPORTED else "unbound",
        so or "no score_oracle declared")

    # ---- protocol ---------------------------------------------------
    proto = (config.get("config") or config.get("protocol")
             or config.get("config_digest"))
    rec("protocol", "resolved" if proto else "missing",
        str(proto or "no protocol identity in CONFIG.json"))

    # ---- generation config ------------------------------------------
    pc = config.get("prompt_contract") or {}
    oc_d = config.get("output_contract_digest") or pc.get(
        "output_contract_digest")
    pc_d = pc.get("digest") or config.get("prompt_contract_digest")
    if pc_d and oc_d:
        rec("generation_config", "resolved",
            f"{pc_d}/{oc_d}")
    else:
        rec("generation_config", "missing",
            "prompt/output contract digests not both bound")

    # ---- predictions ------------------------------------------------
    preds = sorted(run_dir.glob("PREDICTIONS-*.jsonl"))
    rec("predictions", "resolved" if preds else "missing",
        f"{len(preds)} committed prediction files")

    # ---- security / retention (corpus-conditioned) ------------------
    splits = {str(r.get("split") or "") for r in (corpus_rows or [])}
    sec_rows = any(s.startswith("security") for s in splits)
    ret_rows = "retention" in splits or any(
        str(r.get("family") or "") == "retention" for r in
        (corpus_rows or []))
    if sec_rows:
        rec("security", "resolved" if security_doc else "missing",
            "governed sidecar" if security_doc else
            "corpus carries security rows but no governed sidecar "
            "supplied")
    else:
        rec("security", "declared_absent", "no security rows in corpus")
    f_ret = run_dir / "PREDICTIONS-FROZEN-RETENTION.jsonl"
    a_ret = run_dir / "PREDICTIONS-ADAPTED-RETENTION.jsonl"
    if ret_rows:
        ok = f_ret.is_file() and a_ret.is_file()
        rec("retention", "resolved" if ok else "missing",
            "paired frozen/adapted retention files" if ok else
            "corpus declares retention rows but the paired prediction "
            "files are absent")
    else:
        rec("retention", "declared_absent", "no retention rows in "
            "corpus")

    # ---- receipts ----------------------------------------------------
    rec("evaluation_receipt",
        "resolved" if (run_dir / RECEIPT_FILE).is_file() else "missing",
        RECEIPT_FILE)
    if require_execution:
        rec("execution_receipt",
            "resolved" if (run_dir / EXECUTION_FILE).is_file()
            else "missing", EXECUTION_FILE)
    else:
        rec("execution_receipt", "declared_absent",
            "execution provenance not required by policy")

    blocking = [n for n, v in out.items()
                if v["status"] in ("missing", "unbound")]
    return {"identities": out, "all_resolved": not blocking,
            "blocking": blocking}


def resolve_trusted_evidence(run_dir, *, repo_root=None,
                             corpus_rows=None,
                             verifier=None,
                             authority: AuthorityRegistry | None = None,
                             security_doc=None,
                             require_execution=False,
                             expected_model=None) -> dict:
    """Load, validate and resolve every referenced artifact.

    Rejects (non-exhaustive): missing artifacts, hash mismatches,
    incorrect model revisions, unauthorized receipts, inconsistent
    dataset identities, missing mandatory security evidence, receipts
    bound to different candidates, unsupported scoring versions.

    Returns ``{"resolved": bool, "state": <effective level>,
    "identities": {...}, "receipts": {...}, "reasons": [...]}``.
    ``state`` never exceeds what the strongest bound evidence supports:
    integrity → INTEGRITY_VERIFIED; +reconciled → LOCALLY_RECOMPUTED;
    +authorized bound receipt → INDEPENDENTLY_VERIFIED. A self-authored
    run tops out at LOCALLY_RECOMPUTED — always.
    """
    run_dir = Path(run_dir)
    reasons: list[str] = []

    ident = resolve_identities(
        run_dir, corpus_rows=corpus_rows, security_doc=security_doc,
        require_execution=require_execution)
    for name in ident["blocking"]:
        reasons.append(f"mandatory evidence identity {name!r}: "
                       f"{ident['identities'][name]['detail']}")

    verification = verify_run(run_dir, repo_root=repo_root,
                              corpus_rows=corpus_rows)
    state = STATE_LOCAL if verification.get("status") == VERIFIED \
        else STATE_INTEGRITY
    if verification.get("status") != VERIFIED:
        reasons.extend(str(m) for m in
                       verification.get("mismatches") or [])

    receipts: dict[str, dict] = {}

    # ---- evaluation receipt -----------------------------------------
    rdoc = _load_json(run_dir / RECEIPT_FILE)
    if rdoc is not None:
        rres = verify_evaluation_receipt(
            rdoc, verifier=verifier, authority=authority,
            run_dir=run_dir, repo_root=repo_root,
            corpus_rows=corpus_rows, security_doc=security_doc)
        receipts["evaluation"] = rres
        if rres.get("verified"):
            state = STATE_INDEPENDENT
        else:
            reasons.append("evaluation receipt refused: "
                           + str(rres.get("reason")))
            # a receipt claiming to bind this run that DOESN'T is
            # forged-binding evidence — inconsistent, not merely
            # integrity-grade
            state = STATE_INVALID
    elif state == STATE_LOCAL:
        reasons.append("no evaluation receipt — local recomputation "
                       "is not independent verification")

    # ---- execution receipt -------------------------------------------
    edoc = _load_json(run_dir / EXECUTION_FILE)
    adir = run_dir / "adapter" if (run_dir / "adapter").is_dir() else None
    if edoc is not None:
        config = _load_json(run_dir / "CONFIG.json") or {}
        em = expected_model or {
            "model_id": config.get("model_id"),
            "model_revision": config.get("model_revision")}
        eres = verify_execution_receipt(
            edoc, verifier=verifier, authority=authority,
            adapter_dir=adir, expected_model=em,
            require_real=require_execution)
        receipts["execution"] = eres
        if not eres.get("verified"):
            reasons.append("execution receipt refused: "
                           + str(eres.get("reason")))
            state = STATE_INVALID
        elif edoc.get("execution_mode") == MODE_SIMULATED:
            reasons.append("execution receipt is simulated — not "
                           "real-model provenance")
        else:
            # cross-bind: the executed adapter must be the evaluated
            # candidate — receipts over different bytes never merge
            cand = None
            try:
                cand = adapter_artifact_digest_of(adir) \
                    if adir else None
            except ValueError:
                pass
            if cand is not None \
                    and edoc.get("adapter_digest") != cand:
                reasons.append("execution receipt binds adapter "
                               f"{edoc.get('adapter_digest')} != "
                               f"candidate {cand}")
                state = STATE_INVALID
    elif require_execution:
        reasons.append("no execution receipt — evaluator-owned model "
                       "execution is required evidence")

    resolved = (not ident["blocking"]
                and verification.get("status") == VERIFIED
                and all(v.get("verified")
                        for v in receipts.values())
                and not (require_execution and edoc is None))
    return {"resolved": resolved, "state": state,
            "identities": ident["identities"],
            "receipts": receipts, "reasons": reasons}
