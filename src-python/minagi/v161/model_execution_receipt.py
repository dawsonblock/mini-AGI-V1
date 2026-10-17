"""Evaluator-owned model-execution receipts (v4.0.3-r2 / Phase 2).

The v4.0.3-r1 receipt closes the *measurement* boundary: an authorized
evaluator recomputes the researcher-submitted predictions. Diagnostics
9 identified the remaining gap — the evaluator authenticated evidence
the research process GENERATED. Nothing established that the claimed
model and adapter actually produced those outputs.

``model-execution-receipt-v1`` closes that gap: the evaluator loads the
candidate artifacts itself, runs inference on evaluator-controlled
tasks, scores the outputs, and signs a canonical record binding the
whole execution chain:

    execution_id                this execution's identity
    model_id                    base model repository identity
    model_revision              the frozen base revision actually loaded
    tokenizer_digest            tokenizer identity actually loaded
    adapter_digest              exact adapter artifact bytes loaded
                                (null only for a declared frozen
                                baseline execution)
    dataset_digest              commitment to the evaluator's task set
    generation_config_digest    the inference configuration used
    scoring_version             the scorer identity that measured the
                                outputs
    execution_environment_digest the environment the inference ran in
    outputs_digest              the generated predictions manifest
    scores_digest               the independently computed scores
    execution_mode              real_model | frozen_baseline | simulated
    evaluator_authority_id      the signing identity
    execution_status            "EXECUTION_VERIFIED"
    signature                   Ed25519 over the canonical body

Two claims are deliberately kept SEPARATE:

* Execution provenance (this receipt): the model+adapter produced
  these outputs under these conditions. Proved by evaluator-owned
  loading and inference.
* Training provenance (TRAINING_RECEIPT.json, source revision, config,
  artifact lineage): the adapter was trained as claimed. Verified
  separately — never folded into one verification flag.

SIMULATED executions exist for unit tests: a receipt whose
``execution_mode`` is ``simulated`` is a valid signature over mock
outputs and is NEVER accepted as real-model confirmation
(``require_real`` is the default).

Custody note (same limitation as evaluation receipts): the signature
proves authorization by an ``evaluation``-role key; separation of the
private key from the research agent is operational, not provable here.
"""
from __future__ import annotations

import json
import platform
import sys
from pathlib import Path
from typing import Mapping

from egai.common.canonical import digest, validate_digest
from egai.common.crypto import Ed25519Verifier, SignedEnvelope

from .authority import AuthorityRegistry
from .evidence_reconciliation import (
    adapter_artifact_digest_of, sha256_file)

EXECUTION_SCHEMA = "model-execution-receipt-v1"
EXECUTION_FILE = "EXECUTION_RECEIPT.json"
EXECUTION_STATUS = "EXECUTION_VERIFIED"
EXECUTION_PREDICTIONS_FILE = "EVALUATOR_PREDICTIONS.jsonl"
EXECUTION_SCORES_FILE = "EVALUATOR_SCORES.json"
EVALUATOR_ROLE = "evaluation"

MODE_REAL = "real_model"
MODE_BASELINE = "frozen_baseline"
MODE_SIMULATED = "simulated"
EXECUTION_MODES = (MODE_REAL, MODE_BASELINE, MODE_SIMULATED)

#: The evidence-identity fields every execution receipt must carry.
EXECUTION_BINDING_FIELDS = (
    "execution_id",
    "model_id",
    "model_revision",
    "tokenizer_digest",
    "adapter_digest",
    "dataset_digest",
    "generation_config_digest",
    "scoring_version",
    "execution_environment_digest",
    "outputs_digest",
    "scores_digest",
    "execution_mode",
)


def dataset_commitment(tasks) -> str:
    """Commit to the evaluator's task set — id + the prompt the model
    consumed + the expected answer. For evaluator-custody sets the
    ANSWERS stay private: only this commitment is bound, not the
    plaintext."""
    return digest([
        {"id": str(t["id"]),
         "prompt_sha256": str(t.get("prompt_sha256") or ""),
         "expected_sha256": t.get("expected_sha256")}
        for t in tasks])


def executor_environment() -> dict:
    """The execution environment attested in the receipt — recorded for
    provenance, not recomputable."""
    env = {"python": sys.version.split()[0],
           "platform": platform.platform()}
    try:
        import torch
        env["torch"] = torch.__version__
        if torch.cuda.is_available():
            env["device"] = "cuda"
        elif getattr(torch.backends, "mps", None) \
                and torch.backends.mps.is_available():
            env["device"] = "mps"
        else:
            env["device"] = "cpu"
    except ImportError:
        pass
    for mod in ("transformers", "peft"):
        try:
            m = __import__(mod)
            env[mod] = getattr(m, "__version__", "?")
        except ImportError:
            pass
    return env


def execution_binding(*, adapter_dir=None, tasks=None,
                      predictions_path=None, scores=None,
                      generation_config=None,
                      model_id=None, model_revision=None,
                      tokenizer_digest=None, environment=None,
                      execution_id=None, execution_mode=None,
                      scoring_version=None) -> dict:
    """Recompute the execution evidence binding from the artifacts
    themselves — the same function runs at issue and verify time; any
    disagreement is tampering."""
    adapter_digest = None
    if adapter_dir is not None:
        adapter_digest = adapter_artifact_digest_of(Path(adapter_dir))
    outputs_digest = None
    if predictions_path is not None and \
            Path(predictions_path).is_file():
        outputs_digest = "sha256:" + sha256_file(
            Path(predictions_path))
    return {
        "execution_id": execution_id,
        "model_id": model_id,
        "model_revision": model_revision,
        "tokenizer_digest": tokenizer_digest,
        "adapter_digest": adapter_digest,
        "dataset_digest":
            dataset_commitment(tasks) if tasks is not None else None,
        "generation_config_digest":
            digest(generation_config)
            if generation_config is not None else None,
        "scoring_version": scoring_version,
        "execution_environment_digest":
            digest(environment) if environment is not None else None,
        "outputs_digest": outputs_digest,
        "scores_digest": digest(scores) if scores is not None else None,
        "execution_mode": execution_mode,
    }


def issue_execution_receipt(signer, *, execution_id: str,
                            execution_mode: str,
                            model_id: str, model_revision: str,
                            tokenizer_digest: str,
                            adapter_dir=None, tasks=None,
                            predictions_path=None, scores=None,
                            generation_config=None,
                            scoring_version=None,
                            environment=None) -> dict:
    """Evaluator-side constructor: bind the artifacts actually loaded
    and sign the canonical record. ``adapter_dir`` is None only for a
    declared frozen-baseline execution."""
    if execution_mode not in EXECUTION_MODES:
        raise ValueError(
            f"execution_mode must be one of {EXECUTION_MODES}")
    if execution_mode == MODE_REAL and adapter_dir is None:
        raise ValueError(
            "a real-model execution must bind the adapter bytes it "
            "loaded — adapter-free real claims require "
            "execution_mode='frozen_baseline'")
    if execution_mode == MODE_BASELINE and adapter_dir is not None:
        raise ValueError(
            "a frozen_baseline execution cannot bind adapter bytes — "
            "a baseline attests the bare foundation model only")
    environment = environment or executor_environment()
    binding = execution_binding(
        adapter_dir=adapter_dir, tasks=tasks,
        predictions_path=predictions_path, scores=scores,
        generation_config=generation_config,
        model_id=model_id, model_revision=model_revision,
        tokenizer_digest=tokenizer_digest, environment=environment,
        execution_id=execution_id, execution_mode=execution_mode,
        scoring_version=scoring_version)
    body = {"schema": EXECUTION_SCHEMA, **binding,
            "execution_environment": dict(environment),
            "evaluator_authority_id": signer.key_id,
            "execution_status": EXECUTION_STATUS}
    env = signer.sign(body)
    return {**body, "signature": env.signature_b64}


def verify_execution_receipt(doc, *, verifier: Ed25519Verifier | None,
                             authority: AuthorityRegistry | None,
                             adapter_dir=None, tasks=None,
                             predictions_path=None, scores=None,
                             generation_config=None,
                             expected_model=None,
                             allow_detached=False,
                             require_real=True) -> dict:
    """Verify a model-execution receipt.

    ``verified`` is True only when ALL of the following hold:

    * schema ``model-execution-receipt-v1``, every binding field
      present, ``execution_status`` == EXECUTION_VERIFIED;
    * ``execution_mode`` is not ``simulated`` (unless ``require_real``
      is disabled — simulated receipts exist for tests only) and a
      non-baseline execution binds a non-null ``adapter_digest``;
    * ``evaluator_authority_id`` is authorized for the ``evaluation``
      role under ``authority``;
    * the Ed25519 signature verifies over the canonical body;
    * every supplied artifact recomputes to the bound digest — adapter
      bytes, task set, generation configuration, generated predictions
      and scores. An artifact that cannot be supplied cannot be
      binding-checked; with ``allow_detached`` the caller accepts
      signature+role verification only (never suitable for trusted
      import).
    * when ``expected_model`` is given ({model_id, model_revision}),
      the attested identity must match it exactly.
    """
    out = {"verified": False, "binding_checked": False,
           "signer_key_id": "", "reason": "", "mismatches": []}
    fail = out["mismatches"].append

    def done(reason):
        out["reason"] = reason
        return out

    if not isinstance(doc, Mapping):
        return done("execution receipt is not a JSON object")
    if doc.get("schema") != EXECUTION_SCHEMA:
        return done(f"execution receipt schema must be "
                    f"{EXECUTION_SCHEMA!r}")
    missing = [f for f in EXECUTION_BINDING_FIELDS if f not in doc]
    if missing:
        return done(f"execution receipt lacks binding fields {missing}")
    if doc.get("execution_status") != EXECUTION_STATUS:
        return done("execution_status is "
                    f"{doc.get('execution_status')!r}, not "
                    f"{EXECUTION_STATUS!r}")
    mode = doc.get("execution_mode")
    if mode not in EXECUTION_MODES:
        return done(f"execution_mode {mode!r} is not one of "
                    f"{EXECUTION_MODES}")
    if require_real and mode == MODE_SIMULATED:
        return done("simulated execution is not real-model "
                    "confirmation — a mock-model receipt cannot "
                    "certify an artifact")
    if mode == MODE_REAL and not doc.get("adapter_digest"):
        return done("real-model execution binds no adapter bytes — "
                    "adapter-free claims require the frozen_baseline "
                    "mode")
    if mode == MODE_BASELINE and doc.get("adapter_digest") is not None:
        return done("frozen_baseline execution binds an adapter — a "
                    "baseline attests the bare model only")
    kid = str(doc.get("evaluator_authority_id") or "")
    sig = str(doc.get("signature") or "")
    out["signer_key_id"] = kid
    if not kid or not sig:
        return done("execution receipt is unsigned")
    for f in EXECUTION_BINDING_FIELDS:
        v = doc.get(f)
        if v is not None and f.endswith("_digest"):
            try:
                validate_digest(v)
            except ValueError:
                return done(f"{f} is not a sha256 digest: {v!r}")
    if expected_model is not None:
        for k in ("model_id", "model_revision"):
            if str(doc.get(k)) != str(expected_model.get(k)):
                return done(
                    f"attested {k} {doc.get(k)!r} != expected "
                    f"{expected_model.get(k)!r}")
    if authority is None:
        return done("no authority registry supplied — the signer's "
                    "evaluation role cannot be established")
    if not authority.is_authorized(EVALUATOR_ROLE, kid):
        rec = authority.record_for(kid)
        why = ("unregistered signing identity" if rec is None else
               f"key registered for role {rec.role!r}, not "
               f"{EVALUATOR_ROLE!r}")
        return done(f"execution receipt signer {kid} is not authorized "
                    f"for role '{EVALUATOR_ROLE}': {why}")
    if verifier is None:
        return done("execution receipt signature cannot be checked — "
                    "no verifier covering the signer key")
    body = {k: v for k, v in doc.items() if k != "signature"}
    if not verifier.verify(body, SignedEnvelope(kid, sig)):
        return done("execution receipt signature invalid")

    binding_supplied = any(
        a is not None for a in
        (adapter_dir, tasks, scores, generation_config)) \
        or (predictions_path is not None
            and Path(predictions_path).is_file())
    if not binding_supplied:
        if allow_detached:
            out["verified"] = True
            return done("detached signature+role verification only — "
                        "execution artifacts were NOT rebound")
        return done("execution receipt cannot be bound to artifacts — "
                    "nothing to recompute the binding against")

    actual = execution_binding(
        adapter_dir=adapter_dir, tasks=tasks,
        predictions_path=predictions_path, scores=scores,
        generation_config=generation_config,
        model_id=doc["model_id"], model_revision=doc["model_revision"],
        tokenizer_digest=doc["tokenizer_digest"],
        environment=doc.get("execution_environment"),
        execution_id=doc["execution_id"], execution_mode=mode,
        scoring_version=doc["scoring_version"])
    for f in EXECUTION_BINDING_FIELDS:
        want = doc.get(f)
        if f in ("model_id", "model_revision", "tokenizer_digest",
                 "execution_id", "execution_mode", "scoring_version"):
            # attested identities are compared, not recomputed
            if want != actual.get(f):
                fail(f"{f}: receipt {want!r} != bound "
                     f"{actual.get(f)!r}")
        elif f == "adapter_digest":
            if adapter_dir is not None and want != actual.get(f):
                fail(f"{f}: receipt {want!r} != adapter bytes "
                     f"{actual.get(f)!r}")
            elif adapter_dir is None and want is not None:
                # caller could not supply the adapter — the bound
                # digest is unverified here
                if not allow_detached:
                    fail(f"{f}: adapter unavailable to recompute "
                         f"bound digest {want!r}")
        elif f == "dataset_digest":
            if tasks is not None and want != actual.get(f):
                fail(f"{f}: receipt {want!r} != task set "
                     f"{actual.get(f)!r}")
        elif f == "generation_config_digest":
            if generation_config is not None \
                    and want != actual.get(f):
                fail(f"{f}: receipt {want!r} != generation config "
                     f"{actual.get(f)!r}")
        elif f == "outputs_digest":
            if predictions_path is not None and \
                    Path(predictions_path).is_file() \
                    and want != actual.get(f):
                fail(f"{f}: receipt {want!r} != predictions "
                     f"{actual.get(f)!r}")
        elif f == "scores_digest":
            if scores is not None and want != actual.get(f):
                fail(f"{f}: receipt {want!r} != scores "
                     f"{actual.get(f)!r}")
        elif f == "execution_environment_digest":
            if want != actual.get(f):
                fail(f"{f}: receipt {want!r} != environment "
                     f"{actual.get(f)!r}")
    if out["mismatches"]:
        return done("execution receipt does not bind these artifacts "
                    "— the evaluator executed different bytes")
    out["binding_checked"] = True
    out["verified"] = True
    return done("")
