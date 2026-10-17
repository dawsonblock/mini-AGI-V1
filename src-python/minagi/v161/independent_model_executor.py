"""Evaluator-owned model executor (v4.0.3-r2 / Phase 2).

Diagnostics 9's second blocker: the evaluator could sign submitted
predictions without the claimed model ever running. This module is the
evaluator's executor — it receives frozen artifact references, verifies
their identities, loads the foundation model and the adapter in the
evaluator's own process, runs inference on evaluator-controlled tasks,
scores the outputs itself, and issues a signed
``model-execution-receipt-v1``.

Researcher-submitted predictions are never authoritative inference:
the evaluator executes the artifacts itself. What this proves:

* the exact adapter bytes load against the declared base revision;
* the claimed model+tokenizer identity actually produces outputs;
* the measured scores derive from THOSE outputs, not supplied ones.

What it does NOT prove: that the adapter was trained as claimed —
training provenance is a separate verification lane
(TRAINING_RECEIPT.json, source revision, config, lineage). The two
claims are intentionally distinct.

Mock loaders exist for unit tests ONLY: an execution through an
injected ``model_loader`` is recorded ``execution_mode='simulated'``
and can never satisfy real-model confirmation.
"""
from __future__ import annotations

import hashlib
import json
import uuid
from pathlib import Path

from egai.common.crypto import Ed25519Signer

from .evidence_reconciliation import adapter_artifact_digest_of
from .model_execution_receipt import (
    EXECUTION_PREDICTIONS_FILE, EXECUTION_SCORES_FILE, MODE_BASELINE,
    MODE_REAL, MODE_SIMULATED, executor_environment,
    issue_execution_receipt)


def _generate(model, tokenizer, contract, prompt: str,
              max_new_tokens: int) -> str:
    """One contract-bound generation — the same encoding the run
    pipeline uses (encode_generation), never a raw prompt."""
    import torch
    from .prompt_contract import encode_generation
    ids = encode_generation(tokenizer, prompt, contract)
    device = next(model.parameters()).device
    batch = {"input_ids": torch.tensor([ids], device=device),
             "attention_mask": torch.ones((1, len(ids)),
                                          dtype=torch.long,
                                          device=device)}
    with torch.inference_mode():
        out = model.generate(
            **batch, max_new_tokens=int(max_new_tokens),
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(out[0, len(ids):],
                            skip_special_tokens=True).strip()


def _score_fn(scoring_version: str):
    """Map a declared scoring identity to its oracle — the executor
    supports the campaign's authorized scorers only."""
    from .evaluators import exact_match_strict
    known = {"exact-match-v2": exact_match_strict,
             "exact_match_v2": exact_match_strict}
    if scoring_version not in known:
        raise ValueError(
            f"unsupported scoring_version {scoring_version!r} — the "
            f"executor scores with {sorted(known)} only")
    return known[scoring_version]


def execute_evaluation(signer: Ed25519Signer, *,
                       model_id: str, model_revision: str,
                       tasks: list[dict],
                       adapter_dir=None,
                       generation_config: dict | None = None,
                       scoring_version: str = "exact-match-v2",
                       model_loader=None,
                       execution_id: str | None = None,
                       out_dir=None) -> dict:
    """Run evaluator-owned inference and issue a signed execution
    receipt.

    ``tasks`` are evaluator-controlled rows: ``{id, prompt, expected}``
    — the evaluator's own task set, not the run's predictions.
    ``adapter_dir=None`` executes the frozen baseline (declared via
    ``execution_mode='frozen_baseline'``).

    ``model_loader`` is a TEST seam: ``loader(spec) -> (model,
    tokenizer, contract)``. Supplying one marks the execution
    ``simulated`` — a mock-model receipt never satisfies real-model
    confirmation. The real path loads through
    ``hf_runtime.load_causal_lm`` under ``purpose='research'``, which is
    the evaluator's recorded opt-in to open raw artifacts.

    Returns ``{"receipt": ..., "predictions_path": ..., "scores": ...,
    "outputs": {...}}``.
    """
    if not tasks:
        raise ValueError("no evaluator tasks supplied")
    generation_config = dict(generation_config or {})
    max_new = int(generation_config.get("max_new_tokens", 32))
    score = _score_fn(scoring_version)
    execution_id = execution_id or f"exec-{uuid.uuid4().hex[:16]}"

    model = tokenizer = contract = None
    tokenizer_digest = None
    if model_loader is not None:
        model, tokenizer, contract = model_loader(
            {"model_id": model_id, "revision": model_revision,
             "adapter_dir": (str(adapter_dir)
                             if adapter_dir else None)})
        # an injected loader is a TEST seam — the execution is always
        # simulated, baseline or not, and never real-model evidence
        mode = MODE_SIMULATED
        tokenizer_digest = "sha256:" + hashlib.sha256(
            b"simulated-tokenizer").hexdigest()
    else:
        from minagi.platforms.cuda.hf_runtime import (
            HFLoadSpec, load_causal_lm, load_tokenizer)
        from .prompt_contract import (
            build_prompt_contract, output_contract_from_config)
        spec = HFLoadSpec(model_id=model_id,
                          revision=model_revision)
        tokenizer = load_tokenizer(spec)
        output_contract = output_contract_from_config(
            generation_config, max_new_tokens=max_new)
        contract = build_prompt_contract(tokenizer, output_contract)
        model = load_causal_lm(
            spec,
            adapter_path=str(adapter_dir) if adapter_dir else None,
            purpose="research")
        mode = MODE_BASELINE if adapter_dir is None else MODE_REAL
        tokenizer_digest = contract.tokenizer_digest

    # Adapter identity: the bytes must resolve to a digest AND — when
    # the adapter declares a base — be compatible with the base model.
    if adapter_dir is not None:
        adapter_dir = Path(adapter_dir)
        adapter_artifact_digest_of(adapter_dir)   # raises if malformed
        acfg = adapter_dir / "adapter_config.json"
        if acfg.is_file():
            base = str(json.loads(acfg.read_text())
                       .get("base_model_name_or_path") or "")
            if base and base != str(model_id):
                raise ValueError(
                    f"adapter declares base {base!r} but the run is "
                    f"for {model_id!r} — incompatible artifacts")

    outputs: dict[str, str] = {}
    predictions: list[dict] = []
    for t in tasks:
        out = _generate(model, tokenizer, contract,
                        str(t["prompt"]), max_new)
        rid = str(t["id"])
        outputs[rid] = out
        predictions.append({
            "id": rid, "split": t.get("split", "evaluator"),
            "prompt_sha256": hashlib.sha256(
                str(t["prompt"]).encode()).hexdigest(),
            "output": out})

    scores = {str(t["id"]): float(score(outputs[str(t["id"])],
                                        str(t.get("expected", ""))))
              for t in tasks}

    # tasks bound into the receipt carry prompt+expected commitments —
    # for evaluator-custody sets the plaintext never leaves the
    # evaluator; the digest is the commitment.
    bound_tasks = [
        {"id": str(t["id"]),
         "prompt_sha256": hashlib.sha256(
             str(t["prompt"]).encode()).hexdigest(),
         "expected_sha256": hashlib.sha256(
             str(t.get("expected", "")).encode()).hexdigest()}
        for t in tasks]

    predictions_path = None
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        predictions_path = out_dir / EXECUTION_PREDICTIONS_FILE
        predictions_path.write_text(
            "".join(json.dumps(r, sort_keys=True) + "\n"
                    for r in predictions))
        (out_dir / EXECUTION_SCORES_FILE).write_text(
            json.dumps(scores, indent=2, sort_keys=True) + "\n")

    receipt = issue_execution_receipt(
        signer,
        execution_id=execution_id, execution_mode=mode,
        model_id=model_id, model_revision=model_revision,
        tokenizer_digest=tokenizer_digest,
        adapter_dir=adapter_dir, tasks=bound_tasks,
        predictions_path=predictions_path, scores=scores,
        generation_config=generation_config,
        scoring_version=scoring_version)
    return {"receipt": receipt, "predictions_path": predictions_path,
            "scores": scores, "outputs": outputs,
            "execution_mode": mode,
            "n_tasks": len(tasks),
            "n_correct": int(sum(scores.values()))}
