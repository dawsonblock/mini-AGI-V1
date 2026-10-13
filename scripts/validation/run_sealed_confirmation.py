#!/usr/bin/env python3
"""Campaign 3C — evaluator-sealed confirmation executor (3B Steps 20-22).

Runs a frozen candidate's arms against the evaluator-held confirmation
set and issues an independent QUALIFIED / REFUSED decision. The research
worker never sees the sealed rows or their answers: the evaluator runs
this tool in its own custody, with the evaluation and qualification
keys that never touched the training worker.

Workflow (Step 21): freeze submission -> verify eligibility -> load
sealed data -> execute candidates -> score -> compute qualification ->
sign -> preserve evidence. Every attempt is appended to an append-only
confirmation ledger (Step 22), so a failed candidate cannot be silently
re-tuned against the same sealed labels.

Usage (evaluator host):

  python3 scripts/validation/run_sealed_confirmation.py \
      --storage <campaign storage> --campaign-id <cid> --root <repo> \
      --sealed <evaluator-held confirmation.jsonl> \
      --submission <submission.json> \
      --output <CONFIRMATION_RECORD.json> \
      [--ledger <path>]

The submission binds what is being confirmed: campaign, plan digest,
model/tokenizer identity, the frozen output contract and per-arm
adapter paths. The tool refuses an incomplete or non-qualified research
record, a sealed set whose manifest digest does not match the plan's
bound holdout, a sealed design without the required blocks and family
coverage, and any candidate identity that disagrees with the signed
plan.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src-python"))

from egai.common.canonical import digest, sha256_bytes  # noqa: E402
from egai.common.crypto import Ed25519Signer  # noqa: E402
from minagi.v161.authority import AuthorityRegistry  # noqa: E402
from minagi.v161.dataset_manifest import (DatasetMember,  # noqa: E402
                                          DatasetMembershipManifest)
from minagi.v161.evaluators import (containment_match,  # noqa: E402
                                    exact_match, retention_score,
                                    security_regression)
from minagi.v161.stats import (cluster_bootstrap_ci,  # noqa: E402
                               family_delta_analysis)

CONFIRMATION_SCHEMA = "mini-agi-v16.6-confirmation-record-v1"
SUBMISSION_SCHEMA = "mini-agi-v16.6-confirmation-submission-v1"
LEDGER_SCHEMA = "mini-agi-v16.6-confirmation-ledger-v1"

#: The preregistered primary comparisons, strongest first.
PRIMARY_COMPARISONS = (("L7", "L5b"), ("L6", "L5"))

BLOCKS = ("hidden", "retention", "security")


class ConfirmationRefused(Exception):
    """Eligibility/verification failure — nothing was executed."""


def _member(row) -> DatasetMember:
    payload = json.dumps(row, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False).encode()
    return DatasetMember(str(row["id"]), str(row["family"]),
                         sha256_bytes(payload),
                         str(row.get("source", "local")),
                         str(row.get("generator", "manual")))


def sealed_manifest_digest(rows) -> str:
    return DatasetMembershipManifest("final_holdout",
                                     tuple(_member(r) for r in rows)).digest


def load_sealed(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines()
            if line.strip()]
    if not rows:
        raise ConfirmationRefused("sealed confirmation set is empty")
    return rows


def load_submission(path: Path) -> dict:
    doc = json.loads(path.read_text())
    if doc.get("schema") != SUBMISSION_SCHEMA:
        raise ConfirmationRefused(
            f"submission schema must be {SUBMISSION_SCHEMA!r}")
    for key in ("campaign_id", "plan_digest", "model_id", "model_revision",
                "tokenizer_digest", "prompt_contract_digest",
                "output_contract", "adapters"):
        if not doc.get(key):
            raise ConfirmationRefused(f"submission missing {key!r}")
    return doc


def submission_identity(submission: dict) -> str:
    return digest({k: submission.get(k) for k in
                   ("schema", "campaign_id", "plan_digest", "model_id",
                    "model_revision", "tokenizer_digest",
                    "prompt_contract_digest", "output_contract",
                    "adapters")})


def verify_eligibility(storage: Path, plan, submission: dict) -> dict:
    """Freeze step: the research record must be complete and qualified,
    and the submission must agree with the signed plan."""
    rec_path = storage / "QUALIFICATION_RECORD.json"
    if not rec_path.is_file():
        raise ConfirmationRefused(
            "no research qualification record — nothing to confirm")
    rec = json.loads(rec_path.read_text())
    value = rec.get("value", rec)
    if value.get("decision") != "QUALIFIED":
        raise ConfirmationRefused(
            f"research decision {value.get('decision')!r} is not QUALIFIED "
            "— a failed candidate is not eligible for confirmation")
    if value.get("failures"):
        raise ConfirmationRefused(
            "research record carries integrity failures")
    if submission["plan_digest"] != plan.digest:
        raise ConfirmationRefused(
            "submission plan digest does not match the stored plan")
    if submission["tokenizer_digest"] != getattr(plan, "tokenizer_digest",
                                                 None):
        raise ConfirmationRefused(
            "submission tokenizer identity disagrees with the plan")
    return {"research_record_sha256": sha256_bytes(rec_path.read_bytes()),
            "research_decision": value.get("decision"),
            "research_plan_digest": value.get("campaign_plan_digest")}


def verify_sealed(plan, rows, submission: dict) -> dict:
    """Load step: the sealed set must be the bound holdout, and the
    design must cover the required blocks and families."""
    expected = getattr(plan, "final_holdout_digest", "") or ""
    if not expected:
        raise ConfirmationRefused(
            "plan binds no final_holdout_digest — sealed confirmation "
            "requires a preregistered evaluator-held set")
    got = sealed_manifest_digest(rows)
    if got != expected:
        raise ConfirmationRefused(
            f"sealed manifest digest {got} != plan.final_holdout_digest "
            f"{expected}")
    blocks: dict = {}
    for row in rows:
        blocks.setdefault(str(row.get("split", "hidden")), []).append(row)
    families = sorted({str(r["family"]) for r in blocks.get("hidden", [])})
    if len(families) < 2:
        raise ConfirmationRefused(
            f"sealed hidden block covers {len(families)} family(ies) — "
            "family-clustered uncertainty needs at least two")
    for required in ("retention", "security"):
        if not blocks.get(required):
            raise ConfirmationRefused(
                f"sealed set carries no {required} block — the "
                f"{required} gate would be unmeasurable")
    return {"manifest_digest": got,
            "blocks": {k: len(v) for k, v in sorted(blocks.items())},
            "hidden_families": families}


def primary_arms(plan_arms) -> tuple[str, str]:
    for treatment, baseline in PRIMARY_COMPARISONS:
        if treatment in plan_arms and baseline in plan_arms:
            return treatment, baseline
    raise ConfirmationRefused(
        f"plan arms {tuple(plan_arms)} carry no preregistered primary "
        "comparison (L7-L5b or L6-L5)")


def score_sealed(rows, outputs_by_seed, arms, *,
                 retention_scorer="retention_score") -> tuple[dict, dict]:
    """Score step: strict metrics recomputed from the executed outputs.

    Returns (scores_by_arm, per_row_hidden_scores) where the latter is
    keyed arm -> "seed:task_id" -> 0/1 for the paired comparison.
    """
    ret_fn = (containment_match if retention_scorer == "containment_match"
              else retention_score)
    scores: dict = {}
    per_row: dict = {a: {} for a in arms}
    for a in arms:
        hid, ret, sec = [], [], []
        for seed, outputs in outputs_by_seed:
            for row in rows:
                rid = str(row["id"])
                out = str(outputs[a].get(rid, ""))
                split = str(row.get("split", "hidden"))
                if split == "hidden":
                    s = float(exact_match(out, str(row["expected"])))
                    hid.append(s)
                    per_row[a][f"{seed}:{rid}"] = s
                elif split == "retention":
                    ret.append(float(ret_fn(out, str(row["expected"]))))
                elif split == "security":
                    sec.append(int(security_regression(out)))
        scores[a] = {
            "hidden_exact_match": statistics.fmean(hid) if hid else None,
            "retention": statistics.fmean(ret) if ret else None,
            "security_regressions": sum(sec),
            "security_pass_rate": (1.0 - sum(sec) / len(sec))
            if sec else None}
    return scores, per_row


def confirmation_gates(plan, rows, scores, per_row, *,
                       bootstrap_resamples: int = 2000,
                       ci_alpha: float = 0.05) -> tuple[str, list[str], dict]:
    """Qualification step: the preregistered gates applied to the sealed
    predictions only — neural gain, family coverage, uncertainty,
    retention and security."""
    treatment, baseline = primary_arms(tuple(scores))
    hidden = [r for r in rows if str(r.get("split", "hidden")) == "hidden"]
    deltas, fams = [], []
    for seed in plan.seeds:
        for row in hidden:
            key = f"{seed}:{row['id']}"
            t = per_row[treatment].get(key)
            b = per_row[baseline].get(key)
            if t is None or b is None:
                continue
            deltas.append(t - b)
            fams.append(str(row["family"]))
    reasons: list[str] = []
    stats: dict = {"n_pairs": len(deltas)}
    if not deltas:
        reasons.append("no paired sealed rows — the primary comparison "
                       "is unmeasurable")
    else:
        mean_delta = statistics.fmean(deltas)
        stats["primary"] = {"treatment": treatment, "baseline": baseline,
                            "mean_delta": mean_delta, "n_pairs": len(deltas)}
        if mean_delta < plan.minimum_neural_incremental_ft:
            reasons.append(f"sealed mean delta {mean_delta:.4f} < "
                           f"{plan.minimum_neural_incremental_ft}")
        ci = cluster_bootstrap_ci(deltas, fams, bootstrap_resamples,
                                  ci_alpha)
        stats["cluster_bootstrap_ci_row_delta"] = ci
        if ci["lower"] <= plan.min_delta_ft_ci_lower:
            reasons.append(
                f"sealed family-clustered CI lower bound {ci['lower']:.4f} "
                f"<= {plan.min_delta_ft_ci_lower}")
        stats["family_analysis"] = family_delta_analysis(deltas, fams)
    t_ret = scores[treatment]["retention"]
    b_ret = scores[baseline]["retention"]
    if t_ret is None or b_ret is None:
        reasons.append("sealed retention block unmeasured")
    else:
        stats["retention"] = {"treatment": t_ret, "baseline": b_ret,
                              "drop": b_ret - t_ret}
        if b_ret - t_ret > plan.retention_max_drop:
            reasons.append(f"sealed retention drop {b_ret - t_ret:.4f} > "
                           f"{plan.retention_max_drop}")
    sec = scores[treatment]["security_pass_rate"]
    if sec is None:
        reasons.append("sealed security block unmeasured")
    else:
        stats["security"] = {"treatment_pass_rate": sec,
                             "min_required": plan.security_min_pass_rate}
        if sec < plan.security_min_pass_rate:
            reasons.append(f"sealed security pass rate {sec:.3f} < "
                           f"{plan.security_min_pass_rate}")
    return ("QUALIFIED" if not reasons else "REFUSED"), reasons, stats


def run_confirmation(*, storage: Path, plan, submission, sealed_rows,
                     execute_arm, signers: dict,
                     retention_scorer: str = "retention_score",
                     bootstrap_resamples: int = 2000,
                     ci_alpha: float = 0.05) -> dict:
    """The whole evaluator-side flow with an injected executor.

    ``execute_arm(seed, arm_id, rows) -> {task_id: output}`` runs one
    arm for one seed; the CLI supplies the model-backed implementation,
    tests supply deterministic stubs.
    """
    eligibility = verify_eligibility(storage, plan, submission)
    sealed = verify_sealed(plan, sealed_rows, submission)
    arms = tuple(str(a) for a in plan.arms)
    outputs_by_seed = [(seed, {a: execute_arm(seed, a, sealed_rows)
                               for a in arms}) for seed in plan.seeds]
    scores, per_row = score_sealed(sealed_rows, outputs_by_seed, arms,
                                   retention_scorer=retention_scorer)
    decision, reasons, stats = confirmation_gates(
        plan, sealed_rows, scores, per_row,
        bootstrap_resamples=bootstrap_resamples, ci_alpha=ci_alpha)
    body = {"schema": CONFIRMATION_SCHEMA,
            "campaign_id": plan.campaign_id,
            "plan_digest": plan.digest,
            "submission_digest": submission_identity(submission),
            "eligibility": eligibility,
            "sealed": sealed,
            "arms": list(arms),
            "seeds": list(plan.seeds),
            "scores": scores,
            "stats": stats,
            "decision": decision,
            "reasons": reasons,
            "note": "evaluator-sealed confirmation — independent of the "
                    "research worker; promotion remains a separate "
                    "governed step"}
    sig = signers["qualification"].sign(body)
    return {"value": body, "digest": digest(body),
            "signer_key_id": signers["qualification"].key_id,
            "signature_b64": sig.signature_b64}


def append_ledger(path: Path, record: dict, *, actor: str) -> dict:
    """Append-only confirmation ledger (Step 22): attempt identity,
    sealed digest and decision — never rewritten."""
    seq = 0
    if path.is_file():
        seq = sum(1 for line in path.read_text().splitlines() if line.strip())
    entry = {"schema": LEDGER_SCHEMA, "seq": seq,
             "campaign_id": record["value"]["campaign_id"],
             "submission_digest": record["value"]["submission_digest"],
             "sealed_manifest_digest":
                 record["value"]["sealed"]["manifest_digest"],
             "decision": record["value"]["decision"],
             "record_digest": record["digest"],
             "actor": actor}
    with path.open("a") as fh:
        fh.write(json.dumps(entry, sort_keys=True) + "\n")
    return entry


def _load_plan(campaign_dir: Path):
    from minagi.v161.campaign_plan import (ColabCampaignPlanV162,
                                           ColabCampaignPlanV163,
                                           ColabCampaignPlanV164,
                                           ColabCampaignPlanV165,
                                           ColabCampaignPlanV166)
    plan_doc = json.loads((campaign_dir / "CAMPAIGN_PLAN.json").read_text())
    plan_cls = {"mini-agi-v16.6-colab-campaign-plan-v1":
                ColabCampaignPlanV166,
                "mini-agi-v16.5-colab-campaign-plan-v1":
                ColabCampaignPlanV165,
                "mini-agi-v16.4-colab-campaign-plan-v1":
                ColabCampaignPlanV164,
                "mini-agi-v16.3-colab-campaign-plan-v1":
                ColabCampaignPlanV163}.get(plan_doc["value"].get("schema"),
                                           ColabCampaignPlanV162)
    plan = plan_cls(**plan_doc["value"])
    if plan.digest != plan_doc["digest"]:
        raise ConfirmationRefused("stored plan digest does not recompute")
    return plan


def _load_signers(storage: Path) -> dict:
    registry = AuthorityRegistry.load(storage / "trust_root.json")
    signers = {}
    for role in ("evaluation", "qualification"):
        pem = storage / ".keys" / f"{role}.pem"
        if not pem.is_file():
            raise ConfirmationRefused(
                f"evaluator role key {role!r} unavailable — confirmation "
                "must run under evaluator custody")
        s = Ed25519Signer.from_private_bytes(pem.read_bytes())
        if not registry.is_authorized(role, s.key_id):
            raise ConfirmationRefused(f"{role} key is not authorized")
        signers[role] = s
    return signers


def _model_executor(submission: dict, protocol):
    """The real execution path: pinned model + per-arm adapters, greedy
    decoding under the submission's frozen output contract."""
    import torch
    from minagi.platforms.cuda.hf_runtime import (HFLoadSpec,
                                                  load_causal_lm,
                                                  load_tokenizer)
    from minagi.v161.prompt_contract import (build_prompt_contract,
                                             encode_generation,
                                             output_contract_from_config)

    max_new = int(submission["output_contract"].get(
        "max_new_tokens", getattr(protocol, "max_new_tokens", 32)))
    output_contract = output_contract_from_config(
        {"output_contract": submission["output_contract"]}, max_new)
    spec = HFLoadSpec(submission["model_id"], submission["model_revision"],
                      "float32", "none", False)
    tokenizer = load_tokenizer(spec)
    contract = build_prompt_contract(tokenizer, output_contract)
    if contract.digest != submission["prompt_contract_digest"]:
        raise ConfirmationRefused(
            "rebuilt prompt contract digest disagrees with the submission "
            "— the frozen surface cannot be reproduced")

    def execute_arm(seed, arm_id, rows):
        adapter = submission["adapters"].get(arm_id)
        model = load_causal_lm(spec, adapter_path=adapter,
                               purpose="research")
        out = {}
        with torch.inference_mode():
            device = next(model.parameters()).device
            for row in rows:
                ids = encode_generation(tokenizer, str(row["prompt"]),
                                        contract)
                batch = {"input_ids": torch.tensor([ids], device=device),
                         "attention_mask": torch.ones(
                             (1, len(ids)), dtype=torch.long, device=device)}
                gen = model.generate(**batch, max_new_tokens=max_new,
                                     do_sample=False,
                                     pad_token_id=tokenizer.eos_token_id)
                out[str(row["id"])] = tokenizer.decode(
                    gen[0, len(ids):], skip_special_tokens=True).strip()
        return out

    return execute_arm


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--storage", required=True)
    ap.add_argument("--campaign-id", required=True)
    ap.add_argument("--sealed", required=True)
    ap.add_argument("--submission", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--ledger", default=None)
    args = ap.parse_args()

    storage = Path(args.storage)
    campaign_dir = storage / "campaigns" / args.campaign_id
    plan = _load_plan(campaign_dir)
    signers = _load_signers(storage)
    submission = load_submission(Path(args.submission))
    sealed_rows = load_sealed(Path(args.sealed))

    protocol = None
    proto_path = campaign_dir / "EXPERIMENT_PROTOCOL.json"
    if proto_path.is_file():
        from minagi.v161.experiment_protocol import ExperimentProtocolV1
        protocol = ExperimentProtocolV1(**json.loads(
            proto_path.read_text())["value"])
    execute_arm = _model_executor(submission, protocol)

    record = run_confirmation(storage=storage, plan=plan,
                              submission=submission,
                              sealed_rows=sealed_rows,
                              execute_arm=execute_arm, signers=signers,
                              retention_scorer=str(
                                  getattr(protocol, "retention_scorer",
                                          "retention_score")))
    out_path = Path(args.output)
    out_path.write_text(json.dumps(record, indent=2, sort_keys=True))
    ledger = Path(args.ledger) if args.ledger else \
        storage / "CONFIRMATION_LEDGER.jsonl"
    append_ledger(ledger, record, actor=signers["qualification"].key_id)
    print(json.dumps({"decision": record["value"]["decision"],
                      "reasons": record["value"]["reasons"],
                      "record": str(out_path)}, indent=2))
    return 0 if record["value"]["decision"] == "QUALIFIED" else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ConfirmationRefused as exc:
        print(f"confirmation refused: {exc}", file=sys.stderr)
        raise SystemExit(3)
