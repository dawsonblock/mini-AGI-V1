"""Campaign 3C — evaluator-sealed confirmation executor (3B Steps 20-22).

Drives the real tool (`scripts/validation/run_sealed_confirmation.py`)
with injected predictors over a synthetic evaluator custody tree:

  * a clean sealed run executes the arms, scores them, signs QUALIFIED
    with the qualification authority and appends one ledger entry,
  * an incomplete or non-qualified research record is refused before
    anything executes,
  * a sealed set that is not the bound holdout, or whose design lacks
    the required blocks/family coverage, is refused,
  * a candidate identity that disagrees with the signed plan is refused,
  * gate failure yields a signed REFUSED record (evidence preserved),
  * the confirmation ledger is append-only across attempts,
  * confirmation cannot run without the evaluator's own role keys.
"""
import importlib.util
import json
import sys
from dataclasses import asdict
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src-python"))

from egai.common.canonical import digest  # noqa: E402
from minagi.v161.authority import write_trust_root  # noqa: E402
from minagi.v161.campaign_plan import ColabCampaignPlanV166  # noqa: E402
from minagi.v161.experiment_protocol import ExperimentProtocolV1  # noqa: E402

TOOL_PATH = ROOT / "scripts" / "validation" / "run_sealed_confirmation.py"
CID = "c-sealed"
ARMS = ("L1", "L5", "L6", "NC")
SEEDS = (0,)


def _tool():
    spec = importlib.util.spec_from_file_location("rsc_under_test", TOOL_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


TOOL = _tool()


def _rows():
    # distinctive synthetic strings: short lowercase tokens like "ha2"
    # collide with the ubiquitous "sha256:" prefix in serialized
    # records, which the leak tests would flag as a false positive
    return (
        [{"id": f"h{i}", "split": "hidden", "family": f"fam-{i % 2}",
          "prompt": f"SEALED-PROMPT-{i}", "expected": f"SEALED-ANSWER-{i}",
          "source": "test", "generator": "synthetic"} for i in range(4)] +
        [{"id": f"r{i}", "split": "retention", "family": "fam-ret",
          "prompt": f"SEALED-RET-PROMPT-{i}",
          "expected": f"SEALED-RET-ANSWER-{i}",
          "source": "test", "generator": "synthetic"} for i in range(2)] +
        [{"id": f"s{i}", "split": "security", "family": "fam-sec",
          "prompt": f"SEALED-SEC-PROMPT-{i}", "expected": "",
          "source": "test", "generator": "synthetic"} for i in range(2)])


def _protocol():
    return ExperimentProtocolV1.from_config({
        "model": {"dtype": "bfloat16", "quantization": "none",
                  "trust_remote_code": False},
        "retention_scorer": "retention_score",
        "require_native_servable_adapter": False,
        "lora": {"rank": 16, "alpha": 32, "dropout": 0.05,
                 "target_modules": ["q_proj"], "learning_rate": 1e-4,
                 "steps": 100, "max_length": 256, "optimizer": "adamw"}})


class Custody:
    """An evaluator custody tree: storage, plan, sealed set, submission."""

    def __init__(self, tmp_path: Path, *, decision="QUALIFIED",
                 rows=None, plan_digest_override=None):
        self.storage = tmp_path / "storage"
        self.cdir = self.storage / "campaigns" / CID
        self.cdir.mkdir(parents=True)
        write_trust_root(self.storage / ".keys",
                         self.storage / "trust_root.json")
        self.rows = rows if rows is not None else _rows()
        self.protocol = _protocol()
        (self.cdir / "EXPERIMENT_PROTOCOL.json").write_text(json.dumps(
            {"value": asdict(self.protocol),
             "digest": self.protocol.digest}))
        self.plan = ColabCampaignPlanV166(
            campaign_id=CID, model_id="synthetic-model",
            model_revision="rev1",
            dataset_partition_digest=digest("parts"),
            scorer_artifact_digest=digest("scorer"),
            retention_artifact_digest=digest("retention"),
            security_artifact_digest=digest("security"),
            seeds=SEEDS, arms=ARMS, negative_control_arm="NC",
            negative_control_max_ft=0.02,
            minimum_neural_incremental_ft=0.02,
            minimum_retention=0.0, require_zero_security_regressions=False,
            metric_tolerance=0.05,
            model_digest=digest("model"), tokenizer_digest=digest("tok"),
            generation_template_digest=digest("tmpl"),
            retention_max_drop=0.10, security_min_pass_rate=0.5,
            security_max_drop_vs_L1=0.10, min_seeds_positive_ft=1,
            dataset_path="corpus/tasks.jsonl",
            require_family_disjoint_hidden=True,
            bootstrap_resamples=2000, ci_alpha=0.05,
            min_delta_ft_ci_lower=0.0,
            delayed_retention_max_drop=0.10,
            experiment_protocol_digest=self.protocol.digest,
            final_holdout_digest=TOOL.sealed_manifest_digest(self.rows),
            cluster_unit="family", max_false_activation_rate=0.05,
            n_comparisons_preregistered=1)
        (self.cdir / "CAMPAIGN_PLAN.json").write_text(json.dumps(
            {"value": asdict(self.plan), "digest": self.plan.digest}))
        (self.storage / "QUALIFICATION_RECORD.json").write_text(json.dumps(
            {"value": {"decision": decision, "failures": [],
                       "campaign_plan_digest":
                           plan_digest_override or self.plan.digest}}))
        self.submission = {
            "schema": TOOL.SUBMISSION_SCHEMA, "campaign_id": CID,
            "plan_digest": plan_digest_override or self.plan.digest,
            "model_id": "synthetic-model", "model_revision": "rev1",
            "tokenizer_digest": self.plan.tokenizer_digest,
            "prompt_contract_digest": digest("contract"),
            "output_contract": {"instruction": "", "max_new_tokens": 8,
                                "stop_on_eos": True},
            "adapters": {"L6": "unused", "NC": "unused"}}
        self.signers = {}
        from egai.common.crypto import Ed25519Signer
        for role in ("evaluation", "qualification"):
            self.signers[role] = Ed25519Signer.from_private_bytes(
                (self.storage / ".keys" / f"{role}.pem").read_bytes())

    def run(self, execute_arm, **kwargs):
        return TOOL.run_confirmation(
            storage=self.storage, plan=self.plan,
            submission=self.submission, sealed_rows=self.rows,
            execute_arm=execute_arm, signers=self.signers, **kwargs)

    def ledger_entries(self):
        path = self.storage / "CONFIRMATION_LEDGER.jsonl"
        if not path.is_file():
            return []
        return [json.loads(line) for line in path.read_text().splitlines()
                if line.strip()]


def _executor(l5_correct=1, l6_correct=4, sec_output="benign"):
    """Deterministic stub: the first N hidden rows are answered
    correctly; retention rows are always correct."""
    def execute(seed, arm_id, rows):
        out = {}
        hid = [r for r in rows if str(r.get("split")) == "hidden"]
        for i, row in enumerate(hid):
            n = {"L1": l5_correct, "L5": l5_correct, "L6": l6_correct,
                 "NC": l5_correct}.get(arm_id, 0)
            out[str(row["id"])] = (str(row["expected"]) if i < n else "wrong")
        for row in rows:
            split = str(row.get("split"))
            if split == "retention":
                out[str(row["id"])] = str(row["expected"])
            elif split == "security":
                out[str(row["id"])] = sec_output
        return out
    return execute


def test_sealed_confirmation_executes_scores_and_signs(tmp_path):
    c = Custody(tmp_path)
    record = c.run(_executor())
    value = record["value"]
    assert value["decision"] == "QUALIFIED", value["reasons"]
    assert value["eligibility"]["research_decision"] == "QUALIFIED"
    assert value["stats"]["primary"]["treatment"] == "L6"
    assert value["stats"]["primary"]["mean_delta"] == 0.75
    assert value["stats"]["cluster_bootstrap_ci_row_delta"]["n_clusters"] == 2
    assert value["stats"]["family_analysis"]["n_families"] == 2
    assert value["sealed"]["blocks"] == {"hidden": 4, "retention": 2,
                                         "security": 2}
    assert record["signer_key_id"] == c.signers["qualification"].key_id
    TOOL.append_ledger(c.storage / "CONFIRMATION_LEDGER.jsonl", record,
                       actor=c.signers["qualification"].key_id)
    entries = c.ledger_entries()
    assert len(entries) == 1 and entries[0]["decision"] == "QUALIFIED"
    assert entries[0]["seq"] == 0
    assert entries[0]["sealed_manifest_digest"] == \
        value["sealed"]["manifest_digest"]


def test_refuses_non_qualified_research_record(tmp_path):
    c = Custody(tmp_path, decision="REFUSED")
    with pytest.raises(TOOL.ConfirmationRefused, match="not QUALIFIED"):
        c.run(_executor())


def test_refuses_missing_research_record(tmp_path):
    c = Custody(tmp_path)
    (c.storage / "QUALIFICATION_RECORD.json").unlink()
    with pytest.raises(TOOL.ConfirmationRefused, match="no research"):
        c.run(_executor())


def test_refuses_wrong_sealed_set(tmp_path):
    c = Custody(tmp_path)
    c.rows = _rows() + [{"id": "extra", "split": "hidden",
                         "family": "fam-2", "prompt": "p", "expected": "e",
                         "source": "test", "generator": "synthetic"}]
    with pytest.raises(TOOL.ConfirmationRefused, match="sealed manifest"):
        c.run(_executor())


def test_refuses_insufficient_coverage(tmp_path):
    one_family = [dict(r, family="fam-only")
                  for r in _rows() if r["split"] == "hidden"]
    one_family += [r for r in _rows() if r["split"] != "hidden"]
    c = Custody(tmp_path, rows=one_family)
    with pytest.raises(TOOL.ConfirmationRefused, match="at least two"):
        c.run(_executor())

    no_security = [r for r in _rows() if r["split"] != "security"]
    c2 = Custody(tmp_path / "second", rows=no_security)
    with pytest.raises(TOOL.ConfirmationRefused, match="security"):
        c2.run(_executor())


def test_refuses_identity_mismatch(tmp_path):
    c = Custody(tmp_path)
    c.submission["tokenizer_digest"] = digest("other")
    with pytest.raises(TOOL.ConfirmationRefused, match="tokenizer"):
        c.run(_executor())
    c2 = Custody(tmp_path / "second")
    c2.submission["plan_digest"] = digest("other-plan")
    with pytest.raises(TOOL.ConfirmationRefused, match="plan digest"):
        c2.run(_executor())


def test_gate_failure_is_signed_refusal(tmp_path):
    c = Custody(tmp_path)
    record = c.run(_executor(l5_correct=4, l6_correct=4))  # no gain
    value = record["value"]
    assert value["decision"] == "REFUSED"
    assert any("mean delta" in r for r in value["reasons"])
    assert record["signature_b64"]
    assert record["signer_key_id"] == c.signers["qualification"].key_id


def test_confirmation_requires_evaluator_keys(tmp_path):
    c = Custody(tmp_path)
    (c.storage / ".keys" / "qualification.pem").unlink()
    with pytest.raises(TOOL.ConfirmationRefused, match="unavailable"):
        TOOL._load_signers(c.storage)
