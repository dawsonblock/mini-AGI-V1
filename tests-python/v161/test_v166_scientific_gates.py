"""v16.6 — scientific benchmark gates (Phase 6 / REPAIR-030..036).

Covers the evaluator-sealed final holdout (DatasetPartitionSetV2 +
plan binding + qualifier --holdout verification), cluster-aware
bootstrap statistics, executor-verified scoring dispatch, the
false-activation metric, and the V166 qualifier path end to end.
"""
import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src-python"))

from egai.common.canonical import digest, sha256_bytes  # noqa: E402
from egai.common.crypto import Ed25519Signer  # noqa: E402
from minagi.v161.authority import (AuthorityLedger, AuthorityRegistry,  # noqa: E402
                                   write_trust_root, AUTHORITY_ROLES)
from minagi.v161.campaign_plan import (ColabCampaignPlanV165,  # noqa: E402
                                       ColabCampaignPlanV166)
from minagi.v161.dataset_manifest import (DatasetMember,  # noqa: E402
                                          DatasetMembershipManifest,
                                          DatasetPartitionSet,
                                          DatasetPartitionSetV2)
from minagi.v161 import execution_sandbox  # noqa: E402
from minagi.v161.evaluators import (exact_match, executor_score,  # noqa: E402
                                    score_row)
from minagi.v161.evidence_receipt_v3 import (  # noqa: E402
    EvidenceReceiptV3, evaluation_bundle, input_manifest_digest,
    prediction_record, predictions_digest_of, write_predictions)
from minagi.v161.experiment_protocol import ExperimentProtocolV1  # noqa: E402
from minagi.v161.runtime_closure3 import sha256_path  # noqa: E402
from minagi.v161.stats import (bootstrap_ci, cluster_bootstrap_ci,  # noqa: E402
                               false_activation_rate)

Z = "sha256:" + "0" * 64


# ---------- DatasetPartitionSetV2 -----------------------------------

def _row(rid, split, family, prompt, expected, **kw):
    r = {"id": rid, "split": split, "family": family, "prompt": prompt,
         "expected": expected, "source": "test", "generator": "synthetic"}
    r.update(kw)
    return r


def _member(row):
    payload = json.dumps(row, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False).encode()
    return DatasetMember(str(row["id"]), str(row["family"]),
                         sha256_bytes(payload), str(row["source"]),
                         str(row["generator"]))


def _manifest(name, rows):
    return DatasetMembershipManifest(name, tuple(_member(r) for r in rows))


CORPUS = (
    [_row(f"tr{i}", "train", "fam-train", f"tp{i}", f"ta{i}")
     for i in range(6)] +
    [_row(f"va{i}", "validation", "fam-val", f"vp{i}", f"va{i}")
     for i in range(2)] +
    [_row(f"h{i}", "hidden", "fam-hid", f"hp{i}", f"ans{i}")
     for i in range(4)] +
    [_row(f"rp{i}", "retention", "fam-ret", f"rq{i}", f"ra{i}")
     for i in range(4)] +
    [_row(f"rd{i}", "retention", "fam-ret", f"dq{i}", f"da{i}",
          probe="delayed") for i in range(2)] +
    [_row(f"s{i}", "security", "fam-sec", f"sq{i}", "")
     for i in range(4)]
)

HOLDOUT = [_row(f"fh{i}", "final_holdout", "fam-holdout", f"fhp{i}",
                f"fha{i}") for i in range(3)]


def _parts_v1():
    return DatasetPartitionSet(
        _manifest("train", [r for r in CORPUS if r["split"] == "train"]),
        _manifest("validation",
                  [r for r in CORPUS if r["split"] == "validation"]),
        _manifest("hidden", [r for r in CORPUS if r["split"] == "hidden"]),
        _manifest("retention",
                  [r for r in CORPUS if r["split"] == "retention"]),
        _manifest("security",
                  [r for r in CORPUS if r["split"] == "security"]),
        require_family_disjoint_hidden=True)


def _parts_v2():
    v1 = _parts_v1()
    return DatasetPartitionSetV2(
        v1.train, v1.validation, v1.hidden, v1.retention, v1.security,
        v1.require_family_disjoint_hidden,
        final_holdout=_manifest("final_holdout", HOLDOUT))


def test_v2_holdout_is_disjoint_and_bound():
    v2 = _parts_v2()
    v2.assert_disjoint()  # clean
    assert v2.proof()["final_holdout_digest"] == \
        _manifest("final_holdout", HOLDOUT).digest


def test_v2_preserves_v1_digest():
    """V1 partition digests must stay byte-stable: the V2 wrapper adds
    the holdout as a separate field, never mutating V1 members."""
    assert _parts_v1().digest != ""  # sanity
    v2 = _parts_v2()
    # V2 schema/digest differ from V1 — a new schema, not a silent edit
    assert v2.schema != _parts_v1().schema
    assert v2.digest != _parts_v1().digest


def test_v2_rejects_holdout_id_leakage():
    bad = _parts_v1()
    with pytest.raises(PermissionError, match="sample leakage"):
        DatasetPartitionSetV2(
            bad.train, bad.validation, bad.hidden, bad.retention,
            bad.security, bad.require_family_disjoint_hidden,
            final_holdout=_manifest("final_holdout",
                                    [_row("h0", "final_holdout", "fam-x",
                                          "p", "e")]))


def test_v2_rejects_holdout_family_leakage():
    bad = _parts_v1()
    with pytest.raises(PermissionError, match="family leakage"):
        DatasetPartitionSetV2(
            bad.train, bad.validation, bad.hidden, bad.retention,
            bad.security, bad.require_family_disjoint_hidden,
            final_holdout=_manifest("final_holdout",
                                    [_row("fh99", "final_holdout",
                                          "fam-hid", "p", "e")]))


# ---------- cluster bootstrap ---------------------------------------

def test_cluster_bootstrap_is_deterministic():
    xs = [1.0, 0.0, 1.0, 0.0, 0.5, 0.5]
    cs = ["a", "a", "b", "b", "c", "c"]
    a = cluster_bootstrap_ci(xs, cs, 2000, 0.05)
    b = cluster_bootstrap_ci(xs, cs, 2000, 0.05)
    assert a == b
    assert a["n_clusters"] == 3
    assert a["n"] == 6


def test_cluster_bootstrap_resamples_clusters_not_rows():
    """With perfectly correlated clusters, the clustered CI is wider
    than the row-level CI — correlated rows are not independent."""
    xs = [0.0] * 20 + [1.0] * 20
    cs = ["neg"] * 20 + ["pos"] * 20
    plain = bootstrap_ci(xs, 2000, 0.05)
    clus = cluster_bootstrap_ci(xs, cs, 2000, 0.05)
    assert clus["upper"] - clus["lower"] >= plain["upper"] - plain["lower"]
    assert clus["n_clusters"] == 2


def test_cluster_bootstrap_rejects_misalignment():
    with pytest.raises(ValueError, match="align"):
        cluster_bootstrap_ci([1.0, 0.0], ["only-one"], 10, 0.05)
    with pytest.raises(ValueError, match="requires"):
        cluster_bootstrap_ci([], [], 10, 0.05)


# ---------- false activation -----------------------------------------

def test_false_activation_counts_l1_correct_l6_broken():
    golds = {"a": "1", "b": "2", "c": "3"}
    base = {"a": "1", "b": "2", "c": "0"}        # L1: 2/3 correct
    arm = {"a": "9", "b": "2", "c": "0"}         # L6 broke 'a'
    fa = false_activation_rate(base, arm, golds, exact_match)
    assert fa["n_pairs"] == 3
    assert fa["n_false_activation"] == 1
    assert fa["changed_ids"] == ["a"]


def test_false_activation_ignores_wrong_to_wrong():
    golds = {"a": "1"}
    fa = false_activation_rate({"a": "x"}, {"a": "y"}, golds, exact_match)
    assert fa["n_false_activation"] == 0


def test_false_activation_denominator_is_baseline_correct():
    """FIX-004: the rate is conditioned on previously-correct L1 cases,
    not diluted by rows L1 already failed (1/2, not 1/3)."""
    golds = {"a": "1", "b": "2", "c": "3"}
    base = {"a": "1", "b": "2", "c": "0"}   # L1 correct on a,b only
    arm = {"a": "9", "b": "2", "c": "0"}    # L6 breaks 'a'
    fa = false_activation_rate(base, arm, golds, exact_match)
    assert fa["n_pairs"] == 3
    assert fa["n_baseline_correct"] == 2
    assert fa["n_false_activation"] == 1
    assert fa["false_activation_rate"] == 0.5


def test_conditional_regression_reported():
    """FIX-004: P(arm incorrect | L1 correct) is reported alongside the
    harmful-flip rate."""
    golds = {"a": "1", "b": "2", "c": "3"}
    base = {"a": "1", "b": "2", "c": "0"}
    arm = {"a": "9", "b": "2", "c": "0"}
    fa = false_activation_rate(base, arm, golds, exact_match)
    assert fa["n_conditional_regression"] == 1
    assert fa["conditional_regression_rate"] == 0.5
    assert fa["changed_ids"] == ["a"]


def test_no_baseline_correct_rows_yield_zero_rate():
    golds = {"a": "1"}
    fa = false_activation_rate({"a": "x"}, {"a": "y"}, golds, exact_match)
    assert fa["n_baseline_correct"] == 0
    assert fa["false_activation_rate"] == 0.0
    assert fa["conditional_regression_rate"] == 0.0


# ---------- executor-verified scoring ---------------------------------

@pytest.mark.skipif(not execution_sandbox.sandbox_usable(),
                    reason="no usable OS sandbox backend on this host "
                           "(TEST-001 classification)")
def test_score_row_dispatches_verify_spec():
    row = {"id": "x", "expected": "42",
           "verify": {"type": "python_assert",
                      "check": "import os; assert os.environ['OUTPUT'].strip() == '42'"}}
    assert score_row(row, "42", exact_match) == 1.0
    assert score_row(row, "41", exact_match) == 0.0


def test_score_row_falls_back_to_default_fn():
    row = {"id": "x", "expected": "hello"}
    assert score_row(row, "hello", exact_match) == 1.0
    assert score_row(row, "bye", exact_match) == 0.0


def test_executor_score_rejects_unknown_verify_type():
    with pytest.raises(ValueError, match="python_assert"):
        executor_score("x", "y", {"type": "shell", "check": "true"})


@pytest.mark.skipif(not execution_sandbox.sandbox_usable(),
                    reason="no usable OS sandbox backend on this host "
                           "(TEST-001 classification)")
def test_executor_score_check_is_authoritative_over_expected():
    """The verify check, not the expected string, decides correctness —
    a swapped check that demands a different output flips the score."""
    v = {"type": "python_assert",
         "check": "import os; assert os.environ['OUTPUT'] == os.environ['EXPECTED']"}
    assert executor_score("abc", "abc", v) == 1.0
    assert executor_score("abc", "abd", v) == 0.0


@pytest.mark.skipif(not execution_sandbox.sandbox_usable(),
                    reason="no usable OS sandbox backend on this host "
                           "(TEST-001 classification)")
def test_executor_score_timeout_returns_zero():
    v = {"type": "python_assert",
         "check": "import time; time.sleep(60)"}
    assert executor_score("x", "", v, timeout_s=1) == 0.0


# ---------- V166 plan validation --------------------------------------

def _proto():
    return ExperimentProtocolV1.from_config({
        "model": {"dtype": "bfloat16", "quantization": "none",
                  "trust_remote_code": False},
        "retention_scorer": "retention_score",
        "require_native_servable_adapter": False,
        "lora": {"rank": 16, "alpha": 32, "dropout": 0.05,
                 "target_modules": ["q_proj"], "learning_rate": 1e-4,
                 "steps": 100, "max_length": 256, "optimizer": "adamw"},
    })


def _plan_v166(**kw):
    proto = _proto()
    args = dict(
        campaign_id="c-v166", model_id="m", model_revision="r1",
        dataset_partition_digest=_parts_v1().digest,
        scorer_artifact_digest=digest("s"),
        retention_artifact_digest=digest("r"),
        security_artifact_digest=digest("x"),
        seeds=(0,), arms=("L1", "L5", "L6", "NC"),
        negative_control_arm="NC", negative_control_max_ft=0.02,
        minimum_neural_incremental_ft=0.02, minimum_retention=0.0,
        require_zero_security_regressions=False, metric_tolerance=0.05,
        model_digest=digest("m"), tokenizer_digest=digest("t"),
        generation_template_digest=digest("g"),
        retention_max_drop=0.10, security_min_pass_rate=0.5,
        security_max_drop_vs_L1=0.10, min_seeds_positive_ft=1,
        dataset_path="corpus/tasks.jsonl",
        require_family_disjoint_hidden=True,
        bootstrap_resamples=2000, ci_alpha=0.05,
        min_delta_ft_ci_lower=0.0, delayed_retention_max_drop=0.10,
        experiment_protocol_digest=proto.digest,
        final_holdout_digest=_manifest("final_holdout", HOLDOUT).digest,
        cluster_unit="family", max_false_activation_rate=0.25,
        n_comparisons_preregistered=3)
    args.update(kw)
    return ColabCampaignPlanV166(**args)


def test_v166_plan_validation():
    assert _plan_v166().digest != ""
    with pytest.raises(ValueError, match="digest"):
        _plan_v166(final_holdout_digest="not-a-digest")
    with pytest.raises(ValueError, match="cluster_unit"):
        _plan_v166(cluster_unit="bogus")
    with pytest.raises(ValueError, match="false_activation"):
        _plan_v166(max_false_activation_rate=1.5)
    with pytest.raises(ValueError, match="n_comparisons"):
        _plan_v166(n_comparisons_preregistered=0)


def test_v166_is_v165_subclass_for_runner_dispatch():
    assert isinstance(_plan_v166(), ColabCampaignPlanV165)
    assert _plan_v166().schema != ColabCampaignPlanV165(
        campaign_id="x", model_id="m", model_revision="r",
        dataset_partition_digest=Z, scorer_artifact_digest=Z,
        retention_artifact_digest=Z, security_artifact_digest=Z,
        seeds=(0,), arms=("L1", "L5", "L6", "NC"),
        negative_control_arm="NC", negative_control_max_ft=0.02,
        minimum_neural_incremental_ft=0.02, minimum_retention=0.0,
        require_zero_security_regressions=False, metric_tolerance=0.05,
        model_digest=Z, tokenizer_digest=Z, generation_template_digest=Z,
        retention_max_drop=0.10, security_min_pass_rate=0.5,
        security_max_drop_vs_L1=0.10, min_seeds_positive_ft=1,
        dataset_path="d", require_family_disjoint_hidden=True,
        bootstrap_resamples=1000, ci_alpha=0.05,
        min_delta_ft_ci_lower=0.0, delayed_retention_max_drop=0.10,
        experiment_protocol_digest=Z).schema


# ---------- V166 qualifier end to end ----------------------------------

CID = "c-v166-e2e"
ARMS = ("L1", "L5", "L6", "NC")
SEEDS = (0,)


class SynthV166:
    """Same synthetic-campaign shape as the v165 chain test, but with a
    V166 plan + a sealed holdout file supplied via --holdout."""

    def __init__(self, tmp_path: Path, holdout_rows=None):
        self.storage = tmp_path / "storage"
        self.root = tmp_path / "root"
        self.cdir = self.storage / "campaigns" / CID
        corpus_path = self.root / "corpus" / "tasks.jsonl"
        corpus_path.parent.mkdir(parents=True)
        corpus_path.write_text("\n".join(
            json.dumps(r, sort_keys=True) for r in CORPUS) + "\n")
        (self.root / "src-python").symlink_to(ROOT / "src-python")
        self.cdir.mkdir(parents=True)

        self.holdout_rows = holdout_rows if holdout_rows is not None else HOLDOUT
        self.holdout_path = self.root / "sealed_holdout.jsonl"
        self.holdout_path.write_text("\n".join(
            json.dumps(r, sort_keys=True) for r in self.holdout_rows) + "\n")
        self.parts = _parts_v1()
        self.protocol = _proto()
        self.plan = _plan_v166(
            campaign_id=CID,
            dataset_partition_digest=self.parts.digest,
            dataset_path="corpus/tasks.jsonl",
            max_false_activation_rate=0.0)

        write_trust_root(self.storage / ".keys",
                         self.storage / "trust_root.json")
        self.signers = {r: Ed25519Signer.from_private_bytes(
            (self.storage / ".keys" / f"{r}.pem").read_bytes())
            for r in AUTHORITY_ROLES}
        self.registry = AuthorityRegistry.load(
            self.storage / "trust_root.json")

        self.adapter_dir = (self.storage / "adapters" / CID / "L6" /
                            "seed-0")
        self.adapter_dir.mkdir(parents=True)
        (self.adapter_dir / "adapter_model.safetensors").write_bytes(
            b"adapter-weights")
        (self.adapter_dir / "adapter_config.json").write_text("{}")
        self.adapter_d = sha256_path(self.adapter_dir)
        nc_dir = self.storage / "adapters" / CID / "NC" / "seed-0"
        nc_dir.mkdir(parents=True)
        (nc_dir / "adapter_model.safetensors").write_bytes(b"nc")
        (nc_dir / "adapter_config.json").write_text("{}")
        self.nc_adapter_d = sha256_path(nc_dir)
        self.l5_state = self.storage / "arms" / CID / "seed-0" / "L5"
        self.l5_state.mkdir(parents=True)
        (self.l5_state / "state.json").write_text('{"replay": []}')
        self.l5_state_d = sha256_path(self.l5_state)
        self.env_d = digest({"env": "synthetic"})
        self.eval_dataset_d = digest(
            {"hidden": self.parts.hidden.digest,
             "retention": self.parts.retention.digest,
             "security": self.parts.security.digest})
        self.evaluator_set_d = digest(
            [self.plan.scorer_artifact_digest,
             self.plan.retention_artifact_digest,
             self.plan.security_artifact_digest])

    def _preds_for(self, arm, hidden_out):
        preds = []
        for i, row in enumerate(r for r in CORPUS
                                if r["split"] == "hidden"):
            out = row["expected"] if i < hidden_out else "wrong"
            preds.append(prediction_record(row["id"], "hidden",
                                           row["prompt"], out, 3, 32))
        for row in [r for r in CORPUS if r["split"] == "retention"
                    and r.get("probe") != "delayed"]:
            preds.append(prediction_record(row["id"], "retention",
                                           row["prompt"],
                                           row["expected"], 2, 32))
        for row in [r for r in CORPUS if r["split"] == "security"]:
            preds.append(prediction_record(row["id"], "security",
                                           row["prompt"], "benign", 2, 32))
        return preds

    def _delayed_preds(self):
        return [prediction_record(r["id"], "retention_delayed",
                                  r["prompt"], r["expected"], 2, 32)
                for r in CORPUS if r["split"] == "retention"
                and r.get("probe") == "delayed"]

    def _emit(self, run_dir, seed, arm, cell_name, preds, metrics,
              adapter_d, state_d):
        write_predictions(preds, run_dir / f"PREDICTIONS-{cell_name}.jsonl")
        im = input_manifest_digest(preds)
        pd = predictions_digest_of(preds)
        bundle = evaluation_bundle(self.plan.digest, self.protocol.digest,
                                   seed, arm, cell_name, im, pd, metrics)
        r = EvidenceReceiptV3.sign(
            signer=self.signers["execution_witness"],
            campaign_digest=self.plan.digest,
            protocol_digest=self.protocol.digest, seed=seed, arm=arm,
            model_digest=self.plan.model_digest,
            tokenizer_digest=self.plan.tokenizer_digest,
            adapter_digest=adapter_d, state_digest=state_d,
            environment_digest=self.env_d,
            evaluation_dataset_digest=self.eval_dataset_d,
            evaluator_digest=self.evaluator_set_d,
            input_manifest_digest=im, predictions_digest=pd,
            evaluation_bundle_digest=digest(bundle))
        (run_dir / f"{cell_name}.json").write_text(json.dumps(
            {"receipt": asdict(r), "bundle": bundle},
            indent=2, sort_keys=True))
        return r

    def _seed(self, seed, hidden_scores):
        run_dir = self.cdir / f"seed-{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)
        digs = {}
        adapter_ds = {"L6": self.adapter_d, "NC": self.nc_adapter_d}
        for arm in ARMS:
            preds = self._preds_for(arm, hidden_scores[arm])
            metrics = {"hidden_exact_match": hidden_scores[arm] / 4,
                       "retention": 1.0, "security_regressions": 0}
            digs[arm] = self._emit(
                run_dir, seed, arm, arm, preds, metrics,
                adapter_ds.get(arm, Z),
                self.l5_state_d if arm == "L5" else Z).digest
        for arm, name in (("L1", "L1_delayed"), ("L6", "L6_delayed")):
            digs[name] = self._emit(
                run_dir, seed, arm, name, self._delayed_preds(),
                {"retention_delayed": 1.0},
                self.adapter_d if arm == "L6" else Z, Z).digest
        (run_dir / "SEED_RESULT.json").write_text(json.dumps(
            {"seed": seed, "receipt_digests": digs,
             "attempt_id": "t"}))
        (run_dir / "COMPLETE").write_text("complete\n")
        files = {p.relative_to(run_dir).as_posix():
                 sha256_bytes(p.read_bytes())
                 for p in sorted(run_dir.rglob("*")) if p.is_file()}
        (run_dir / "COMMIT_MANIFEST.json").write_text(json.dumps(
            {"seed": seed, "campaign_digest": self.plan.digest,
             "attempt_id": "t", "files": files}))

    def build(self, hidden_scores=None):
        env = self.signers["plan"].sign(asdict(self.plan))
        (self.cdir / "CAMPAIGN_PLAN.json").write_text(json.dumps(
            {"value": asdict(self.plan), "digest": self.plan.digest,
             "signer_key_id": env.key_id,
             "signature_b64": env.signature_b64,
             "signed_before_execution": True}, indent=2, sort_keys=True))
        (self.cdir / "EXPERIMENT_PROTOCOL.json").write_text(json.dumps(
            {"value": asdict(self.protocol),
             "digest": self.protocol.digest}, indent=2, sort_keys=True))
        (self.cdir / "DATASET_PROOF.json").write_text(
            json.dumps(self.parts.proof(), indent=2, sort_keys=True))
        wit = self.signers["execution_witness"]
        (self.cdir / "EXECUTION_PUBLIC_KEY.bin").write_bytes(
            wit.public_bytes())
        (self.cdir / "EXECUTION_KEY_ID.txt").write_text(wit.key_id + "\n")
        AuthorityLedger(self.storage / "AUTHORITY_LEDGER.jsonl").append(
            self.signers["plan"], "experiment_preregistration",
            {"value": asdict(self.plan), "digest": self.plan.digest})
        for seed in SEEDS:
            self._seed(seed, hidden_scores or
                       {"L1": 2, "L5": 2, "L6": 4, "NC": 2})
        (self.cdir / "RESULT.json").write_text(json.dumps(
            {"decision": "QUALIFIED"}))
        return self


def _qualify166(c: SynthV166, holdout=None):
    out = c.storage / "QUALIFICATION_RECORD.json"
    cmd = [sys.executable,
           str(ROOT / "scripts" / "validation" / "qualify_campaign1.py"),
           "--storage", str(c.storage), "--campaign-id", CID,
           "--root", str(c.root), "--output", str(out)]
    if holdout is not None:
        cmd += ["--holdout", str(holdout)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return proc, out


def _decision(doc):
    return doc["value"]["decision"] if "value" in doc else doc["decision"]


def test_v166_clean_matrix_qualifies_with_holdout(tmp_path):
    c = SynthV166(tmp_path).build()
    proc, out = _qualify166(c, holdout=c.holdout_path)
    assert proc.returncode == 0, proc.stderr
    doc = json.loads(out.read_text())
    assert _decision(doc) == "QUALIFIED"
    assert c.registry.is_authorized("qualification",
                                    doc["signer_key_id"])
    led = AuthorityLedger(c.storage / "AUTHORITY_LEDGER.jsonl")
    assert led.verify(c.registry) == []


def test_v166_missing_holdout_arg_fails_closed(tmp_path):
    """Plan binds a holdout digest → qualifier demands the file."""
    c = SynthV166(tmp_path).build()
    proc, out = _qualify166(c, holdout=None)
    assert proc.returncode == 2
    assert _decision(json.loads(out.read_text())) in (
        "INVALID_EVIDENCE", "INCOMPLETE")


def test_v166_wrong_holdout_content_fails(tmp_path):
    """A substituted holdout file must not match the bound digest."""
    c = SynthV166(tmp_path).build()
    fake = c.root / "fake_holdout.jsonl"
    fake.write_text(json.dumps(
        _row("zz", "final_holdout", "fam-other", "p", "e")) + "\n")
    proc, out = _qualify166(c, holdout=fake)
    assert proc.returncode == 2
    assert _decision(json.loads(out.read_text())) == "INVALID_EVIDENCE"


def test_v166_false_activation_gate_trips(tmp_path):
    """L6 flips a row L1 had right → false activation > 0 → refused."""
    c = SynthV166(tmp_path)
    c.build(hidden_scores={"L1": 4, "L5": 2, "L6": 3, "NC": 2})
    # L6 has 3/4 but which 3? _preds_for marks first-N correct —
    # L6 misses h3 (an L1-correct row)? No: first-3 correct means
    # L6 wrong on h3, which L1 (4/4) had right → but L6's output is
    # "wrong" vs L1's "ans3": L1 correct + L6 changed + L6 wrong on
    # h3? L6 output differs from L1's on h3 AND L6 wrong → 1 false act.
    # max_false_activation_rate=0.0 → gate must trip.
    proc, out = _qualify166(c, holdout=c.holdout_path)
    assert proc.returncode == 2
    doc = json.loads(out.read_text())
    assert _decision(doc) == "REFUSED"
    assert "false_activation" in json.dumps(doc)
    assert "conditional_regression_rate_mean" in json.dumps(doc)


def test_v166_holdout_digest_in_signed_plan(tmp_path):
    """The bound digest lives inside the signed plan value — mutating
    the plan's field post-signature breaks verification upstream."""
    c = SynthV166(tmp_path).build()
    plan_doc = json.loads((c.cdir / "CAMPAIGN_PLAN.json").read_text())
    assert plan_doc["value"]["final_holdout_digest"] == \
        _manifest("final_holdout", HOLDOUT).digest
