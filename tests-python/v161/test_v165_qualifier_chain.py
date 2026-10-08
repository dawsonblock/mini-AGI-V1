"""v16.5 — end-to-end authority chain: synthetic v165 campaign evidence
through the real qualifier entrypoint.

Builds a complete on-disk campaign exactly as the runner emits it
(signed plan + protocol + partition proof + atomic per-seed bundles),
then runs scripts/validation/qualify_campaign1.py as a subprocess:

  * a clean matrix must produce a qualification-authority-signed
    QUALIFIED record (metrics recomputed from canonical predictions)
  * tampered predictions, fabricated metrics, an unauthorized plan
    signer, a revoked witness, and missing cells must each fail closed
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
from minagi.v161.campaign_plan import ColabCampaignPlanV165  # noqa: E402
from minagi.v161.dataset_manifest import (DatasetMember,  # noqa: E402
                                          DatasetMembershipManifest,
                                          DatasetPartitionSet)
from minagi.v161.evaluators import (exact_match, retention_score,  # noqa: E402
                                    security_regression)
from minagi.v161.evidence_receipt_v3 import (  # noqa: E402
    EvidenceReceiptV3, evaluation_bundle, input_manifest_digest,
    prediction_record, predictions_digest_of, write_predictions)
from minagi.v161.experiment_protocol import ExperimentProtocolV1  # noqa: E402
from minagi.v161.runtime_closure3 import sha256_path  # noqa: E402

Z = "sha256:" + "0" * 64
CID = "c-synthetic-v165"
ARMS = ("L1", "L5", "L6", "NC")
SEEDS = (0,)


def _row(rid, split, family, prompt, expected, probe=None):
    r = {"id": rid, "split": split, "family": family, "prompt": prompt,
         "expected": expected, "source": "test", "generator": "synthetic"}
    if probe:
        r["probe"] = probe
    return r


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


def _member(row):
    payload = json.dumps(row, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False).encode()
    return DatasetMember(str(row["id"]), str(row["family"]),
                         sha256_bytes(payload), str(row["source"]),
                         str(row["generator"]))


def _by_split(name):
    return [r for r in CORPUS if r["split"] == name]


def _parts():
    return DatasetPartitionSet(
        DatasetMembershipManifest("train", tuple(_member(r)
                                                 for r in _by_split("train"))),
        DatasetMembershipManifest("validation", tuple(_member(r)
                                                      for r in _by_split("validation"))),
        DatasetMembershipManifest("hidden", tuple(_member(r)
                                                  for r in _by_split("hidden"))),
        DatasetMembershipManifest("retention", tuple(_member(r)
                                                     for r in _by_split("retention"))),
        DatasetMembershipManifest("security", tuple(_member(r)
                                                    for r in _by_split("security"))),
        require_family_disjoint_hidden=True)


class SyntheticCampaign:
    """A complete v165 evidence tree, built with the same code paths the
    runner uses — predictions, bundles, receipts, commit manifests."""

    def __init__(self, tmp_path: Path):
        self.storage = tmp_path / "storage"
        self.root = tmp_path / "root"
        self.cdir = self.storage / "campaigns" / CID
        corpus_path = self.root / "corpus" / "tasks.jsonl"
        corpus_path.parent.mkdir(parents=True)
        corpus_path.write_text("\n".join(
            json.dumps(r, sort_keys=True) for r in CORPUS) + "\n")
        # the qualifier imports egai/minagi via <root>/src-python
        (self.root / "src-python").symlink_to(ROOT / "src-python")
        self.cdir.mkdir(parents=True)

        self.parts = _parts()
        self.protocol = ExperimentProtocolV1.from_config({
            "model": {"dtype": "bfloat16", "quantization": "none",
                      "trust_remote_code": False},
            "retention_scorer": "retention_score",
            "require_native_servable_adapter": False,
            "lora": {"rank": 16, "alpha": 32, "dropout": 0.05,
                     "target_modules": ["q_proj"], "learning_rate": 1e-4,
                     "steps": 100, "max_length": 256,
                     "optimizer": "adamw"},
        })
        scorer_d, retention_d, security_d = (digest("scorer"),
                                             digest("retention"),
                                             digest("security"))
        self.plan = ColabCampaignPlanV165(
            campaign_id=CID, model_id="synthetic-model",
            model_revision="rev1",
            dataset_partition_digest=self.parts.digest,
            scorer_artifact_digest=scorer_d,
            retention_artifact_digest=retention_d,
            security_artifact_digest=security_d,
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
            experiment_protocol_digest=self.protocol.digest)

        write_trust_root(self.storage / ".keys",
                         self.storage / "trust_root.json")
        self.signers = {r: Ed25519Signer.from_private_bytes(
            (self.storage / ".keys" / f"{r}.pem").read_bytes())
            for r in AUTHORITY_ROLES}
        self.registry = AuthorityRegistry.load(
            self.storage / "trust_root.json")

        # physical artifacts the receipts must close over
        self.adapter_dir = (self.storage / "adapters" / CID / "L6" /
                            "seed-0")
        self.adapter_dir.mkdir(parents=True)
        (self.adapter_dir / "adapter_model.safetensors").write_bytes(
            b"adapter-weights")
        (self.adapter_dir / "adapter_config.json").write_text("{}")
        self.adapter_d = sha256_path(self.adapter_dir)
        nc_dir = (self.storage / "adapters" / CID / "NC" / "seed-0")
        nc_dir.mkdir(parents=True)
        (nc_dir / "adapter_model.safetensors").write_bytes(b"nc-weights")
        (nc_dir / "adapter_config.json").write_text("{}")
        self.nc_adapter_d = sha256_path(nc_dir)
        self.l5_state = (self.storage / "arms" / CID / "seed-0" / "L5")
        self.l5_state.mkdir(parents=True)
        (self.l5_state / "state.json").write_text('{"replay": []}')
        self.l5_state_d = sha256_path(self.l5_state)
        self.env_d = digest({"env": "synthetic-runtime"})
        self.eval_dataset_d = digest(
            {"hidden": self.parts.hidden.digest,
             "retention": self.parts.retention.digest,
             "security": self.parts.security.digest})
        self.evaluator_set_d = digest([scorer_d, retention_d, security_d])

    # -- emission, mirroring scripts/run_campaign1.py emit() ----------
    def _preds_for(self, arm, hidden_out):
        preds = []
        hid = _by_split("hidden")
        for i, row in enumerate(hid):
            out = row["expected"] if i < hidden_out else "wrong"
            preds.append(prediction_record(row["id"], "hidden",
                                           row["prompt"], out, 3, 32))
        for row in [r for r in _by_split("retention")
                    if r.get("probe") != "delayed"]:
            preds.append(prediction_record(row["id"], "retention",
                                           row["prompt"],
                                           row["expected"], 2, 32))
        for row in _by_split("security"):
            preds.append(prediction_record(row["id"], "security",
                                           row["prompt"], "benign", 2, 32))
        return preds

    def _delayed_preds(self):
        return [prediction_record(r["id"], "retention_delayed",
                                  r["prompt"], r["expected"], 2, 32)
                for r in _by_split("retention")
                if r.get("probe") == "delayed"]

    def _metrics(self, preds):
        hid = [p for p in preds if p["block"] == "hidden"]
        ret = [p for p in preds if p["block"] == "retention"]
        sec = [p for p in preds if p["block"] == "security"]
        rows_by = {r["id"]: r for r in _by_split("hidden")}
        em = sum(exact_match(p["output"],
                             rows_by[p["id"]]["expected"])
                 for p in hid) / len(hid)
        rows_ret = {r["id"]: r for r in _by_split("retention")}
        rs = sum(retention_score(p["output"],
                                 rows_ret[p["id"]]["expected"])
                 for p in ret) / len(ret)
        ns = sum(int(security_regression(p["output"])) for p in sec)
        return {"hidden_exact_match": em, "retention": rs,
                "security_regressions": ns, "eval_seconds": 0.1,
                "generated_tokens": 10, "arm_state_bytes": 1,
                "wall_seconds": 0.2, "train_seconds": 0.0,
                "trainable_params": 0}

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
        """hidden_scores: arm -> #correct (of 4); emits committed seed."""
        run_dir = self.cdir / f"seed-{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)
        digs = {}
        adapter_ds = {"L6": self.adapter_d, "NC": self.nc_adapter_d}
        for arm in ARMS:
            preds = self._preds_for(arm, hidden_scores[arm])
            metrics = self._metrics(preds)
            ad = adapter_ds.get(arm, Z)
            sd_ = self.l5_state_d if arm == "L5" else Z
            digs[arm] = self._emit(run_dir, seed, arm, arm, preds,
                                   metrics, ad, sd_).digest
        for arm, name in (("L1", "L1_delayed"), ("L6", "L6_delayed")):
            preds = self._delayed_preds()
            metrics = {"retention_delayed": 1.0, "eval_seconds": 0.1,
                       "generated_tokens": 4}
            digs[name] = self._emit(
                run_dir, seed, arm, name, preds, metrics,
                self.adapter_d if arm == "L6" else Z, Z).digest
        sd = {"seed": seed,
              "hidden_exact_match":
                  {a: hidden_scores[a] / 4 for a in ARMS},
              "receipt_digests": digs, "attempt_id": "test-attempt"}
        (run_dir / "SEED_RESULT.json").write_text(json.dumps(sd))
        (run_dir / "COMPLETE").write_text("complete\n")
        files = {p.relative_to(run_dir).as_posix():
                 sha256_bytes(p.read_bytes())
                 for p in sorted(run_dir.rglob("*")) if p.is_file()}
        (run_dir / "COMMIT_MANIFEST.json").write_text(json.dumps(
            {"seed": seed, "campaign_digest": self.plan.digest,
             "attempt_id": "test-attempt", "files": files}))
        return sd

    def build(self, plan_signer=None):
        """Write plan-level docs + evidence; returns the witness signer
        used (for cross-signing attacks in tests)."""
        plan_signer = plan_signer or self.signers["plan"]
        env = plan_signer.sign(asdict(self.plan))
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
            plan_signer, "experiment_preregistration",
            {"value": asdict(self.plan), "digest": self.plan.digest})
        # L6 wins by 2 correct answers over L5 -> delta_ft_neural = 0.5;
        # NC scores like L1 -> zero negative-control transfer.
        for seed in SEEDS:
            self._seed(seed, {"L1": 2, "L5": 2, "L6": 4, "NC": 2})
        (self.cdir / "RESULT.json").write_text(json.dumps(
            {"decision": "QUALIFIED"}))
        return wit


def _qualify(c: SyntheticCampaign):
    out = c.storage / "QUALIFICATION_RECORD.json"
    proc = subprocess.run(
        [sys.executable,
         str(ROOT / "scripts" / "validation" / "qualify_campaign1.py"),
         "--storage", str(c.storage), "--campaign-id", CID,
         "--root", str(c.root), "--output", str(out)],
        capture_output=True, text=True)
    return proc, out


def _decision(doc):
    return doc["value"]["decision"] if "value" in doc else doc["decision"]


def test_clean_matrix_qualifies_and_signs(tmp_path):
    c = SyntheticCampaign(tmp_path)
    c.build()
    proc, out = _qualify(c)
    assert proc.returncode == 0, proc.stderr
    doc = json.loads(out.read_text())
    assert _decision(doc) == "QUALIFIED"
    # qualification record signed by the qualification authority
    assert c.registry.is_authorized("qualification",
                                    doc["signer_key_id"])
    # evaluation bundle signed by the evaluation authority
    bundle = json.loads((c.cdir / "EVALUATION_BUNDLE.json").read_text())
    assert c.registry.is_authorized("evaluation",
                                    bundle["signer_key_id"])
    assert doc["value"]["scoring"] == \
        "independently-recomputed-from-predictions"
    # both records land in the authority ledger, chaining from the
    # preregistration the runner wrote
    led = AuthorityLedger(c.storage / "AUTHORITY_LEDGER.jsonl")
    assert led.verify(c.registry) == []
    kinds = [json.loads(l)["kind"]
             for l in (c.storage / "AUTHORITY_LEDGER.jsonl")
             .read_text().splitlines() if l.strip()]
    assert kinds == ["experiment_preregistration", "evaluation_bundle",
                     "qualification_record"]


def test_tampered_prediction_fails_closed(tmp_path):
    c = SyntheticCampaign(tmp_path)
    c.build()
    p = c.cdir / "seed-0" / "PREDICTIONS-L6.jsonl"
    recs = [json.loads(l) for l in p.read_text().splitlines()]
    recs[0]["output"] = "fabricated-right-answer"
    p.write_text("\n".join(json.dumps(r) for r in recs) + "\n")
    proc, out = _qualify(c)
    assert proc.returncode == 2
    assert _decision(json.loads(out.read_text())) == "INVALID_EVIDENCE"


def test_fabricated_reported_metric_fails(tmp_path):
    c = SyntheticCampaign(tmp_path)
    c.build()
    cell = c.cdir / "seed-0" / "L5.json"
    doc = json.loads(cell.read_text())
    doc["bundle"]["metrics_reported"]["hidden_exact_match"] = 1.0
    cell.write_text(json.dumps(doc))
    proc, out = _qualify(c)
    assert proc.returncode == 2
    d = _decision(json.loads(out.read_text()))
    assert d == "INVALID_EVIDENCE"


def test_missing_cell_is_incomplete_not_qualified(tmp_path):
    c = SyntheticCampaign(tmp_path)
    c.build()
    (c.cdir / "seed-0" / "L5.json").unlink()
    proc, out = _qualify(c)
    assert proc.returncode == 2
    assert _decision(json.loads(out.read_text())) == "INCOMPLETE"


def test_unauthorized_plan_signer_rejected(tmp_path):
    c = SyntheticCampaign(tmp_path)
    # execution_witness signing the plan — role confusion must fail
    c.build(plan_signer=c.signers["execution_witness"])
    proc, out = _qualify(c)
    assert proc.returncode == 2
    assert _decision(json.loads(out.read_text())) == "INVALID_EVIDENCE"


def test_rogue_key_plan_rejected(tmp_path):
    c = SyntheticCampaign(tmp_path)
    rogue = Ed25519Signer.generate()
    c.build(plan_signer=rogue)
    proc, out = _qualify(c)
    assert proc.returncode == 2
    assert _decision(json.loads(out.read_text())) == "INVALID_EVIDENCE"


def test_off_corpus_prediction_id_fails(tmp_path):
    """A prediction record pointing at a task id outside the committed
    corpus partition cannot be smuggled into the score."""
    c = SyntheticCampaign(tmp_path)
    c.build()
    p = c.cdir / "seed-0" / "PREDICTIONS-L6.jsonl"
    recs = [json.loads(l) for l in p.read_text().splitlines()]
    recs[0] = prediction_record("injected-task", "hidden", "hp0",
                                "ans0", 3, 32)
    # keep the receipt-consistent digests intact by re-signing with the
    # witness key — a compromised worker can do this; the qualifier's
    # corpus join must still catch it
    p.write_text("".join(json.dumps(r, sort_keys=True) + "\n"
                         for r in recs))
    cell = json.loads((c.cdir / "seed-0" / "L6.json").read_text())
    im = input_manifest_digest(recs)
    pd = predictions_digest_of(recs)
    bundle = evaluation_bundle(c.plan.digest, c.protocol.digest, 0, "L6",
                               "L6", im, pd,
                               cell["bundle"]["metrics_reported"])
    r = EvidenceReceiptV3.sign(
        signer=c.signers["execution_witness"],
        campaign_digest=c.plan.digest,
        protocol_digest=c.protocol.digest, seed=0, arm="L6",
        model_digest=c.plan.model_digest,
        tokenizer_digest=c.plan.tokenizer_digest,
        adapter_digest=c.adapter_d, state_digest=Z,
        environment_digest=c.env_d,
        evaluation_dataset_digest=c.eval_dataset_d,
        evaluator_digest=c.evaluator_set_d,
        input_manifest_digest=im, predictions_digest=pd,
        evaluation_bundle_digest=digest(bundle))
    (c.cdir / "seed-0" / "L6.json").write_text(json.dumps(
        {"receipt": asdict(r), "bundle": bundle}))
    proc, out = _qualify(c)
    assert proc.returncode == 2
    assert _decision(json.loads(out.read_text())) == "INVALID_EVIDENCE"


def test_missing_trust_root_fails_closed(tmp_path):
    c = SyntheticCampaign(tmp_path)
    c.build()
    (c.storage / "trust_root.json").unlink()
    proc, out = _qualify(c)
    assert proc.returncode == 2
    assert _decision(json.loads(out.read_text())) == "INVALID_EVIDENCE"


def test_revoked_witness_key_rejected(tmp_path):
    c = SyntheticCampaign(tmp_path)
    c.build()
    # revoke the witness in the trust root — old receipts must stop
    # qualifying even though the signature itself still verifies
    doc = json.loads((c.storage / "trust_root.json").read_text())
    for a in doc["authorities"]:
        if a["role"] == "execution_witness":
            a["status"] = "revoked"
    (c.storage / "trust_root.json").write_text(json.dumps(doc))
    proc, out = _qualify(c)
    assert proc.returncode == 2
    assert _decision(json.loads(out.read_text())) == "INVALID_EVIDENCE"
