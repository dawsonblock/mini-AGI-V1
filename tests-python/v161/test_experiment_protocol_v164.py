"""v16.4 — experiment protocol binding, physical identity, and
verified-resume tests."""
import json
from dataclasses import asdict

import pytest

from egai.common.canonical import digest
from egai.common.crypto import Ed25519Signer, Ed25519Verifier
from minagi.v15.native_adapter import native_adapter_supports_target
from minagi.v161.campaign_plan import (ColabCampaignPlanV163,
                                       ColabCampaignPlanV164)
from minagi.v161.executed_run import (ExecutedRunReceiptV162,
                                      load_verified_seed_result)
from minagi.v161.experiment_protocol import ExperimentProtocolV1

D = digest({"x": 1})
Z = "sha256:" + "0" * 64

CFG = {
    "campaign_id": "c-v164",
    "model": {"id": "m", "revision": "r", "dtype": "bfloat16",
              "quantization": "none", "trust_remote_code": False},
    "seeds": [0, 1, 2, 3, 4],
    "arms": ["L1", "L2", "L3", "L4", "L5", "L6", "NC"],
    "max_new_tokens": 32,
    "practice_samples": 8,
    "retrieval_k": 3,
    "memory_k": 2,
    "replay_k": 2,
    "retention_samples": 12,
    "retention_scorer": "containment_match",
    "lora": {"rank": 16, "alpha": 32, "dropout": 0.05,
             "target_modules": ["lm_head"], "learning_rate": 1e-4,
             "steps": 100, "max_length": 256},
}


def _cfg(**kw):
    cfg = json.loads(json.dumps(CFG))
    cfg.update(kw)
    return cfg


def _lora(**kw):
    cfg = _cfg()
    cfg["lora"].update(kw)
    return cfg


# ---- ExperimentProtocolV1 ----

def test_protocol_binds_learning_hyperparams():
    base = ExperimentProtocolV1.from_config(CFG).digest
    for alt in (_lora(rank=8), _lora(learning_rate=1e-3),
                _lora(target_modules=["lm_head", "q_proj"]),
                _lora(steps=50), _lora(max_length=128),
                _lora(dropout=0.1)):
        assert ExperimentProtocolV1.from_config(alt).digest != base


def test_protocol_binds_memory_and_generation():
    base = ExperimentProtocolV1.from_config(CFG).digest
    for alt in (_cfg(retrieval_k=4), _cfg(memory_k=3), _cfg(replay_k=5),
                _cfg(practice_samples=16), _cfg(retention_samples=8),
                _cfg(max_new_tokens=64),
                _cfg(retention_scorer="retention_score")):
        assert ExperimentProtocolV1.from_config(alt).digest != base
    m = _cfg(); m["model"]["dtype"] = "float32"
    assert ExperimentProtocolV1.from_config(m).digest != base
    m = _cfg(); m["model"]["quantization"] = "4bit"
    assert ExperimentProtocolV1.from_config(m).digest != base


def test_protocol_requires_full_lora_block():
    cfg = _cfg()
    del cfg["lora"]["target_modules"]
    with pytest.raises(ValueError, match="missing required keys"):
        ExperimentProtocolV1.from_config(cfg)


def test_protocol_validation():
    with pytest.raises(ValueError):
        ExperimentProtocolV1.from_config(_lora(rank=0))
    with pytest.raises(ValueError):
        ExperimentProtocolV1.from_config(_lora(target_modules=[]))
    with pytest.raises(ValueError):
        ExperimentProtocolV1.from_config(_lora(learning_rate=0))
    with pytest.raises(ValueError):
        ExperimentProtocolV1.from_config(_cfg(retention_scorer="bogus"))
    p = ExperimentProtocolV1.from_config(CFG)
    with pytest.raises(ValueError):
        ExperimentProtocolV1(**{**asdict(p), "decoding": "sample"})


def test_protocol_lora_spec_kwargs():
    p = ExperimentProtocolV1.from_config(CFG)
    kw = p.lora_train_spec_kwargs()
    assert kw["target_modules"] == ("lm_head",)
    assert kw["rank"] == 16 and kw["steps"] == 100


# ---- ColabCampaignPlanV164 ----

def _plan164(**kw):
    base = dict(campaign_id="c1b", model_id="m", model_revision="r",
                dataset_partition_digest=D, scorer_artifact_digest=D,
                retention_artifact_digest=D, security_artifact_digest=D,
                seeds=(0, 1, 2, 3, 4),
                model_digest=D, tokenizer_digest=D,
                generation_template_digest=D,
                experiment_protocol_digest=D)
    base.update(kw)
    return ColabCampaignPlanV164(**base)


def test_plan_v164_binds_protocol():
    p = _plan164()
    assert p.schema == "mini-agi-v16.4-colab-campaign-plan-v1"
    assert p.experiment_protocol_digest == D
    assert _plan164().digest != _plan164(
        experiment_protocol_digest=digest({"y": 2})).digest


def test_plan_v164_requires_protocol_digest():
    with pytest.raises(ValueError):
        _plan164(experiment_protocol_digest="")
    with pytest.raises(ValueError):
        _plan164(experiment_protocol_digest="deadbeef")


def test_plan_v164_inherits_v163_rules():
    with pytest.raises(ValueError):
        _plan164(min_seeds_positive_ft=6)
    with pytest.raises(ValueError):
        _plan164(model_digest="")
    assert isinstance(_plan164(), ColabCampaignPlanV163)


# ---- verified seed resume ----

ARMS = ("L1", "L6")


def _seed_dir(tmp_path, seed=0, campaign_digest=None, signer=None):
    signer = signer or Ed25519Signer.generate("w")
    campaign_digest = campaign_digest or D
    run_dir = tmp_path / f"seed-{seed}"
    run_dir.mkdir(parents=True)
    rec_digests, hidden = {}, {}
    for arm in ARMS:
        r = ExecutedRunReceiptV162.sign(
            signer=signer, campaign_digest=campaign_digest, arm=arm,
            seed=seed, environment_digest=D, model_digest=D,
            tokenizer_digest=D, adapter_digest=Z, state_digest=Z,
            dataset_digest=D, evaluator_digest=D,
            metrics={"hidden_exact_match": 0.5})
        rec_digests[arm] = r.digest
        hidden[arm] = 0.5
        (run_dir / f"{arm}.json").write_text(json.dumps(
            {"receipt": asdict(r), "outputs": {}}))
    (run_dir / "SEED_RESULT.json").write_text(json.dumps(
        {"seed": seed, "hidden_exact_match": hidden,
         "receipt_digests": rec_digests}))
    (run_dir / "COMPLETE").write_text("complete\n")
    return run_dir, signer


def _verifier(signer):
    v = Ed25519Verifier()
    v.register(signer.key_id, signer.public_bytes())
    return v


def test_seed_evidence_accepted(tmp_path):
    run_dir, signer = _seed_dir(tmp_path)
    sd = load_verified_seed_result(run_dir, 0, D, ARMS, _verifier(signer))
    assert sd is not None and sd["seed"] == 0


def test_seed_evidence_rejects_tampered_receipt(tmp_path):
    run_dir, signer = _seed_dir(tmp_path)
    doc = json.loads((run_dir / "L1.json").read_text())
    doc["receipt"]["metrics"]["hidden_exact_match"] = 0.99
    (run_dir / "L1.json").write_text(json.dumps(doc))
    assert load_verified_seed_result(run_dir, 0, D, ARMS,
                                     _verifier(signer)) is None


def test_seed_evidence_rejects_wrong_campaign(tmp_path):
    run_dir, signer = _seed_dir(tmp_path,
                                campaign_digest=digest({"other": "plan"}))
    assert load_verified_seed_result(run_dir, 0, D, ARMS,
                                     _verifier(signer)) is None


def test_seed_evidence_rejects_wrong_key(tmp_path):
    run_dir, signer = _seed_dir(tmp_path)
    other = Ed25519Signer.generate("forged")
    assert load_verified_seed_result(run_dir, 0, D, ARMS,
                                     _verifier(other)) is None


def test_seed_evidence_rejects_incomplete_matrix(tmp_path):
    run_dir, signer = _seed_dir(tmp_path)
    (run_dir / "L6.json").unlink()
    assert load_verified_seed_result(run_dir, 0, D, ARMS,
                                     _verifier(signer)) is None


def test_seed_evidence_rejects_wrong_seed_label(tmp_path):
    run_dir, signer = _seed_dir(tmp_path)
    sd = json.loads((run_dir / "SEED_RESULT.json").read_text())
    sd["seed"] = 7
    (run_dir / "SEED_RESULT.json").write_text(json.dumps(sd))
    assert load_verified_seed_result(run_dir, 0, D, ARMS,
                                     _verifier(signer)) is None


def test_seed_evidence_rejects_missing_complete(tmp_path):
    run_dir, signer = _seed_dir(tmp_path)
    (run_dir / "COMPLETE").unlink()
    assert load_verified_seed_result(run_dir, 0, D, ARMS,
                                     _verifier(signer)) is None


def test_seed_evidence_rejects_metric_disagreement(tmp_path):
    run_dir, signer = _seed_dir(tmp_path)
    sd = json.loads((run_dir / "SEED_RESULT.json").read_text())
    sd["hidden_exact_match"]["L1"] = 0.9
    (run_dir / "SEED_RESULT.json").write_text(json.dumps(sd))
    assert load_verified_seed_result(run_dir, 0, D, ARMS,
                                     _verifier(signer)) is None


# ---- witness key persistence ----

def test_witness_key_persistence_roundtrip():
    s1 = Ed25519Signer.generate("witness")
    s2 = Ed25519Signer.from_private_bytes(s1.private_bytes(), "witness")
    assert s2.key_id == s1.key_id
    assert s2.public_bytes() == s1.public_bytes()
    r = ExecutedRunReceiptV162.sign(
        signer=s1, campaign_digest=D, arm="L1", seed=0,
        environment_digest=D, model_digest=D, tokenizer_digest=D,
        adapter_digest=Z, state_digest=Z, dataset_digest=D,
        evaluator_digest=D, metrics={})
    assert r.verify(_verifier(s2))


# ---- lane-mode seed subset parsing ----

def _parse(raw, planned=(0, 1, 2, 3, 4)):
    import sys
    from pathlib import Path
    scripts = str(Path(__file__).resolve().parents[2] / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    from run_campaign1 import parse_seed_subset
    return parse_seed_subset(raw, planned)


def test_seed_subset_valid():
    assert _parse("0,2,4") == frozenset({0, 2, 4})
    assert _parse("3") == frozenset({3})
    assert _parse(" 0 , 1 ") == frozenset({0, 1})


def test_seed_subset_rejects_unplanned():
    with pytest.raises(SystemExit):
        _parse("0,1,99")
    with pytest.raises(SystemExit):
        _parse("")
    with pytest.raises(SystemExit):
        _parse("a,b")


# ---- native adapter target gate ----

def test_native_adapter_target_gate():
    assert native_adapter_supports_target("lm_head")
    assert native_adapter_supports_target("output.weight")
    assert not native_adapter_supports_target("q_proj")
    assert not native_adapter_supports_target("k_proj")
    assert not native_adapter_supports_target("v_proj")
    assert not native_adapter_supports_target("o_proj")
    assert not native_adapter_supports_target("gate_proj")
