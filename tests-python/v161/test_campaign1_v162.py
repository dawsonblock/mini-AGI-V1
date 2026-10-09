
import pytest

from egai.common.canonical import digest
from egai.common.crypto import Ed25519Signer, Ed25519Verifier
from minagi.v161.arms import (FrozenArm, GroundedReplayArm, RetrievalArm,
                              SemanticMemoryArm, SkillsArm,
                              shuffled_label_texts)
from minagi.v161.campaign_plan import (ColabCampaignPlanV162,
                                       ColabCampaignPlanV163,
                                       ColabCampaignPlanV164)
from minagi.v161.evaluators import containment_match
from minagi.v161.executed_run import (ExecutedRunReceiptV161,
                                      ExecutedRunReceiptV162)

D = digest({"x": 1})
Z = "sha256:" + "0" * 64

ROWS = [
    {"id": "t1", "family": "f-a", "split": "train",
     "prompt": "Task: Reverse the characters of the input word.\nInput: cobalt\nAnswer:",
     "expected": "tlaboc"},
    {"id": "t2", "family": "f-a", "split": "train",
     "prompt": "Task: Reverse the characters of the input word.\nInput: amber\nAnswer:",
     "expected": "rebma"},
    {"id": "t3", "family": "f-b", "split": "train",
     "prompt": "Task: Convert the input word to uppercase.\nInput: jade\nAnswer:",
     "expected": "JADE"},
    {"id": "h1", "family": "f-z", "split": "hidden",
     "prompt": "Task: Sort the characters of the input word alphabetically.\nInput: quartz\nAnswer:",
     "expected": "aqrtuz"},
]


def _plan(**kw):
    base = dict(campaign_id="c", model_id="m", model_revision="r",
                dataset_partition_digest=D, scorer_artifact_digest=D,
                retention_artifact_digest=D, security_artifact_digest=D,
                seeds=(0, 1, 2))
    base.update(kw)
    return ColabCampaignPlanV162(**base)


def test_plan_v162_defaults_and_digest():
    p = _plan()
    assert p.arms == ("L1", "L2", "L3", "L4", "L5", "L6", "NC")
    assert p.reproduction_semantics == "statistical-equivalence"
    assert p.adapter_bitwise_required is False
    assert p.digest == _plan().digest


def test_plan_v162_rejects_bitwise_required():
    with pytest.raises(ValueError):
        _plan(adapter_bitwise_required=True)


def test_plan_v162_rejects_missing_required_arms():
    with pytest.raises(ValueError):
        _plan(arms=("L1", "L2"))
    with pytest.raises(ValueError):
        _plan(arms=("L1", "L5", "L6"))  # NC missing


def test_plan_v162_rejects_unknown_arm():
    with pytest.raises(ValueError):
        _plan(arms=("L1", "L5", "L6", "NC", "XX"))


def test_receipt_v162_sign_verify_all_arms():
    s = Ed25519Signer.generate("executor")
    v = Ed25519Verifier()
    v.register(s.key_id, s.public_bytes())
    for arm in ("L1", "L2", "L3", "L4", "L5", "L6", "NC"):
        r = ExecutedRunReceiptV162.sign(
            signer=s, campaign_digest=D, arm=arm, seed=0,
            environment_digest=D, model_digest=D, tokenizer_digest=D,
            adapter_digest=Z, state_digest=Z, dataset_digest=D,
            evaluator_digest=D, metrics={"hidden_exact_match": 0.5})
        assert r.verify(v)


def test_receipt_v162_rejects_v161_only_arms_still_valid():
    s = Ed25519Signer.generate("executor")
    v = Ed25519Verifier()
    v.register(s.key_id, s.public_bytes())
    with pytest.raises(ValueError):
        ExecutedRunReceiptV162.sign(
            signer=s, campaign_digest=D, arm="A0", seed=0,
            environment_digest=D, model_digest=D, tokenizer_digest=D,
            adapter_digest=Z, state_digest=Z, dataset_digest=D,
            evaluator_digest=D, metrics={})
    # v161 receipt untouched: still signs A0/A1 and verifies
    r = ExecutedRunReceiptV161.sign(
        signer=s, campaign_digest=D, arm="A0", seed=0,
        environment_digest=D, model_digest=D, tokenizer_digest=D,
        adapter_digest=Z, dataset_digest=D, evaluator_digest=D,
        metrics={"score": 1.0})
    assert r.verify(v)


def test_receipt_v162_state_digest_bound():
    s = Ed25519Signer.generate("e")
    v = Ed25519Verifier()
    v.register(s.key_id, s.public_bytes())
    r = ExecutedRunReceiptV162.sign(
        signer=s, campaign_digest=D, arm="L2", seed=0,
        environment_digest=D, model_digest=D, tokenizer_digest=D,
        adapter_digest=Z, state_digest=D, dataset_digest=D,
        evaluator_digest=D, metrics={})
    tampered = ExecutedRunReceiptV162(
        campaign_digest=D, arm="L2", seed=0, environment_digest=D,
        model_digest=D, tokenizer_digest=D, adapter_digest=Z,
        state_digest=Z, dataset_digest=D, evaluator_digest=D,
        metrics={}, signer_key_id=r.signer_key_id,
        signature_b64=r.signature_b64)
    assert not tampered.verify(v)


def test_retrieval_arm_deterministic(tmp_path):
    a, b = RetrievalArm(k=2), RetrievalArm(k=2)
    a.build_state(tmp_path / "a", ROWS[:3])
    b.build_state(tmp_path / "b", ROWS[:3])
    out_a = a.augment(ROWS[3]["prompt"], "f-z")
    out_b = b.augment(ROWS[3]["prompt"], "f-z")
    assert out_a == out_b
    assert "Examples:" in out_a and ROWS[3]["prompt"] in out_a
    assert (tmp_path / "a" / "INDEX.json").read_bytes() == \
           (tmp_path / "b" / "INDEX.json").read_bytes()


def test_memory_arm_consolidates_per_family(tmp_path):
    arm = SemanticMemoryArm()
    arm.build_state(tmp_path, ROWS[:3])
    assert len(arm.entries) == 2  # f-a x2, f-b x1
    out = arm.augment(ROWS[3]["prompt"], "f-z")
    assert "Memory[" in out


def test_skills_arm_whole_family_block(tmp_path):
    arm = SkillsArm()
    arm.build_state(tmp_path, ROWS[:3])
    out = arm.augment(ROWS[3]["prompt"], "f-z")
    assert "Skill: skill-" in out


def test_replay_arm_never_sees_hidden_labels(tmp_path):
    arm = GroundedReplayArm(k=1)
    arm.build_state(tmp_path, ROWS[:3])
    # wrong predictions are never replayed
    arm.add_trace(ROWS[0], "wrong", 0.0)
    assert arm.augment(ROWS[3]["prompt"], "f-z") == ROWS[3]["prompt"]
    arm.add_trace(ROWS[0], "tlaboc", 1.0)
    out = arm.augment(ROWS[3]["prompt"], "f-z")
    assert "Verified experience" in out and "tlaboc" in out
    arm.persist()
    assert (tmp_path / "REPLAY.json").is_file()


def test_frozen_arm_identity():
    assert FrozenArm().augment("p", "f") == "p"


def test_shuffled_labels_permute_deterministically():
    a = shuffled_label_texts(ROWS[:3], seed=7)
    b = shuffled_label_texts(ROWS[:3], seed=7)
    assert a == b
    # answers are drawn from the same set but permuted
    originals = {r["expected"] for r in ROWS[:3]}
    for text in a:
        assert any(text.endswith(" " + e) for e in originals)


def test_shuffled_labels_strict_derangement():
    # a negative control must corrupt ALL supervision — no fixed points
    for seed in range(20):
        shuffled = shuffled_label_texts(ROWS[:3], seed=seed)
        for i, row in enumerate(ROWS[:3]):
            assert not shuffled[i].endswith(" " + str(row["expected"])), \
                f"seed {seed}: row {i} kept its own label"


# ---- v16.3 / Campaign 1b ----

def test_containment_match_prose_and_traps():
    assert containment_match("The capital of France is Paris.", "Paris") == 1.0
    assert containment_match("paris", "Paris") == 1.0
    assert containment_match("The answer is 17.", "7") == 0.0
    assert containment_match("There are 37 days", "7") == 0.0
    assert containment_match("It is 1000 degrees.", "100") == 0.0
    assert containment_match("", "Paris") == 0.0
    assert containment_match("London", "Paris") == 0.0


def _plan163(**kw):
    base = dict(campaign_id="c1b", model_id="m", model_revision="r",
                dataset_partition_digest=D, scorer_artifact_digest=D,
                retention_artifact_digest=D, security_artifact_digest=D,
                seeds=(0, 1, 2, 3, 4),
                model_digest=D, tokenizer_digest=D,
                generation_template_digest=D)
    base.update(kw)
    return ColabCampaignPlanV163(**base)


def test_plan_v163_binds_identity_digests():
    p = _plan163()
    assert p.schema == "mini-agi-v16.3-colab-campaign-plan-v1"
    assert p.model_digest == D
    assert p.security_min_pass_rate == 0.5
    assert p.security_max_drop_vs_L1 == 0.10
    assert p.retention_max_drop == 0.10
    assert p.min_seeds_positive_ft == 4
    assert p.digest == _plan163().digest


def test_plan_v163_rejects_bad_bounds():
    with pytest.raises(ValueError):
        _plan163(security_min_pass_rate=1.5)
    with pytest.raises(ValueError):
        _plan163(retention_max_drop=-0.1)
    with pytest.raises(ValueError):
        _plan163(min_seeds_positive_ft=6)  # > n_seeds
    with pytest.raises(ValueError):
        _plan163(model_digest="")  # missing identity binding


def test_plan_v163_fields_change_digest():
    # identity binding is part of the signed surface
    assert _plan163().digest != _plan163(tokenizer_digest=digest({"y": 2})).digest


def _plan164(**kw):
    base = dict(campaign_id="c2", model_id="m", model_revision="r",
                dataset_partition_digest=D, scorer_artifact_digest=D,
                retention_artifact_digest=D, security_artifact_digest=D,
                seeds=tuple(range(10)),
                model_digest=D, tokenizer_digest=D,
                generation_template_digest=D)
    base.update(kw)
    return ColabCampaignPlanV164(**base)


def test_plan_v164_defaults_and_schema():
    p = _plan164()
    assert p.schema == "mini-agi-v16.4-colab-campaign-plan-v1"
    assert p.bootstrap_resamples == 20000
    assert p.ci_alpha == 0.05
    assert p.min_delta_ft_ci_lower == 0.0
    assert p.delayed_retention_max_drop == 0.10
    assert p.dataset_path == "configs/campaign2_tasks.jsonl"
    assert p.require_family_disjoint_hidden is True
    assert p.digest == _plan164().digest


def test_plan_v164_rejects_bad_bootstrap():
    with pytest.raises(ValueError):
        _plan164(bootstrap_resamples=100)
    with pytest.raises(ValueError):
        _plan164(ci_alpha=0.9)
    with pytest.raises(ValueError):
        _plan164(delayed_retention_max_drop=-0.1)
    with pytest.raises(ValueError):
        _plan164(dataset_path="../escape.jsonl")


def test_plan_v164_binds_dataset_path():
    assert _plan164().digest != _plan164(dataset_path="configs/other.jsonl").digest
    assert _plan164().digest != _plan164(bootstrap_resamples=10000).digest


def test_bootstrap_ci_deterministic_and_sane():
    from minagi.v161.stats import bootstrap_ci
    xs = [0.05, 0.12, 0.08, 0.2, 0.15, 0.09, 0.11, 0.18, 0.07, 0.14]
    a = bootstrap_ci(xs, 5000, 0.05)
    b = bootstrap_ci(xs, 5000, 0.05)
    assert a == b  # deterministic
    assert a["lower"] > 0.0 and a["upper"] < 1.0
    assert a["mean"] == pytest.approx(sum(xs) / len(xs))
    # a vector straddling zero must NOT clear a >0 lower-bound gate
    mixed = bootstrap_ci([-0.3, 0.4, -0.2, 0.5, 0.1, -0.1, 0.3, -0.4, 0.2, 0.0],
                         5000, 0.05)
    assert mixed["lower"] < 0.0
    with pytest.raises(ValueError):
        bootstrap_ci([], 1000, 0.05)
