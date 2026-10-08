"""Phase 7/8 — adaptive plasticity + governed recursive generations.

Plasticity: cheapest-first mechanism ladder enforced structurally —
a weight-adaptation proposal requires recorded attempts at retrieval
and skill compilation first; nothing here authorizes execution.

Generations: G(n+1) may only be preregistered after the promotion
authority signs off on G(n); fresh evaluation corpora per generation.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src-python"))

from egai.common.canonical import digest  # noqa: E402
from minagi.v161.generations import (GenerationChain,  # noqa: E402
                                     GenerationRecord)
from minagi.v161.plasticity import (DynamicLoraPolicyV1,  # noqa: E402
                                    FailureEvidence, FailureKind,
                                    Mechanism, PlasticityProposal,
                                    RankAllocator, diagnose, propose)

D = lambda s: digest(s)  # noqa: E731


def _ev(**kw):
    base = dict(task_digest=D("task"), failure_signature=D("sig"),
                retrieval_found=False, retrieval_helped=False,
                repeats_prior_signature=False, harness_error=False)
    base.update(kw)
    return FailureEvidence(**base)


# ---------- diagnosis ladder ----------------------------------------

def test_harness_error_outranks_everything():
    k, m = diagnose(_ev(harness_error=True, retrieval_found=True,
                        retrieval_helped=True))
    assert (k, m) == (FailureKind.HARNESS_DEFECT, Mechanism.HARNESS)


def test_helpful_retrieval_stays_cheapest():
    k, m = diagnose(_ev(retrieval_found=True, retrieval_helped=True))
    assert (k, m) == (FailureKind.MISSING_INFORMATION, Mechanism.RETRIEVAL)


def test_repeated_unhelped_signature_becomes_skill():
    k, m = diagnose(_ev(retrieval_found=True, retrieval_helped=False,
                        repeats_prior_signature=True))
    assert (k, m) == (FailureKind.REPEATABLE_PROCEDURE, Mechanism.SKILL)


def test_persistent_deficiency_escalates_to_weights():
    k, m = diagnose(_ev(retrieval_found=True, retrieval_helped=False,
                        repeats_prior_signature=False))
    assert (k, m) == (FailureKind.PERSISTENT_DEFICIENCY, Mechanism.WEIGHTS)


def test_diagnosis_deterministic():
    e = _ev(repeats_prior_signature=True)
    assert diagnose(e) == diagnose(e)


# ---------- proposals enforce the ladder -----------------------------

def test_weights_proposal_requires_prior_attempts():
    ev = _ev()  # diagnoses to WEIGHTS
    with pytest.raises(ValueError, match="prior cheaper"):
        propose(ev, proposal_id="p1",
                candidate_config_digest=D("cfg"),
                baseline_digest=D("base"))


def test_valid_weights_proposal_binds_attempts():
    ev = _ev()
    p = propose(ev, proposal_id="p1",
                candidate_config_digest=D("cfg"),
                baseline_digest=D("base"),
                prior_attempts=(D("retrieval-attempt"), D("skill-attempt")))
    assert p.mechanism == Mechanism.WEIGHTS.value
    assert len(p.prior_attempts) == 2  # retrieval + skill rungs


def test_retrieval_proposal_needs_no_prior():
    ev = _ev(retrieval_found=True, retrieval_helped=True)
    p = propose(ev, proposal_id="p0",
                candidate_config_digest=D("cfg"), baseline_digest=D("b"))
    assert p.mechanism == Mechanism.RETRIEVAL.value
    assert p.digest != ""


# ---------- dynamic-rank policy --------------------------------------

POLICY = DynamicLoraPolicyV1(base_rank=4, max_rank=16, rank_growth_step=4,
                             per_task_rank_budget=12, total_rank_budget=24)


def test_policy_validation():
    with pytest.raises(ValueError, match="base_rank"):
        DynamicLoraPolicyV1(base_rank=0, max_rank=8, rank_growth_step=2,
                            per_task_rank_budget=4, total_rank_budget=8)
    with pytest.raises(ValueError, match="per-task"):
        DynamicLoraPolicyV1(base_rank=4, max_rank=8, rank_growth_step=2,
                            per_task_rank_budget=16, total_rank_budget=32)


def test_allocator_respects_budgets():
    ra = RankAllocator(POLICY)
    assert ra.allocate("t1") == 4                # base rank
    assert ra.allocate("t1") == 4                # idempotent
    assert ra.allocate("t2", requested=12) == 12  # at per-task cap
    assert ra.allocate("t3", requested=16) == 8   # total budget: 24-16=8
    with pytest.raises(RuntimeError, match="exhausted"):
        ra.allocate("t4")


def test_growth_clamped_and_protected():
    ra = RankAllocator(POLICY)
    ra.allocate("t1")
    # activation breach blocks growth under protection policy
    assert ra.grow("t1", false_activation_breach=True) == 4
    assert ra.grow("t1") == 8
    assert ra.grow("t1") == 12  # per-task cap reached
    assert ra.grow("t1") == 12  # no further growth


# ---------- generation chain -----------------------------------------

def _gen(i, parent_dig="", corpus_tag=None, promo=""):
    return GenerationRecord(
        generation=i, campaign_digest=D(f"camp{i}"),
        corpus_digest=D(f"corpus-{corpus_tag or i}"),
        control_arm_digest=D(f"ctrl{i}"),
        parent_generation_digest=parent_dig,
        promotion_decision_digest=promo)


def test_genesis_then_promotion_gated_child():
    chain = GenerationChain()
    chain.append(_gen(0))
    # child cannot be preregistered before parent promotion
    with pytest.raises(PermissionError, match="not promoted"):
        chain.append(_gen(1, parent_dig=chain.tip.digest))
    chain.record_promotion(0, D("promotion-decision-0"))
    chain.append(_gen(1, parent_dig=chain.tip.digest))
    assert len(chain.records) == 2
    assert chain.verify() == []


def test_corpus_reuse_rejected():
    chain = GenerationChain()
    chain.append(_gen(0))
    chain.record_promotion(0, D("pd0"))
    with pytest.raises(ValueError, match="corpus reuse"):
        chain.append(_gen(1, parent_dig=chain.tip.digest, corpus_tag="0"))


def test_wrong_parent_digest_rejected():
    chain = GenerationChain()
    chain.append(_gen(0))
    chain.record_promotion(0, D("pd0"))
    with pytest.raises(ValueError, match="parent digest"):
        chain.append(_gen(1, parent_dig=D("wrong-parent")))


def test_skipped_generation_rejected():
    chain = GenerationChain()
    chain.append(_gen(0))
    chain.record_promotion(0, D("pd0"))
    chain.append(_gen(1, parent_dig=chain.tip.digest))
    with pytest.raises(ValueError, match="next generation"):
        chain.append(_gen(3, parent_dig=chain.tip.digest))


def test_promotion_only_at_tip():
    chain = GenerationChain()
    chain.append(_gen(0))
    chain.record_promotion(0, D("pd0"))
    chain.append(_gen(1, parent_dig=chain.tip.digest))
    # non-tip promotion refused
    with pytest.raises(ValueError, match="chain tip"):
        chain.record_promotion(0, D("pd0-again"))
    # the parent's record digest is pinned by the child's linkage — the
    # parent object itself is unchanged by the child's promotion state
    g0_digest = chain.records[0].digest
    chain.record_promotion(1, D("pd1"))
    assert chain.records[0].digest == g0_digest
    # double-promotion refused
    with pytest.raises(ValueError, match="already promoted"):
        chain.record_promotion(1, D("pd1-again"))
    assert chain.verify() == []


def test_verify_catches_tampered_sequence():
    """A forged chain (constructed out of band) must not verify."""
    chain = GenerationChain()
    bad = GenerationRecord(generation=1, campaign_digest=D("c"),
                           corpus_digest=D("k"), control_arm_digest=D("x"),
                           parent_generation_digest=D("nobody"))
    chain._records.append(bad)  # bypass append to simulate grafting
    assert chain.verify() != []
