"""Phase 7/8 — adaptive plasticity + governed recursive generations.

Plasticity: cheapest-first mechanism ladder enforced structurally —
a weight-adaptation proposal requires recorded attempts at retrieval
and skill compilation first; nothing here authorizes execution.

Generations: G(n+1) may only be preregistered after the promotion
authority signs off on G(n); fresh evaluation corpora per generation.
FIX-003: promotion is an append-only event validated against the
independent promotion authority — a bare digest, a forged signature, an
expired or revoked decision, or a decision bound to a different
record/campaign never promotes anything.
"""
import base64
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src-python"))

from egai.common.canonical import digest  # noqa: E402
from egai.common.crypto import Ed25519Signer  # noqa: E402
from minagi.v161.authority import (AuthorityRecord,  # noqa: E402
                                   AuthorityRegistry)
from minagi.v161.generations import (PROMOTION_DECISION_SCHEMA,  # noqa: E402
                                     GenerationChain, GenerationRecord)
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


# ---------- FIX-002: growth arithmetic --------------------------------

def test_growth_cannot_decrease_rank_when_budget_exhausted():
    """Reproduces the FIX-002 defect: with the total budget fully spent
    the original grow() clamped to `total - spent == 0`, driving the
    rank to zero. Growth must be blocked instead."""
    policy = DynamicLoraPolicyV1(base_rank=8, max_rank=16, rank_growth_step=4,
                                 per_task_rank_budget=16, total_rank_budget=16)
    ra = RankAllocator(policy)
    assert ra.allocate("t1") == 8
    assert ra.allocate("t2") == 8          # spent == total: headroom 0
    assert ra.grow("t1") == 8              # original code returned 0
    assert ra.allocations == {"t1": 8, "t2": 8}
    assert ra.spent == 16


def test_growth_uses_current_rank_plus_headroom():
    """maximum_allowed = current + headroom, not headroom alone: a task
    holding 8 of a 24-rank budget with 8 left may reach 16."""
    policy = DynamicLoraPolicyV1(base_rank=8, max_rank=16, rank_growth_step=8,
                                 per_task_rank_budget=16, total_rank_budget=24)
    ra = RankAllocator(policy)
    ra.allocate("t1")                      # 8
    ra.allocate("t2")                      # 8, spent 16, headroom 8
    assert ra.grow("t1") == 16             # original code returned 8
    assert ra.spent == 24


def test_growth_invariants_hold_under_random_sequences():
    """Rank never decreases; spent == sum(allocations); global and
    per-task budgets never exceeded — over arbitrary operation orders."""
    import random
    rng = random.Random(7)
    policy = DynamicLoraPolicyV1(base_rank=4, max_rank=32, rank_growth_step=3,
                                 per_task_rank_budget=20, total_rank_budget=48)
    ra = RankAllocator(policy)
    tasks = [f"t{i}" for i in range(8)]
    for _ in range(500):
        t = rng.choice(tasks)
        if t in ra.allocations:
            before = ra.allocations[t]
            got = ra.grow(t, false_activation_breach=rng.random() < 0.3)
            assert got >= before
            assert ra.allocations[t] >= before
        else:
            try:
                ra.allocate(t, requested=rng.randrange(0, 30))
            except RuntimeError:
                pass
        assert ra.spent == sum(ra.allocations.values())
        assert ra.spent <= policy.total_rank_budget
        assert all(0 < v <= policy.per_task_rank_budget
                   for v in ra.allocations.values())


# ---------- generation chain -----------------------------------------

NOW = datetime(2026, 10, 8, 12, 0, tzinfo=timezone.utc)
NOW_TS = int(NOW.timestamp())


def _gen(i, parent_dig="", corpus_tag=None):
    return GenerationRecord(
        generation=i, campaign_digest=D(f"camp{i}"),
        corpus_digest=D(f"corpus-{corpus_tag or i}"),
        control_arm_digest=D(f"ctrl{i}"),
        parent_generation_digest=parent_dig)


def _promotion_registry(*extra_roles):
    signer = Ed25519Signer.generate()
    records = [AuthorityRecord(
        role="promotion", key_id=signer.key_id,
        public_key_b64=base64.b64encode(
            signer.public_bytes()).decode())]
    records.extend(extra_roles)
    return signer, AuthorityRegistry(records)


def _decision(signer, record, *, authorized_at=None, expires_at=None,
              **overrides):
    value = {
        "schema": PROMOTION_DECISION_SCHEMA,
        "generation": record.generation,
        "generation_record_digest": record.digest,
        "campaign_digest": record.campaign_digest,
        "qualification_digest": D(f"qualification-{record.generation}"),
        "authorized_at": NOW_TS - 60 if authorized_at is None else authorized_at,
        "expires_at": NOW_TS + 3600 if expires_at is None else expires_at,
    }
    value.update(overrides)
    env = signer.sign(value)
    return {"value": value, "digest": digest(value),
            "signer_key_id": env.key_id, "signature_b64": env.signature_b64}


def _promoted_chain(n=1):
    """Chain with generations 0..n-1, each promoted by a valid signed
    decision."""
    signer, registry = _promotion_registry()
    chain = GenerationChain()
    for i in range(n):
        parent = chain.tip.digest if chain.tip else ""
        chain.append(_gen(i, parent_dig=parent))
        chain.record_promotion(
            i, _decision(signer, chain.tip), registry=registry, now=NOW)
    return chain, signer, registry


def test_genesis_then_promotion_gated_child():
    chain, signer, registry = _promoted_chain(1)
    # child cannot be preregistered before parent promotion
    unpromoted = GenerationChain()
    unpromoted.append(_gen(0))
    with pytest.raises(PermissionError, match="not promoted"):
        unpromoted.append(_gen(1, parent_dig=unpromoted.tip.digest))
    chain.append(_gen(1, parent_dig=chain.tip.digest))
    assert len(chain.records) == 2
    assert chain.verify(registry, now=NOW) == []


def test_corpus_reuse_rejected():
    chain, signer, registry = _promoted_chain(1)
    with pytest.raises(ValueError, match="corpus reuse"):
        chain.append(_gen(1, parent_dig=chain.tip.digest, corpus_tag="0"))


def test_wrong_parent_digest_rejected():
    chain, signer, registry = _promoted_chain(1)
    with pytest.raises(ValueError, match="parent digest"):
        chain.append(_gen(1, parent_dig=D("wrong-parent")))


def test_skipped_generation_rejected():
    chain, signer, registry = _promoted_chain(2)
    with pytest.raises(ValueError, match="next generation"):
        chain.append(_gen(3, parent_dig=chain.tip.digest))


def test_promotion_only_at_tip():
    chain, signer, registry = _promoted_chain(1)
    chain.append(_gen(1, parent_dig=chain.tip.digest))
    # non-tip promotion refused
    with pytest.raises(ValueError, match="chain tip"):
        chain.record_promotion(
            0, _decision(signer, chain.records[0]), registry=registry,
            now=NOW)
    # records are immutable evidence: a child's promotion never rewrites
    # the parent
    g0_digest = chain.records[0].digest
    chain.record_promotion(
        1, _decision(signer, chain.tip), registry=registry, now=NOW)
    assert chain.records[0].digest == g0_digest
    # double-promotion refused
    with pytest.raises(ValueError, match="already promoted"):
        chain.record_promotion(
            1, _decision(signer, chain.tip), registry=registry, now=NOW)
    assert chain.verify(registry, now=NOW) == []


def test_verify_catches_tampered_sequence():
    """A forged chain (constructed out of band) must not verify."""
    chain = GenerationChain()
    bad = GenerationRecord(generation=1, campaign_digest=D("c"),
                           corpus_digest=D("k"), control_arm_digest=D("x"),
                           parent_generation_digest=D("nobody"))
    chain._records.append(bad)  # bypass append to simulate grafting
    assert chain.verify() != []


# ---------- FIX-003: verified promotion authorization ------------------

def test_bare_digest_is_not_authorization():
    """The original record_promotion accepted any digest-shaped string;
    promotion now requires a signed decision envelope."""
    signer, registry = _promotion_registry()
    chain = GenerationChain()
    chain.append(_gen(0))
    with pytest.raises(PermissionError, match="signed envelope"):
        chain.record_promotion(0, D("promotion-decision-0"),
                               registry=registry, now=NOW)
    assert chain.is_promoted(0) is False


def test_unregistered_signer_rejected():
    signer, registry = _promotion_registry()
    outsider = Ed25519Signer.generate()
    chain = GenerationChain()
    chain.append(_gen(0))
    with pytest.raises(PermissionError, match="not an authorized"):
        chain.record_promotion(
            0, _decision(outsider, chain.tip), registry=registry, now=NOW)


def test_wrong_role_signer_rejected():
    """A plan-role key — even a registered one — cannot promote."""
    plan_signer = Ed25519Signer.generate()
    plan_rec = AuthorityRecord(
        role="plan", key_id=plan_signer.key_id,
        public_key_b64=base64.b64encode(
            plan_signer.public_bytes()).decode())
    signer, registry = _promotion_registry(plan_rec)
    chain = GenerationChain()
    chain.append(_gen(0))
    with pytest.raises(PermissionError, match="not an authorized"):
        chain.record_promotion(
            0, _decision(plan_signer, chain.tip), registry=registry,
            now=NOW)


def test_tampered_decision_value_rejected():
    signer, registry = _promotion_registry()
    chain = GenerationChain()
    chain.append(_gen(0))
    doc = _decision(signer, chain.tip)
    doc["value"]["generation"] = 99  # tamper after signing
    with pytest.raises(ValueError, match="digest mismatch"):
        chain.record_promotion(0, doc, registry=registry, now=NOW)


def test_decision_bound_to_other_record_rejected():
    signer, registry = _promotion_registry()
    chain = GenerationChain()
    chain.append(_gen(0))
    other = _gen(0, corpus_tag="other")  # different campaign/corpus binding
    with pytest.raises(ValueError, match="bind this generation record"):
        chain.record_promotion(
            0, _decision(signer, other), registry=registry, now=NOW)
    # a decision whose campaign digest disagrees with the bound record
    doc = _decision(signer, chain.tip, campaign_digest=D("some-other-campaign"))
    with pytest.raises(ValueError, match="campaign"):
        chain.record_promotion(0, doc, registry=registry, now=NOW)


def test_decision_generation_mismatch_rejected():
    signer, registry = _promotion_registry()
    chain = GenerationChain()
    chain.append(_gen(0))
    doc = _decision(signer, chain.tip, generation=1)
    with pytest.raises(ValueError, match="generation mismatch"):
        chain.record_promotion(0, doc, registry=registry, now=NOW)


def test_expired_promotion_decision_rejected():
    signer, registry = _promotion_registry()
    chain = GenerationChain()
    chain.append(_gen(0))
    doc = _decision(signer, chain.tip, authorized_at=NOW_TS - 7200,
                    expires_at=NOW_TS - 3600)
    with pytest.raises(PermissionError, match="expired"):
        chain.record_promotion(0, doc, registry=registry, now=NOW)


def test_future_promotion_decision_rejected():
    signer, registry = _promotion_registry()
    chain = GenerationChain()
    chain.append(_gen(0))
    doc = _decision(signer, chain.tip, authorized_at=NOW_TS + 3600,
                    expires_at=NOW_TS + 7200)
    with pytest.raises(PermissionError, match="not yet valid"):
        chain.record_promotion(0, doc, registry=registry, now=NOW)


def test_missing_expiry_rejected():
    signer, registry = _promotion_registry()
    chain = GenerationChain()
    chain.append(_gen(0))
    doc = _decision(signer, chain.tip)
    del doc["value"]["expires_at"]
    env = signer.sign(doc["value"])
    doc["digest"] = digest(doc["value"])
    doc["signer_key_id"] = env.key_id
    doc["signature_b64"] = env.signature_b64
    with pytest.raises(ValueError, match="authorized_at/expires_at"):
        chain.record_promotion(0, doc, registry=registry, now=NOW)


def test_revoked_promotion_decision_rejected():
    signer, registry = _promotion_registry()
    chain = GenerationChain()
    chain.append(_gen(0))
    doc = _decision(signer, chain.tip)
    with pytest.raises(PermissionError, match="revoked"):
        chain.record_promotion(
            0, doc, registry=registry, now=NOW,
            revoked_decision_digests=[doc["digest"]])


def test_valid_signed_promotion_unlocks_child():
    chain, signer, registry = _promoted_chain(1)
    assert chain.is_promoted(0) is True
    assert len(chain.promotion_events) == 1
    chain.append(_gen(1, parent_dig=chain.tip.digest))
    assert chain.verify(registry, now=NOW) == []


def test_chain_verify_rechecks_signatures_and_expiry():
    chain, signer, registry = _promoted_chain(1)
    # a decision that expires later must be flagged when verification
    # happens after expiry
    later = NOW_TS + 7200
    late = datetime.fromtimestamp(later, tz=timezone.utc)
    problems = chain.verify(registry, now=late)
    assert any("expired" in p for p in problems)


def test_chain_verify_flags_mutated_event_decision():
    chain, signer, registry = _promoted_chain(1)
    # events carry the signed value; mutating it after the fact is
    # detected by digest re-verification
    chain._events[0].decision["expires_at"] = NOW_TS + 999999
    problems = chain.verify(registry, now=NOW)
    assert any("digest mismatch" in p or "signature invalid" in p
               for p in problems)


def test_chain_verify_flags_grafted_event():
    """An event whose record does not exist in the chain is a violation."""
    chain, signer, registry = _promoted_chain(1)
    ghost_record = _gen(7, parent_dig=D("ghost-parent"), corpus_tag="ghost")
    doc = _decision(signer, ghost_record)
    from minagi.v161.generations import check_promotion_decision
    ghost = check_promotion_decision(
        ghost_record, doc, registry=registry, now=NOW)
    chain._events.append(ghost)
    problems = chain.verify(registry, now=NOW)
    assert any("no such generation record" in p for p in problems)


def test_chain_verify_flags_duplicate_promotion_events():
    """Two independently signed decisions for one record is still an
    invalid chain — exactly one promotion per generation."""
    chain, signer, registry = _promoted_chain(1)
    doc = _decision(signer, chain.records[0],
                    qualification_digest=D("second-qualification"))
    from minagi.v161.generations import check_promotion_decision
    duplicate = check_promotion_decision(
        chain.records[0], doc, registry=registry, now=NOW)
    chain._events.append(duplicate)
    problems = chain.verify(registry, now=NOW)
    assert any("only one promotion per generation" in p
               for p in problems)
