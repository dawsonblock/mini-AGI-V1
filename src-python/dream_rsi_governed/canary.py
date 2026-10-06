from __future__ import annotations

from datetime import datetime, timezone
from typing import Callable

from .canonical import sha256_digest
from .models import CanaryAttestation


def compare_live_canary(
    candidate_policy_id: str,
    baseline_policy_id: str,
    run_candidate: Callable[[], tuple[float, float]],
    run_baseline: Callable[[], tuple[float, float]],
    environment_digest: str,
    max_quality_regression: float = 0.0,
    max_cost_multiplier: float = 1.25,
) -> CanaryAttestation:
    cq, cc = run_candidate()
    bq, bc = run_baseline()
    quality_ok = cq + max_quality_regression >= bq
    cost_ok = cc <= max_cost_multiplier * max(bc, 1e-12)
    passed = quality_ok and cost_ok
    created = datetime.now(timezone.utc).isoformat()
    run_digest = sha256_digest({
        "candidate_policy_id": candidate_policy_id,
        "baseline_policy_id": baseline_policy_id,
        "candidate_score": cq,
        "baseline_score": bq,
        "candidate_cost": cc,
        "baseline_cost": bc,
        "environment_digest": environment_digest,
        "created_at": created,
    })
    return CanaryAttestation(
        candidate_policy_id=candidate_policy_id,
        baseline_policy_id=baseline_policy_id,
        candidate_score=cq,
        baseline_score=bq,
        candidate_cost=cc,
        baseline_cost=bc,
        passed=passed,
        environment_digest=environment_digest,
        run_digest=run_digest,
        created_at=created,
    )
