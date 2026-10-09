from __future__ import annotations

from dataclasses import dataclass
import math
import random

from egai.common.canonical import digest, validate_digest


def paired_bootstrap_ci_v145(candidate, baseline, *, confidence=.95, iterations=4000, seed=0):
    if len(candidate) != len(baseline) or not candidate:
        raise ValueError("non-empty paired samples required")
    rng=random.Random(seed); n=len(candidate); vals=[]
    for _ in range(int(iterations)):
        idx=[rng.randrange(n) for _ in range(n)]
        vals.append(math.fsum(float(candidate[i])-float(baseline[i]) for i in idx)/n)
    vals.sort(); alpha=(1-confidence)/2
    lo=vals[max(0,int(alpha*len(vals)))]; hi=vals[min(len(vals)-1,int((1-alpha)*len(vals))-1)]
    return lo,hi


@dataclass(frozen=True)
class FrozenArmRecordV145:
    task_digest: str
    family_id: str
    ring: str
    foundation_model_digest: str
    production_identity_digest: str
    arm: str
    score: float
    security_regressions: int = 0
    retention_score: float = 1.0
    schema: str = "mini-agi-v14.1-alpha5-frozen-arm-record-v1"
    def __post_init__(self):
        validate_digest(self.task_digest); validate_digest(self.foundation_model_digest); validate_digest(self.production_identity_digest)
        if self.arm not in {"A0","A1"}: raise ValueError("arm must be A0 or A1")
        if not math.isfinite(float(self.score)): raise ValueError("score must be finite")
        if self.security_regressions < 0: raise ValueError("security regressions cannot be negative")
    @property
    def digest(self): return digest(self)


@dataclass(frozen=True)
class FrozenBaselineReportV145:
    foundation_model_digest: str
    production_identity_digest: str
    task_digests: tuple[str,...]
    mean_delta: float
    ci95: tuple[float,float]
    per_ring: dict
    security_regressions: int
    min_retention: float
    decision: str
    reasons: tuple[str,...]
    schema: str = "mini-agi-v14.1-alpha5-frozen-baseline-report-v1"
    @property
    def digest(self): return digest(self)


class FrozenBaselineGateV145:
    """A0/A1 gate: persistent learning must beat the same frozen foundation."""
    def __init__(self, *, required_rings=("R0","R1","R2"), minimum_effect=0.0,
                 minimum_retention=0.95, bootstrap_iterations=4000, seed=0):
        self.required_rings=tuple(required_rings); self.minimum_effect=float(minimum_effect)
        self.minimum_retention=float(minimum_retention); self.iterations=int(bootstrap_iterations); self.seed=int(seed)

    def evaluate(self, a0, a1):
        a0=tuple(a0); a1=tuple(a1)
        if not a0 or len(a0)!=len(a1): raise ValueError("paired A0/A1 records required")
        m0={r.task_digest:r for r in a0}; m1={r.task_digest:r for r in a1}
        if set(m0)!=set(m1): raise ValueError("A0/A1 task sets differ")
        ordered=sorted(m0)
        model={m0[d].foundation_model_digest for d in ordered}|{m1[d].foundation_model_digest for d in ordered}
        identity={m0[d].production_identity_digest for d in ordered}|{m1[d].production_identity_digest for d in ordered}
        if len(model)!=1: raise PermissionError("frozen-baseline experiment changed foundation model")
        if len(identity)!=1: raise PermissionError("A0/A1 production identity mismatch")
        baseline=[m0[d].score for d in ordered]; candidate=[m1[d].score for d in ordered]
        ci=paired_bootstrap_ci_v145(candidate,baseline,iterations=self.iterations,seed=self.seed)
        delta=math.fsum(c-b for c,b in zip(candidate,baseline))/len(ordered)
        per_ring={}; reasons=[]
        for ring in self.required_rings:
            ds=[d for d in ordered if m0[d].ring==ring and m1[d].ring==ring]
            if not ds:
                reasons.append(f"{ring} lacks paired tasks"); continue
            b=[m0[d].score for d in ds]; c=[m1[d].score for d in ds]
            rci=paired_bootstrap_ci_v145(c,b,iterations=max(500,self.iterations//2),seed=self.seed+len(ring))
            per_ring[ring]={"n":len(ds),"mean_delta":math.fsum(x-y for x,y in zip(c,b))/len(ds),"ci95":rci}
            if rci[0] <= self.minimum_effect: reasons.append(f"{ring} transfer lower bound is not > {self.minimum_effect}")
        security=sum(r.security_regressions for r in a1)
        retention=min(r.retention_score for r in a1)
        if security: reasons.append("security regressions must be zero")
        if retention < self.minimum_retention: reasons.append("retention floor violated")
        if ci[0] <= self.minimum_effect: reasons.append("overall paired improvement lower bound is not positive")
        decision="PASS" if not reasons else "BLOCK"
        return FrozenBaselineReportV145(next(iter(model)),next(iter(identity)),tuple(ordered),delta,ci,per_ring,security,retention,decision,tuple(reasons))
