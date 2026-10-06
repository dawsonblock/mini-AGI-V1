from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import json
import math
from typing import Callable, Iterable, Mapping, Protocol, Sequence

from egai.common.canonical import digest, validate_digest


class CandidateStatusV145(str, Enum):
    PROPOSED = "proposed"
    TRANSFER_QUALIFIED = "transfer_qualified"
    BLOCKED = "blocked"


@dataclass(frozen=True)
class TrajectoryStepV145:
    step_id: str
    observation_digest: str
    action_digest: str
    outcome_digest: str
    score: float
    success: bool
    tool_name: str = ""
    metadata_digest: str = ""

    def __post_init__(self):
        if not self.step_id: raise ValueError("step_id required")
        for d in (self.observation_digest, self.action_digest, self.outcome_digest): validate_digest(d)
        if self.metadata_digest: validate_digest(self.metadata_digest)
        if not math.isfinite(float(self.score)): raise ValueError("step score must be finite")

    @property
    def digest(self): return digest(self)


@dataclass(frozen=True)
class VerifiedTrajectoryV145:
    trajectory_id: str
    task_family: str
    input_text: str
    attempted_output: str
    repaired_output: str
    production_identity_digest: str
    evidence_digests: tuple[str, ...]
    verification_receipt_digests: tuple[str, ...]
    steps: tuple[TrajectoryStepV145, ...] = ()
    outcome_score: float = 1.0
    verified: bool = True
    tags: tuple[str, ...] = ()

    def __post_init__(self):
        if not self.trajectory_id or not self.task_family: raise ValueError("trajectory identity/family required")
        validate_digest(self.production_identity_digest)
        if not self.evidence_digests or not self.verification_receipt_digests:
            raise ValueError("trajectory requires evidence and verification receipts")
        for d in self.evidence_digests + self.verification_receipt_digests: validate_digest(d)
        if not math.isfinite(float(self.outcome_score)) or not 0 <= float(self.outcome_score) <= 1:
            raise ValueError("outcome_score must be in [0,1]")
        ids=[s.step_id for s in self.steps]
        if len(ids) != len(set(ids)): raise ValueError("duplicate trajectory step id")

    @property
    def input_digest(self): return digest({"input": self.input_text})
    @property
    def repaired_output_digest(self): return digest({"output": self.repaired_output})
    @property
    def digest(self): return digest(self)


@dataclass(frozen=True)
class LocalLessonV145:
    task_family: str
    operation_signature: str
    trigger: str
    procedure: str
    rationale: str
    source_trajectory_digest: str
    evidence_digests: tuple[str, ...]
    verification_receipt_digests: tuple[str, ...]
    @property
    def digest(self): return digest(self)


@dataclass(frozen=True)
class ContradictionV145:
    task_family: str
    input_digest: str
    output_digests: tuple[str, ...]
    trajectory_digests: tuple[str, ...]
    @property
    def digest(self): return digest(self)


@dataclass(frozen=True)
class SkillCandidateArtifactV145:
    skill_id: str
    name: str
    task_family: str
    trigger: str
    procedure: str
    rationale: str
    operation_signature: str
    supporting_trajectory_digests: tuple[str, ...]
    supporting_evidence: tuple[str, ...]
    verification_receipts: tuple[str, ...]
    distinct_input_digests: tuple[str, ...]
    source_lesson_digests: tuple[str, ...]
    status: CandidateStatusV145 = CandidateStatusV145.PROPOSED
    transfer_score: float | None = None
    holdout_trajectory_digests: tuple[str, ...] = ()
    transfer_receipt_digest: str = ""
    known_failures: tuple[str, ...] = ()
    schema: str = "mini-agi-v14.1-alpha5-trajectory-skill-candidate-v1"

    def __post_init__(self):
        if len(self.supporting_trajectory_digests) < 2 or len(self.distinct_input_digests) < 2:
            raise ValueError("skill candidate requires repeated support across distinct inputs")
        for seq in (self.supporting_trajectory_digests, self.supporting_evidence, self.verification_receipts,
                    self.distinct_input_digests, self.source_lesson_digests, self.holdout_trajectory_digests):
            for d in seq: validate_digest(d)
        if self.transfer_receipt_digest: validate_digest(self.transfer_receipt_digest)
        if self.status is CandidateStatusV145.TRANSFER_QUALIFIED and not self.transfer_receipt_digest:
            raise ValueError("transfer-qualified candidate requires receipt")

    @property
    def digest(self): return digest(self)


class LessonExtractorV145(Protocol):
    def extract(self, trajectory: VerifiedTrajectoryV145) -> LocalLessonV145 | None: ...


class ConservativeLessonExtractorV145:
    VERSION = "conservative-trajectory-extractor/alpha5"
    @staticmethod
    def _collapse_ws(s: str) -> str: return " ".join(s.split())

    def _infer(self, inp: str, out: str):
        if out == inp[::-1]: return "reverse", "Reverse the input characters.", "verified repairs consistently reverse the complete input"
        if out == inp.upper() and out != inp: return "uppercase", "Convert the input text to uppercase.", "verified repairs consistently apply uppercase normalization"
        if out == inp.lower() and out != inp: return "lowercase", "Convert the input text to lowercase.", "verified repairs consistently apply lowercase normalization"
        if out == "".join(sorted(inp)) and out != inp: return "sort-characters", "Sort input characters in ascending order.", "verified repairs consistently sort characters"
        if out == inp.strip() and out != inp: return "strip-whitespace", "Remove leading and trailing whitespace.", "verified repairs consistently strip boundary whitespace"
        if out == self._collapse_ws(inp) and out != inp: return "collapse-whitespace", "Collapse internal whitespace runs to single spaces.", "verified repairs consistently normalize whitespace"
        try:
            obj=json.loads(inp); canonical=json.dumps(obj,sort_keys=True,separators=(",",":"),ensure_ascii=False)
            if out == canonical and out != inp: return "canonical-json", "Parse JSON and emit canonical compact JSON with sorted keys.", "verified repairs consistently canonicalize JSON"
        except Exception: pass
        pos=out.find(inp)
        if pos >= 0 and len(out) > len(inp):
            pre,suf=out[:pos],out[pos+len(inp):]
            if pre and not suf: return f"prefix:{pre}", f"Prefix the input with {pre!r}.", "verified repairs consistently add the same prefix"
            if suf and not pre: return f"suffix:{suf}", f"Append {suf!r} to the input.", "verified repairs consistently add the same suffix"
        return None

    def extract(self, trajectory: VerifiedTrajectoryV145):
        if not trajectory.verified or trajectory.outcome_score < 1.0: return None
        inferred=self._infer(trajectory.input_text, trajectory.repaired_output)
        if inferred is None: return None
        sig,procedure,rationale=inferred
        return LocalLessonV145(trajectory.task_family,sig,trajectory.task_family,procedure,rationale,trajectory.digest,
                               tuple(sorted(set(trajectory.evidence_digests))),tuple(sorted(set(trajectory.verification_receipt_digests))))


@dataclass(frozen=True)
class InductionReportV145:
    candidates: tuple[SkillCandidateArtifactV145, ...]
    contradictions: tuple[ContradictionV145, ...]
    rejected_trajectory_digests: tuple[str, ...]
    accepted_trajectory_digests: tuple[str, ...]
    @property
    def digest(self): return digest(self)


class TrajectorySkillInducerV145:
    """Proposal-only induction. Contradictory verified trajectories are blocked."""
    def __init__(self, extractor: LessonExtractorV145 | None = None, *, min_support=2, min_distinct_inputs=2):
        if min_support < 2 or min_distinct_inputs < 2: raise ValueError("requires >=2 support/distinct inputs")
        self.extractor=extractor or ConservativeLessonExtractorV145(); self.min_support=int(min_support); self.min_distinct_inputs=int(min_distinct_inputs)

    @staticmethod
    def _contradictions(xs: Sequence[VerifiedTrajectoryV145]):
        buckets={}
        for t in xs:
            if t.verified: buckets.setdefault((t.task_family,t.input_digest),[]).append(t)
        out=[]
        for (family,input_digest),group in buckets.items():
            outputs=sorted({x.repaired_output_digest for x in group})
            if len(outputs)>1: out.append(ContradictionV145(family,input_digest,tuple(outputs),tuple(sorted(x.digest for x in group))))
        return tuple(sorted(out,key=lambda x:x.digest))

    def induce(self, trajectories: Iterable[VerifiedTrajectoryV145]):
        xs=tuple(trajectories); contradictions=self._contradictions(xs); blocked={(c.task_family,c.input_digest) for c in contradictions}
        lessons=[]; rejected=set(); accepted=set(); by_digest={t.digest:t for t in xs}
        for t in xs:
            if not t.verified or t.outcome_score < 1.0 or (t.task_family,t.input_digest) in blocked:
                rejected.add(t.digest); continue
            l=self.extractor.extract(t)
            if l is None: rejected.add(t.digest); continue
            lessons.append(l); accepted.add(t.digest)
        grouped={}
        for l in lessons: grouped.setdefault((l.task_family,l.operation_signature,l.trigger,l.procedure),[]).append(l)
        candidates=[]
        for (family,sig,trigger,procedure),group in grouped.items():
            ts=[by_digest[l.source_trajectory_digest] for l in group]; inputs=sorted({t.input_digest for t in ts})
            if len(group)<self.min_support or len(inputs)<self.min_distinct_inputs: continue
            evid=sorted({d for l in group for d in l.evidence_digests}); rec=sorted({d for l in group for d in l.verification_receipt_digests})
            lesson_ds=sorted({l.digest for l in group}); traj_ds=sorted({l.source_trajectory_digest for l in group})
            sid="skill-"+digest({"family":family,"signature":sig,"procedure":procedure}).split(":",1)[1][:20]
            candidates.append(SkillCandidateArtifactV145(sid,f"{family}: {sig}",family,trigger,procedure,
                f"consolidated {len(group)} independently verified trajectories using {sig}",sig,
                tuple(traj_ds),tuple(evid),tuple(rec),tuple(inputs),tuple(lesson_ds)))
        return InductionReportV145(tuple(sorted(candidates,key=lambda x:x.digest)),contradictions,tuple(sorted(rejected)),tuple(sorted(accepted)))


class HeldOutTransferQualifierV145:
    def __init__(self, *, evaluator_id: str, minimum_score: float=1.0, cas=None):
        if not evaluator_id: raise ValueError("evaluator_id required")
        if not 0 <= minimum_score <= 1: raise ValueError("minimum score must be in [0,1]")
        self.evaluator_id=evaluator_id; self.minimum_score=float(minimum_score); self.cas=cas

    def qualify(self, candidate: SkillCandidateArtifactV145, holdouts: Sequence[VerifiedTrajectoryV145],
                evaluator: Callable[[SkillCandidateArtifactV145,VerifiedTrajectoryV145],float]):
        if candidate.status is not CandidateStatusV145.PROPOSED: raise ValueError("candidate must be proposed")
        if not holdouts: raise ValueError("held-out trajectories required")
        support=set(candidate.supporting_trajectory_digests); scores=[]; ds=[]
        for t in holdouts:
            if t.digest in support: raise ValueError("support trajectory cannot be holdout")
            if not t.verified or t.task_family != candidate.task_family: raise ValueError("invalid holdout")
            s=float(evaluator(candidate,t))
            if not math.isfinite(s) or not 0 <= s <= 1: raise ValueError("invalid transfer score")
            scores.append(s); ds.append(t.digest)
        aggregate=sum(scores)/len(scores)
        receipt_body={"schema":"mini-agi-v14.1-alpha5-heldout-transfer-receipt-v1","evaluator_id":self.evaluator_id,
                      "candidate_digest":candidate.digest,"holdouts":ds,"scores":scores,"aggregate":aggregate}
        receipt=self.cas.put_json(receipt_body) if self.cas is not None else digest(receipt_body)
        body={**asdict(candidate)}
        body["status"] = CandidateStatusV145.TRANSFER_QUALIFIED if aggregate >= self.minimum_score else CandidateStatusV145.BLOCKED
        body["transfer_score"] = aggregate; body["holdout_trajectory_digests"] = tuple(ds)
        body["transfer_receipt_digest"] = receipt if aggregate >= self.minimum_score else ""
        if aggregate < self.minimum_score: body["known_failures"] = tuple(candidate.known_failures)+(f"held-out transfer score {aggregate:.6f} below {self.minimum_score:.6f}",)
        return SkillCandidateArtifactV145(**body)
