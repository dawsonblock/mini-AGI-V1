from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import time
import uuid

from egai.common.canonical import digest, validate_digest


@dataclass(frozen=True)
class BeliefRevisionV144:
    belief_id: str
    revision: int
    subject: str
    predicate: str
    object_json: str
    valid_from: float
    valid_to: float | None
    transaction_from: float
    evidence_digests: tuple[str, ...]
    confidence: float
    supersedes_digest: str = ""
    created_at: float = 0.0
    schema: str = "mini-agi-v14.1-alpha4-belief-revision-v1"

    def __post_init__(self):
        if not self.belief_id or not self.subject or not self.predicate:
            raise ValueError("belief identity/relation required")
        if self.revision < 1:
            raise ValueError("revision must be >=1")
        if self.valid_to is not None and self.valid_to <= self.valid_from:
            raise ValueError("valid_to must be after valid_from")
        if not 0.0 <= float(self.confidence) <= 1.0:
            raise ValueError("confidence must be in [0,1]")
        for d in self.evidence_digests:
            validate_digest(d)
        if self.supersedes_digest:
            validate_digest(self.supersedes_digest)
        if not self.created_at:
            object.__setattr__(self, "created_at", self.transaction_from)

    @property
    def digest(self):
        return digest(self)


@dataclass(frozen=True)
class SkillRevisionV144:
    skill_id: str
    revision: int
    status: str
    procedure_digest: str
    positive_evidence: tuple[str, ...]
    negative_evidence: tuple[str, ...]
    failure_envelope_digest: str
    supersedes_digest: str = ""
    created_at: float = 0.0
    schema: str = "mini-agi-v14.1-alpha4-skill-revision-v1"

    def __post_init__(self):
        if not self.skill_id or self.revision < 1:
            raise ValueError("skill identity/revision required")
        if self.status not in {"DISCOVERED","PROPOSED","CANDIDATE","QUALIFIED","ACTIVE","DEGRADED","RETIRED"}:
            raise ValueError("invalid skill status")
        validate_digest(self.procedure_digest); validate_digest(self.failure_envelope_digest)
        for d in self.positive_evidence + self.negative_evidence:
            validate_digest(d)
        if self.supersedes_digest:
            validate_digest(self.supersedes_digest)
        if not self.created_at:
            object.__setattr__(self, "created_at", time.time())

    @property
    def digest(self):
        return digest(self)


class ContentBoundPersistencePlannerV144:
    """Precomputes immutable learned-state objects before promotion.

    The returned digest is the exact mutation target carried in BuildManifest and
    PromotionAuthorization.  Applying the revision later cannot change its bytes.
    """

    def __init__(self, *, cas, governance_db):
        self.cas = cas
        self.db = governance_db

    def prepare_belief(self, *, subject: str, predicate: str, object_value, valid_from: float,
                       evidence_digests: tuple[str, ...], confidence: float, belief_id: str | None = None,
                       valid_to: float | None = None, transaction_from: float | None = None):
        belief_id = belief_id or "BEL-" + uuid.uuid4().hex
        head = self.db.belief_head(belief_id)
        prev = "" if head is None else str(head["belief_digest"])
        rev = 1
        if head is not None:
            row = self.db.conn.execute("SELECT revision FROM belief_revisions_v144 WHERE belief_digest=?", (prev,)).fetchone()
            rev = int(row["revision"]) + 1
        tx = float(transaction_from or time.time())
        obj_json = json.dumps(object_value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        belief = BeliefRevisionV144(belief_id, rev, str(subject), str(predicate), obj_json, float(valid_from),
                                    None if valid_to is None else float(valid_to), tx, tuple(evidence_digests),
                                    float(confidence), prev, tx)
        value = self.cas.put_json(asdict(belief))
        if value != belief.digest:
            raise RuntimeError("belief CAS digest mismatch")
        return belief

    def prepare_skill(self, *, status: str, procedure_digest: str, positive_evidence: tuple[str, ...],
                      negative_evidence: tuple[str, ...], failure_envelope_digest: str,
                      skill_id: str | None = None):
        skill_id = skill_id or "SK-" + uuid.uuid4().hex
        head = self.db.skill_head(skill_id)
        prev = "" if head is None else str(head["skill_digest"])
        rev = 1
        if head is not None:
            row = self.db.conn.execute("SELECT revision,status FROM skill_revisions_v144 WHERE skill_digest=?", (prev,)).fetchone()
            rev = int(row["revision"]) + 1
        skill = SkillRevisionV144(skill_id, rev, str(status).upper(), validate_digest(procedure_digest),
                                   tuple(positive_evidence), tuple(negative_evidence),
                                   validate_digest(failure_envelope_digest), prev, time.time())
        value = self.cas.put_json(asdict(skill))
        if value != skill.digest:
            raise RuntimeError("skill CAS digest mismatch")
        return skill
