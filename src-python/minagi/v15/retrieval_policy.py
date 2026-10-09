from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable, Mapping, Sequence

from egai.common.canonical import digest, validate_digest

_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def _tokens(text: str) -> frozenset[str]:
    return frozenset(t.lower() for t in _TOKEN_RE.findall(str(text)))


@dataclass(frozen=True)
class RetrievalPolicy:
    top_k: int = 4
    minimum_score: float = 0.0
    family_bonus: float = 2.0
    lexical_weight: float = 1.0
    schema: str = "mini-agi-v15.6-retrieval-policy-v1"

    def __post_init__(self) -> None:
        if not 1 <= int(self.top_k) <= 64:
            raise ValueError("top_k must be in [1,64]")
        if float(self.minimum_score) < 0.0:
            raise ValueError("minimum_score must be non-negative")
        if float(self.family_bonus) < 0.0 or float(self.lexical_weight) < 0.0:
            raise ValueError("retrieval weights must be non-negative")

    @property
    def digest(self) -> str:
        return digest(self)

    @property
    def root_hex(self) -> str:
        return self.digest.split(":", 1)[1]

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "RetrievalPolicy":
        return cls(
            top_k=int(value.get("top_k", 4)),
            minimum_score=float(value.get("minimum_score", 0.0)),
            family_bonus=float(value.get("family_bonus", 2.0)),
            lexical_weight=float(value.get("lexical_weight", 1.0)),
            schema=str(value.get("schema", "mini-agi-v15.6-retrieval-policy-v1")),
        )


@dataclass(frozen=True)
class EpisodicMemoryRecord:
    record_id: str
    task_family: str
    input_text: str
    output_text: str
    evidence_digest: str
    schema: str = "mini-agi-v15.6-episodic-memory-record-v1"

    def __post_init__(self) -> None:
        if not self.record_id or not self.task_family:
            raise ValueError("record identity and task family required")
        validate_digest(self.evidence_digest)

    @property
    def digest(self) -> str:
        return digest(self)


@dataclass(frozen=True)
class RetrievalHit:
    record: EpisodicMemoryRecord
    score: float


class RetrievalPolicyRuntime:
    def __init__(self, policy: RetrievalPolicy, records: Iterable[EpisodicMemoryRecord] = ()):
        self.policy = policy
        self.records = tuple(records)
        ids = [r.record_id for r in self.records]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate episodic record id")

    def retrieve(self, *, task_family: str, query: str) -> tuple[RetrievalHit, ...]:
        q = _tokens(query)
        ranked: list[RetrievalHit] = []
        for record in self.records:
            rt = _tokens(record.input_text)
            union = q | rt
            lexical = 0.0 if not union else len(q & rt) / len(union)
            score = self.policy.lexical_weight * lexical
            if record.task_family == task_family:
                score += self.policy.family_bonus
            if score >= self.policy.minimum_score:
                ranked.append(RetrievalHit(record, float(score)))
        ranked.sort(key=lambda h: (-h.score, h.record.record_id, h.record.digest))
        return tuple(ranked[: self.policy.top_k])

    @staticmethod
    def prompt_context(hits: Sequence[RetrievalHit]) -> str:
        if not hits:
            return ""
        chunks = []
        for hit in hits:
            r = hit.record
            chunks.append(f"[{r.record_id}] input={r.input_text!r} output={r.output_text!r}")
        return "\n".join(chunks)
