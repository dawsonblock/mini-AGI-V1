from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum

from minagi.egai.canonical import sha256_json


class FalsificationKind(str, Enum):
    COUNTEREXAMPLE = "counterexample"
    NEGATIVE_CONTROL = "negative_control"
    ADVERSARIAL = "adversarial"
    ABLATION = "ablation"
    OOD = "ood"
    RETENTION = "retention"
    SECURITY = "security"


@dataclass(frozen=True)
class FalsificationCase:
    case_id: str
    kind: FalsificationKind
    payload_digest: str
    expected_property_digest: str
    fresh_task_lease_digest: str = ""

    @property
    def digest(self) -> str:
        body = asdict(self)
        body["kind"] = self.kind.value
        return sha256_json(body)


@dataclass(frozen=True)
class FalsificationPlan:
    candidate_digest: str
    cases: tuple[FalsificationCase, ...]
    generated_by: str
    preregistered: bool = True

    @property
    def digest(self) -> str:
        return sha256_json({
            "candidate_digest": self.candidate_digest,
            "cases": [c.digest for c in self.cases],
            "generated_by": self.generated_by,
            "preregistered": self.preregistered,
        })

    def validate(self) -> None:
        if not self.preregistered:
            raise PermissionError("falsification plan must be preregistered")
        if not self.cases:
            raise ValueError("falsification plan requires cases")
        kinds = {c.kind for c in self.cases}
        mandatory = {FalsificationKind.NEGATIVE_CONTROL, FalsificationKind.RETENTION, FalsificationKind.SECURITY}
        missing = mandatory - kinds
        if missing:
            raise ValueError(f"missing mandatory falsification classes: {sorted(x.value for x in missing)}")
        ids = [c.case_id for c in self.cases]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate falsification case IDs")
