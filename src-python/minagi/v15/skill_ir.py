from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from typing import Iterable, Mapping

from egai.common.canonical import digest, validate_digest


@dataclass(frozen=True)
class SkillInstruction:
    op: str
    arg: str = ""

    def __post_init__(self) -> None:
        allowed = {
            "identity", "reverse", "uppercase", "lowercase", "sort_chars",
            "strip", "collapse_ws", "canonical_json", "prefix", "suffix",
        }
        if self.op not in allowed:
            raise ValueError(f"unsupported SkillIR op: {self.op}")
        if self.op in {"prefix", "suffix"} and not self.arg:
            raise ValueError(f"{self.op} requires arg")
        if self.op not in {"prefix", "suffix"} and self.arg:
            raise ValueError(f"{self.op} does not accept arg")


@dataclass(frozen=True)
class SkillIR:
    skill_id: str
    task_family: str
    instructions: tuple[SkillInstruction, ...]
    source_candidate_digest: str
    supporting_trajectory_digests: tuple[str, ...]
    transfer_receipt_digest: str
    schema: str = "mini-agi-v15.1-skill-ir-v1"

    def __post_init__(self) -> None:
        if not self.skill_id or not self.task_family:
            raise ValueError("skill identity and task family required")
        if not self.instructions:
            raise ValueError("SkillIR requires at least one instruction")
        validate_digest(self.source_candidate_digest)
        validate_digest(self.transfer_receipt_digest)
        for d in self.supporting_trajectory_digests:
            validate_digest(d)
        if len(set(self.supporting_trajectory_digests)) != len(self.supporting_trajectory_digests):
            raise ValueError("duplicate supporting trajectory digest")

    @property
    def digest(self) -> str:
        return digest(self)


class SkillIRCompiler:
    """Compile a transfer-qualified trajectory candidate into deterministic SkillIR.

    This compiler is intentionally conservative. Unknown signatures fail closed
    instead of becoming free-form code or prompts.
    """

    def compile(self, candidate) -> SkillIR:
        status = getattr(candidate, "status", None)
        status_value = getattr(status, "value", str(status))
        if status_value != "transfer_qualified":
            raise PermissionError("only transfer-qualified skill candidates may compile")
        receipt = str(getattr(candidate, "transfer_receipt_digest", ""))
        validate_digest(receipt)
        signature = str(getattr(candidate, "operation_signature"))
        instruction = self._instruction(signature)
        return SkillIR(
            skill_id=str(candidate.skill_id),
            task_family=str(candidate.task_family),
            instructions=(instruction,),
            source_candidate_digest=str(candidate.digest),
            supporting_trajectory_digests=tuple(sorted(set(candidate.supporting_trajectory_digests))),
            transfer_receipt_digest=receipt,
        )

    @staticmethod
    def _instruction(signature: str) -> SkillInstruction:
        direct = {
            "reverse": "reverse",
            "uppercase": "uppercase",
            "lowercase": "lowercase",
            "sort-characters": "sort_chars",
            "strip-whitespace": "strip",
            "collapse-whitespace": "collapse_ws",
            "canonical-json": "canonical_json",
        }
        if signature in direct:
            return SkillInstruction(direct[signature])
        if signature.startswith("prefix:"):
            return SkillInstruction("prefix", signature.split(":", 1)[1])
        if signature.startswith("suffix:"):
            return SkillInstruction("suffix", signature.split(":", 1)[1])
        raise ValueError(f"operation signature is not executable SkillIR: {signature}")


class SkillIRExecutor:
    def execute(self, skill: SkillIR, input_text: str) -> str:
        value = str(input_text)
        for ins in skill.instructions:
            if ins.op == "identity":
                pass
            elif ins.op == "reverse":
                value = value[::-1]
            elif ins.op == "uppercase":
                value = value.upper()
            elif ins.op == "lowercase":
                value = value.lower()
            elif ins.op == "sort_chars":
                value = "".join(sorted(value))
            elif ins.op == "strip":
                value = value.strip()
            elif ins.op == "collapse_ws":
                value = " ".join(value.split())
            elif ins.op == "canonical_json":
                value = json.dumps(json.loads(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
            elif ins.op == "prefix":
                value = ins.arg + value
            elif ins.op == "suffix":
                value = value + ins.arg
            else:  # defensive; dataclass validation should make this unreachable
                raise RuntimeError(f"unhandled SkillIR op: {ins.op}")
        return value


@dataclass(frozen=True)
class SkillPolicyBundle:
    skills: tuple[SkillIR, ...]
    schema: str = "mini-agi-v15.1-skill-policy-bundle-v1"

    def __post_init__(self) -> None:
        if not self.skills:
            raise ValueError("skill policy bundle cannot be empty")
        ids = [s.skill_id for s in self.skills]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate skill id in policy bundle")
        canonical = tuple(sorted(self.skills, key=lambda s: (s.task_family, s.skill_id, s.digest)))
        if canonical != self.skills:
            raise ValueError("skills must be in canonical order")

    @property
    def digest(self) -> str:
        return digest(self)

    @property
    def root_hex(self) -> str:
        return self.digest.split(":", 1)[1]

    @classmethod
    def from_skills(cls, skills: Iterable[SkillIR]) -> "SkillPolicyBundle":
        return cls(tuple(sorted(tuple(skills), key=lambda s: (s.task_family, s.skill_id, s.digest))))

    def lookup(self, task_family: str) -> SkillIR | None:
        matches = [s for s in self.skills if s.task_family == task_family]
        if not matches:
            return None
        if len(matches) > 1:
            raise RuntimeError(f"ambiguous skill policy for task family {task_family!r}")
        return matches[0]


class SkillPolicyRuntime:
    def __init__(self, bundle: SkillPolicyBundle, *, executor: SkillIRExecutor | None = None):
        self.bundle = bundle
        self.executor = executor or SkillIRExecutor()

    def run(self, *, task_family: str, input_text: str, fallback=None) -> str:
        skill = self.bundle.lookup(task_family)
        if skill is None:
            if fallback is None:
                return str(input_text)
            return str(fallback(task_family, input_text))
        return self.executor.execute(skill, input_text)


def skill_ir_from_mapping(value: Mapping[str, object]) -> SkillIR:
    instructions = tuple(
        SkillInstruction(str(v["op"]), str(v.get("arg", "")))
        for v in value.get("instructions", ())
    )
    return SkillIR(
        skill_id=str(value["skill_id"]),
        task_family=str(value["task_family"]),
        instructions=instructions,
        source_candidate_digest=str(value["source_candidate_digest"]),
        supporting_trajectory_digests=tuple(str(v) for v in value.get("supporting_trajectory_digests", ())),
        transfer_receipt_digest=str(value["transfer_receipt_digest"]),
        schema=str(value.get("schema", "mini-agi-v15.1-skill-ir-v1")),
    )


def skill_policy_bundle_from_mapping(value: Mapping[str, object]) -> SkillPolicyBundle:
    return SkillPolicyBundle(
        skills=tuple(skill_ir_from_mapping(v) for v in value.get("skills", ())),
        schema=str(value.get("schema", "mini-agi-v15.1-skill-policy-bundle-v1")),
    )
