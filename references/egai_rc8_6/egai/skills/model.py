from dataclasses import dataclass, field
from typing import Any
from egai.common.canonical import digest

@dataclass(frozen=True)
class SkillManifest:
    skill_id: str
    version: str
    purpose: str
    activation_conditions: tuple[str,...]
    preconditions: tuple[str,...]
    contraindications: tuple[str,...]
    inputs: tuple[str,...]
    outputs: tuple[str,...]
    implementation: dict[str,Any]
    dependencies: tuple[str,...]=()
    permissions: tuple[str,...]=()
    resource_limits: dict[str,Any]=field(default_factory=dict)
    termination_conditions: tuple[str,...]=()
    success_postconditions: tuple[str,...]=()
    verifier: dict[str,Any]=field(default_factory=dict)
    known_failure_modes: tuple[str,...]=()
    supporting_evidence: tuple[str,...]=()
    contradicting_evidence: tuple[str,...]=()
    qualification_receipt: str=""
    rollback_target: str=""

    @property
    def manifest_digest(self): return digest(self)
