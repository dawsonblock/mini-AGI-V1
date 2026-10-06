from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

from egai.common.canonical import digest
from minagi.integration.qw3_state import RuntimeStateMismatch, ServedArtifactManifest
from .retrieval_policy import EpisodicMemoryRecord, RetrievalPolicy, RetrievalPolicyRuntime
from .runtime_closure import sha256_file
from .skill_ir import SkillPolicyRuntime, skill_policy_bundle_from_mapping


@dataclass(frozen=True)
class AgentExecutionReceipt:
    epoch_digest: str
    served_manifest_digest: str
    artifact_root: str
    retrieval_policy_root: str
    skill_policy_root: str
    task_family: str
    input_digest: str
    output_digest: str
    route: str
    skill_id: str | None
    retrieved_record_digests: tuple[str, ...]
    retrieved_evidence_digests: tuple[str, ...]
    schema: str = "mini-agi-v15.6-agent-execution-receipt-v1"

    @property
    def digest(self) -> str:
        return digest(self)


@dataclass(frozen=True)
class AgentExecutionResult:
    output: str
    receipt: AgentExecutionReceipt


class GovernedAgentRuntime:
    """Execute bound SkillIR/retrieval policy under an existing StateEpoch lease.

    This runtime owns no mutation or promotion authority. Before every task it:
    (1) re-hashes the physical policy artifacts, and (2) asks the existing
    governed-serving contract to verify the native QW3 state for the lease.
    A matching SkillIR executes locally; otherwise retrieved verified episodes
    are supplied as bounded context to the governed QW3 request.
    """

    def __init__(self, *, lease, manifest: ServedArtifactManifest, serving_contract,
                 skill_policy_artifact: str | Path | None,
                 retrieval_policy_artifact: str | Path | None,
                 memory_records: Iterable[EpisodicMemoryRecord] = (),
                 model_call: Callable[[Mapping[str, object], Mapping[str, str]], str] | None = None):
        self.lease = lease
        self.manifest = manifest
        self.serving_contract = serving_contract
        self.skill_path = None if skill_policy_artifact is None else Path(skill_policy_artifact)
        self.retrieval_path = None if retrieval_policy_artifact is None else Path(retrieval_policy_artifact)
        self.memory_records = tuple(memory_records)
        self.model_call = model_call
        self._skill_runtime: SkillPolicyRuntime | None = None
        self._retrieval_runtime: RetrievalPolicyRuntime | None = None
        self._load_bound_policies()

    @staticmethod
    def _require_regular(path: Path, label: str) -> None:
        if path.is_symlink() or not path.is_file():
            raise RuntimeStateMismatch(f"{label} must be a non-symlink regular file")

    @staticmethod
    def _input_digest(text: str) -> str:
        return hashlib.sha256(str(text).encode("utf-8")).hexdigest()

    def _verify_file_root(self, path: Path | None, expected: str, label: str) -> None:
        zero = "0" * 64
        if expected == zero:
            if path is not None:
                raise RuntimeStateMismatch(f"{label} zero-root cannot have a supplied artifact")
            return
        if path is None:
            raise RuntimeStateMismatch(f"{label} artifact required by nonzero root")
        self._require_regular(path, label)
        actual = sha256_file(path)
        if actual != expected:
            raise RuntimeStateMismatch(f"{label} physical root mismatch: expected {expected} got {actual}")

    def _load_bound_policies(self) -> None:
        self._verify_file_root(self.skill_path, self.manifest.skill_policy_root, "skill_policy")
        self._verify_file_root(self.retrieval_path, self.manifest.retrieval_policy_root, "retrieval_policy")
        if self.skill_path is not None:
            raw = json.loads(self.skill_path.read_text("utf-8"))
            raw = raw.get("value", raw) if isinstance(raw, dict) else raw
            bundle = skill_policy_bundle_from_mapping(raw)
            if bundle.root_hex != self.manifest.skill_policy_root:
                raise RuntimeStateMismatch("skill_policy canonical digest does not match physical root")
            self._skill_runtime = SkillPolicyRuntime(bundle)
        if self.retrieval_path is not None:
            raw = json.loads(self.retrieval_path.read_text("utf-8"))
            raw = raw.get("value", raw) if isinstance(raw, dict) else raw
            policy = RetrievalPolicy.from_mapping(raw)
            if policy.root_hex != self.manifest.retrieval_policy_root:
                raise RuntimeStateMismatch("retrieval_policy canonical digest does not match physical root")
            self._retrieval_runtime = RetrievalPolicyRuntime(policy, self.memory_records)

    def _preflight(self) -> dict[str, str]:
        # Re-measure every call so post-start tampering cannot remain resident.
        self._verify_file_root(self.skill_path, self.manifest.skill_policy_root, "skill_policy")
        self._verify_file_root(self.retrieval_path, self.manifest.retrieval_policy_root, "retrieval_policy")
        return self.serving_contract.request_headers(lease=self.lease, manifest=self.manifest)

    def run(self, *, task_family: str, input_text: str) -> AgentExecutionResult:
        headers = self._preflight()
        hits = () if self._retrieval_runtime is None else self._retrieval_runtime.retrieve(
            task_family=task_family, query=input_text,
        )
        skill = None if self._skill_runtime is None else self._skill_runtime.bundle.lookup(task_family)
        if skill is not None:
            output = self._skill_runtime.executor.execute(skill, input_text)
            route = "skill_ir"
            skill_id = skill.skill_id
        else:
            route = "qw3"
            skill_id = None
            context = RetrievalPolicyRuntime.prompt_context(hits)
            payload = {
                "model": "governed",
                "messages": [
                    {"role": "system", "content": "Use verified episodic context when relevant."},
                    {"role": "user", "content": (context + "\n\n" if context else "") + str(input_text)},
                ],
            }
            if self.model_call is not None:
                output = str(self.model_call(payload, headers))
            else:
                response = self.serving_contract.post_json(
                    path="/v1/chat/completions", payload=payload, lease=self.lease, manifest=self.manifest,
                )
                try:
                    output = str(response["choices"][0]["message"]["content"])
                except Exception as exc:
                    raise RuntimeError("unexpected QW3 chat response shape") from exc
        receipt = AgentExecutionReceipt(
            epoch_digest=self.manifest.epoch_digest,
            served_manifest_digest=self.manifest.manifest_digest,
            artifact_root=self.manifest.artifact_root,
            retrieval_policy_root=self.manifest.retrieval_policy_root,
            skill_policy_root=self.manifest.skill_policy_root,
            task_family=str(task_family),
            input_digest=self._input_digest(input_text),
            output_digest=self._input_digest(output),
            route=route,
            skill_id=skill_id,
            retrieved_record_digests=tuple(h.record.digest for h in hits),
            retrieved_evidence_digests=tuple(h.record.evidence_digest for h in hits),
        )
        return AgentExecutionResult(output, receipt)
