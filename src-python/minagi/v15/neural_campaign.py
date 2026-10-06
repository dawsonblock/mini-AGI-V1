from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json
from typing import Callable, Mapping, Sequence

from egai.common.canonical import canonical_bytes, digest, validate_digest
from minagi.integration.qw3_state import ServedArtifactManifest
from minagi.v14.experiment_v145 import FrozenBaselineGateV145
from .neural_eval import QW3NeuralArm, SealedNeuralAdapterEvaluator, SealedNeuralExperiment

ZERO = "0" * 64


def _hex64(value: str, name: str) -> str:
    value = str(value).lower()
    if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"{name} must be a lowercase SHA-256 hex digest")
    return value


@dataclass(frozen=True)
class NeuralGateSpec:
    required_rings: tuple[str, ...] = ("R0", "R1", "R2")
    minimum_effect: float = 0.0
    minimum_retention: float = 0.95
    bootstrap_iterations: int = 4000
    seed: int = 0
    schema: str = "mini-agi-v15.8-neural-gate-spec-v1"

    def __post_init__(self) -> None:
        if not self.required_rings or any(not str(x) for x in self.required_rings):
            raise ValueError("required_rings must be non-empty")
        if not 0.0 <= float(self.minimum_retention) <= 1.0:
            raise ValueError("minimum_retention must be in [0,1]")
        if int(self.bootstrap_iterations) < 100:
            raise ValueError("bootstrap_iterations must be >= 100")

    @property
    def digest(self) -> str:
        return digest(self)

    def build(self) -> FrozenBaselineGateV145:
        return FrozenBaselineGateV145(
            required_rings=self.required_rings,
            minimum_effect=self.minimum_effect,
            minimum_retention=self.minimum_retention,
            bootstrap_iterations=self.bootstrap_iterations,
            seed=self.seed,
        )


@dataclass(frozen=True)
class FreshTaskCommitmentRef:
    task_id: str
    generation: int
    commitment_digest: str
    schema: str = "mini-agi-v15.8-fresh-task-commitment-ref-v1"

    def __post_init__(self) -> None:
        if not self.task_id:
            raise ValueError("task_id required")
        if int(self.generation) < 1:
            raise ValueError("generation must be >= 1")
        validate_digest(self.commitment_digest)

    @classmethod
    def from_commitment(cls, value) -> "FreshTaskCommitmentRef":
        return cls(str(value.task_id), int(value.generation), str(value.commitment_digest))

    @property
    def digest(self) -> str:
        return digest(self)


@dataclass(frozen=True)
class NeuralArmRunSpec:
    arm: str
    endpoint: str
    epoch_digest: str
    manifest_digest: str
    artifact_root: str
    adapter_set_root: str
    native_adapter_bundle_root: str
    production_identity_digest: str
    schema: str = "mini-agi-v15.8-neural-arm-run-spec-v1"

    def __post_init__(self) -> None:
        if self.arm not in {"A0", "A1"}:
            raise ValueError("arm must be A0 or A1")
        if not (self.endpoint.startswith("http://") or self.endpoint.startswith("https://")):
            raise ValueError("endpoint must use http:// or https://")
        validate_digest(self.epoch_digest)
        validate_digest(self.production_identity_digest)
        for name in ("manifest_digest", "artifact_root", "adapter_set_root", "native_adapter_bundle_root"):
            _hex64(getattr(self, name), name)

    @classmethod
    def from_arm(cls, arm: QW3NeuralArm) -> "NeuralArmRunSpec":
        return cls(
            arm=arm.arm,
            endpoint=str(arm.contract.base_url),
            epoch_digest=arm.manifest.epoch_digest,
            manifest_digest=arm.manifest.manifest_digest,
            artifact_root=arm.manifest.artifact_root,
            adapter_set_root=arm.manifest.adapter_set_root,
            native_adapter_bundle_root=arm.manifest.native_adapter_bundle_root,
            production_identity_digest=arm.production_identity_digest,
        )

    @property
    def digest(self) -> str:
        return digest(self)

    def assert_matches(self, arm: QW3NeuralArm) -> None:
        actual = NeuralArmRunSpec.from_arm(arm)
        if actual != self:
            raise PermissionError(f"{self.arm} runtime arm does not match preregistered run spec")


@dataclass(frozen=True)
class NeuralCampaignPlan:
    run_id: str
    a0: NeuralArmRunSpec
    a1: NeuralArmRunSpec
    task_commitments: tuple[FreshTaskCommitmentRef, ...]
    gate: NeuralGateSpec
    scorer_id: str
    retention_policy_id: str
    security_policy_id: str
    schema: str = "mini-agi-v15.8-neural-campaign-plan-v1"

    def __post_init__(self) -> None:
        if not self.run_id or any(ch.isspace() for ch in self.run_id):
            raise ValueError("run_id must be non-empty and contain no whitespace")
        if self.a0.arm != "A0" or self.a1.arm != "A1":
            raise ValueError("campaign requires ordered A0/A1 arm specs")
        if self.a0.production_identity_digest != self.a1.production_identity_digest:
            raise PermissionError("A0/A1 production identity mismatch in campaign plan")
        if self.a0.adapter_set_root == self.a1.adapter_set_root:
            raise ValueError("campaign A0/A1 adapter roots must differ")
        if self.a1.adapter_set_root != ZERO and self.a1.native_adapter_bundle_root == ZERO:
            raise PermissionError("candidate adapter root lacks native adapter bundle")
        if not self.task_commitments:
            raise ValueError("campaign requires at least one sealed task commitment")
        if len({x.task_id for x in self.task_commitments}) != len(self.task_commitments):
            raise ValueError("duplicate task_id in campaign")
        for label in (self.scorer_id, self.retention_policy_id, self.security_policy_id):
            if not str(label):
                raise ValueError("campaign policy identities must be non-empty")

    @classmethod
    def create(cls, *, run_id: str, a0: QW3NeuralArm, a1: QW3NeuralArm,
               commitments: Sequence, gate: NeuralGateSpec | None = None,
               scorer_id: str = "exact-match-v1",
               retention_policy_id: str = "retention-v1",
               security_policy_id: str = "security-v1") -> "NeuralCampaignPlan":
        return cls(
            run_id=str(run_id),
            a0=NeuralArmRunSpec.from_arm(a0),
            a1=NeuralArmRunSpec.from_arm(a1),
            task_commitments=tuple(FreshTaskCommitmentRef.from_commitment(x) for x in commitments),
            gate=gate or NeuralGateSpec(),
            scorer_id=str(scorer_id),
            retention_policy_id=str(retention_policy_id),
            security_policy_id=str(security_policy_id),
        )

    @property
    def digest(self) -> str:
        return digest(self)

    def canonical_json(self) -> bytes:
        return canonical_bytes(self)


@dataclass(frozen=True)
class QW3LaunchSpec:
    """Reproducible native launch recipe for one governed experiment arm.

    Paths are deployment inputs, not trust anchors. QW3 independently measures
    the bytes and refuses startup when they disagree with the manifest roots.
    """
    arm: str
    executable: str
    model_path: str
    host: str
    port: int
    manifest: ServedArtifactManifest
    kvmem_archive: str | None = None
    adapter_set_artifact: str | None = None
    retrieval_policy_artifact: str | None = None
    skill_policy_artifact: str | None = None
    native_adapter_bundle: str | None = None
    extra_args: tuple[str, ...] = ()
    schema: str = "mini-agi-v15.8-qw3-launch-spec-v1"

    def __post_init__(self) -> None:
        if self.arm not in {"A0", "A1"}:
            raise ValueError("arm must be A0 or A1")
        if not self.executable or not self.model_path or not self.host:
            raise ValueError("executable, model_path, and host are required")
        if not 1 <= int(self.port) <= 65535:
            raise ValueError("port must be in 1..65535")
        optional = (
            (self.manifest.adapter_set_root, self.adapter_set_artifact, "adapter-set"),
            (self.manifest.retrieval_policy_root, self.retrieval_policy_artifact, "retrieval-policy"),
            (self.manifest.skill_policy_root, self.skill_policy_artifact, "skill-policy"),
            (self.manifest.native_adapter_bundle_root, self.native_adapter_bundle, "native-adapter"),
        )
        for root, path, label in optional:
            if root == ZERO and path is not None:
                raise ValueError(f"{label} path supplied for zero root")
            if root != ZERO and path is None:
                raise ValueError(f"{label} path required for nonzero root")

    @property
    def digest(self) -> str:
        return digest(self)

    def argv(self) -> tuple[str, ...]:
        m = self.manifest
        args = [
            self.executable, "serve", "--model", self.model_path,
            "--host", self.host, "--port", str(self.port),
            "--state-epoch-id", m.epoch_digest,
            "--state-manifest-digest", m.manifest_digest,
            "--state-artifact-root", m.artifact_root,
            "--state-adapter-set-root", m.adapter_set_root,
            "--state-foundation-digest", m.foundation_digest,
            "--state-tokenizer-digest", m.tokenizer_digest,
            "--state-kvmem-archive-root", m.kv_archive_root,
            "--state-retrieval-policy-root", m.retrieval_policy_root,
            "--state-skill-policy-root", m.skill_policy_root,
            "--state-runtime-binary-digest", m.runtime_binary_digest,
        ]
        if self.kvmem_archive is not None:
            args += ["--kvmem-archive", self.kvmem_archive]
        if self.adapter_set_artifact is not None:
            args += ["--state-adapter-set-artifact", self.adapter_set_artifact]
        if self.retrieval_policy_artifact is not None:
            args += ["--state-retrieval-policy-artifact", self.retrieval_policy_artifact]
        if self.skill_policy_artifact is not None:
            args += ["--state-skill-policy-artifact", self.skill_policy_artifact]
        if self.native_adapter_bundle is not None:
            args += ["--state-native-adapter-bundle-root", m.native_adapter_bundle_root,
                     "--native-adapter-bundle", self.native_adapter_bundle]
        args += list(self.extra_args)
        return tuple(args)


@dataclass(frozen=True)
class NeuralCampaignResultBundle:
    plan_digest: str
    experiment_digest: str
    baseline_report_digest: str
    pair_digests: tuple[str, ...]
    fresh_task_receipt_digests: tuple[str, ...]
    runtime_evidence_digests: tuple[str, ...]
    candidate_adapter_set_root: str
    candidate_native_adapter_bundle_root: str
    decision: str
    promotion_ready: bool
    schema: str = "mini-agi-v15.8-neural-campaign-result-v1"

    def __post_init__(self) -> None:
        validate_digest(self.plan_digest)
        validate_digest(self.experiment_digest)
        validate_digest(self.baseline_report_digest)
        for value in (*self.pair_digests, *self.fresh_task_receipt_digests, *self.runtime_evidence_digests):
            validate_digest(value)
        _hex64(self.candidate_adapter_set_root, "candidate_adapter_set_root")
        _hex64(self.candidate_native_adapter_bundle_root, "candidate_native_adapter_bundle_root")
        if self.decision not in {"PASS", "BLOCK"}:
            raise ValueError("decision must be PASS or BLOCK")
        if bool(self.promotion_ready) != (self.decision == "PASS"):
            raise ValueError("promotion_ready must equal PASS decision")

    @property
    def digest(self) -> str:
        return digest(self)

    def write_json(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(canonical_bytes(self))
        return path


class PreregisteredNeuralCampaignRunner:
    """Execute a frozen real-weight A0/A1 campaign without promotion authority."""

    def __init__(self, *, evaluator: SealedNeuralAdapterEvaluator, cas):
        self.evaluator = evaluator
        self.cas = cas

    def preregister(self, plan: NeuralCampaignPlan) -> str:
        value = self.cas.put_json(asdict(plan))
        if value != plan.digest:
            raise RuntimeError("campaign plan CAS digest mismatch")
        return value

    def _require_preregistered(self, plan: NeuralCampaignPlan) -> None:
        if not self.cas.exists(plan.digest):
            raise PermissionError("campaign plan must be preregistered before task consumption")
        if self.cas.get_bytes(plan.digest) != plan.canonical_json():
            raise PermissionError("preregistered campaign plan bytes do not match supplied plan")

    @staticmethod
    def _assert_commitments(plan: NeuralCampaignPlan, commitments: Sequence) -> None:
        actual = tuple(FreshTaskCommitmentRef.from_commitment(x) for x in commitments)
        if actual != plan.task_commitments:
            raise PermissionError("runtime task commitments differ from preregistered campaign")

    def run(self, *, plan: NeuralCampaignPlan, commitments: Sequence,
            a0: QW3NeuralArm, a1: QW3NeuralArm,
            score_fn: Callable[[str, str], float] | None = None,
            retention_fn: Callable[[str, str, str], float] | None = None,
            security_fn: Callable[[str, str, str], int] | None = None) -> NeuralCampaignResultBundle:
        self._require_preregistered(plan)
        plan.a0.assert_matches(a0)
        plan.a1.assert_matches(a1)
        self._assert_commitments(plan, commitments)
        # The gate is part of the preregistration. Refuse to run with a mutable
        # evaluator gate that differs from the frozen campaign settings.
        g = self.evaluator.gate
        gate_actual = NeuralGateSpec(tuple(g.required_rings), g.minimum_effect, g.minimum_retention, g.iterations, g.seed)
        if gate_actual != plan.gate:
            raise PermissionError("evaluator gate differs from preregistered campaign")
        experiment = self.evaluator.evaluate(
            commitments=commitments, a0=a0, a1=a1,
            score_fn=score_fn, retention_fn=retention_fn, security_fn=security_fn,
        )
        pair_digests = tuple(x.digest for x in experiment.pairs)
        receipts = tuple(x.fresh_task_receipt_digest for x in experiment.pairs)
        evidence = tuple(
            d for x in experiment.pairs
            for d in (x.a0_runtime_evidence_digest, x.a1_runtime_evidence_digest)
        )
        result = NeuralCampaignResultBundle(
            plan_digest=plan.digest,
            experiment_digest=experiment.digest,
            baseline_report_digest=experiment.baseline_report.digest,
            pair_digests=pair_digests,
            fresh_task_receipt_digests=receipts,
            runtime_evidence_digests=evidence,
            candidate_adapter_set_root=plan.a1.adapter_set_root,
            candidate_native_adapter_bundle_root=plan.a1.native_adapter_bundle_root,
            decision=experiment.baseline_report.decision,
            promotion_ready=experiment.baseline_report.decision == "PASS",
        )
        if self.cas.put_json(asdict(result)) != result.digest:
            raise RuntimeError("campaign result CAS digest mismatch")
        return result
