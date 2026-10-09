from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable, Mapping, Sequence

from egai.common.canonical import digest, validate_digest
from minagi.integration.qw3_state import QW3RuntimeState, ServedArtifactManifest
from minagi.v14.experiment_v145 import FrozenArmRecordV145, FrozenBaselineGateV145, FrozenBaselineReportV145

ZERO = "0" * 64


def _shared_production_identity(manifest: ServedArtifactManifest) -> str:
    """Identity that must stay frozen across A0/A1 except adapter state.

    Epoch/manifest/artifact roots are intentionally excluded because they must
    change when the authorized adapter changes. Everything else is frozen.
    """
    return digest({
        "schema": "mini-agi-v15.7-neural-production-identity-v1",
        "foundation_digest": manifest.foundation_digest,
        "tokenizer_digest": manifest.tokenizer_digest,
        "kv_archive_root": manifest.kv_archive_root,
        "retrieval_policy_root": manifest.retrieval_policy_root,
        "skill_policy_root": manifest.skill_policy_root,
        "runtime_binary_digest": manifest.runtime_binary_digest,
    })


def _assert_only_adapter_delta(a0: ServedArtifactManifest, a1: ServedArtifactManifest) -> None:
    fields = (
        "foundation_digest", "tokenizer_digest", "kv_archive_root",
        "retrieval_policy_root", "skill_policy_root", "runtime_binary_digest",
    )
    changed = [name for name in fields if getattr(a0, name) != getattr(a1, name)]
    if changed:
        raise PermissionError("A0/A1 changed non-adapter serving state: " + ", ".join(changed))
    if a0.adapter_set_root == a1.adapter_set_root:
        raise ValueError("A0/A1 must use distinct adapter_set_root values")
    if a1.adapter_set_root != ZERO and a1.native_adapter_bundle_root == ZERO:
        raise PermissionError("A1 nonzero adapter set is not backed by a native adapter bundle")


@dataclass(frozen=True)
class NeuralRuntimeEvidence:
    arm: str
    epoch_digest: str
    manifest_digest: str
    artifact_root: str
    adapter_set_root: str
    native_adapter_bundle_root: str
    runtime_state_digest: str
    output_digest: str
    schema: str = "mini-agi-v15.7-neural-runtime-evidence-v1"

    def __post_init__(self) -> None:
        if self.arm not in {"A0", "A1"}:
            raise ValueError("arm must be A0 or A1")
        for value in (self.epoch_digest,):
            validate_digest(value)
        validate_digest(self.runtime_state_digest)
        validate_digest(self.output_digest)

    @property
    def digest(self) -> str:
        return digest(self)


@dataclass(frozen=True)
class NeuralArmResult:
    output: str
    evidence: NeuralRuntimeEvidence
    retention_score: float = 1.0
    security_regressions: int = 0

    def __post_init__(self) -> None:
        if not 0.0 <= float(self.retention_score) <= 1.0:
            raise ValueError("retention_score must be in [0,1]")
        if self.security_regressions < 0:
            raise ValueError("security_regressions cannot be negative")


class QW3NeuralArm:
    """One independently verified native QW3 arm for real-weight evaluation.

    The contract is re-verified before every generation.  This class does not
    train, qualify, or promote anything; it only records exactly which runtime
    state produced an output.
    """

    def __init__(self, *, arm: str, lease, manifest: ServedArtifactManifest, contract,
                 model_call: Callable[[Mapping, Mapping[str, str]], Mapping | str] | None = None):
        if arm not in {"A0", "A1"}:
            raise ValueError("arm must be A0 or A1")
        self.arm = arm
        self.lease = lease
        self.manifest = manifest
        self.contract = contract
        self.model_call = model_call

    @property
    def production_identity_digest(self) -> str:
        return _shared_production_identity(self.manifest)

    def verify(self) -> QW3RuntimeState:
        state = self.contract.verify_loaded_state(lease=self.lease, manifest=self.manifest)
        if not state.runtime_closure_verified:
            raise PermissionError("neural evaluation requires measured QW3 runtime closure")
        measured = (
            state.measured_foundation_digest, state.measured_tokenizer_digest,
            state.measured_runtime_binary_digest, state.measured_kvmem_archive_root,
            state.measured_adapter_set_root, state.measured_retrieval_policy_root,
            state.measured_skill_policy_root,
        )
        expected = (
            self.manifest.foundation_digest, self.manifest.tokenizer_digest,
            self.manifest.runtime_binary_digest, self.manifest.kv_archive_root,
            self.manifest.adapter_set_root, self.manifest.retrieval_policy_root,
            self.manifest.skill_policy_root,
        )
        if measured != expected:
            raise PermissionError("neural evaluation runtime measurements do not match served manifest")
        return state

    @staticmethod
    def _extract_output(value: Mapping | str) -> str:
        if isinstance(value, str):
            return value
        try:
            return str(value["choices"][0]["message"]["content"])
        except Exception as exc:
            raise RuntimeError("unexpected QW3 chat response shape") from exc

    def run(self, *, task_family: str, input_text: str,
            retention_score: float = 1.0, security_regressions: int = 0) -> NeuralArmResult:
        state = self.verify()
        headers = self.contract.request_headers(lease=self.lease, manifest=self.manifest)
        payload = {
            "model": "governed",
            "metadata": {"task_family": str(task_family), "evaluation_arm": self.arm},
            "messages": [{"role": "user", "content": str(input_text)}],
        }
        if self.model_call is None:
            response = self.contract.post_json(
                path="/v1/chat/completions", payload=payload,
                lease=self.lease, manifest=self.manifest,
            )
        else:
            response = self.model_call(payload, headers)
        output = self._extract_output(response)
        state_digest = digest({
            "schema": "mini-agi-v15.7-qw3-runtime-state-evidence-v1",
            **asdict(state),
        })
        evidence = NeuralRuntimeEvidence(
            arm=self.arm,
            epoch_digest=self.manifest.epoch_digest,
            manifest_digest=self.manifest.manifest_digest,
            artifact_root=self.manifest.artifact_root,
            adapter_set_root=self.manifest.adapter_set_root,
            native_adapter_bundle_root=self.manifest.native_adapter_bundle_root,
            runtime_state_digest=state_digest,
            output_digest=digest({"output": output}),
        )
        return NeuralArmResult(output, evidence, retention_score, security_regressions)


@dataclass(frozen=True)
class SealedNeuralPair:
    task_digest: str
    fresh_task_receipt_digest: str
    a0_record: FrozenArmRecordV145
    a1_record: FrozenArmRecordV145
    a0_runtime_evidence_digest: str
    a1_runtime_evidence_digest: str
    schema: str = "mini-agi-v15.7-sealed-neural-pair-v1"

    def __post_init__(self) -> None:
        for value in (self.task_digest, self.fresh_task_receipt_digest,
                      self.a0_runtime_evidence_digest, self.a1_runtime_evidence_digest):
            validate_digest(value)

    @property
    def digest(self) -> str:
        return digest(self)


@dataclass(frozen=True)
class SealedNeuralExperiment:
    baseline_manifest_digest: str
    candidate_manifest_digest: str
    baseline_adapter_set_root: str
    candidate_adapter_set_root: str
    production_identity_digest: str
    pairs: tuple[SealedNeuralPair, ...]
    baseline_report: FrozenBaselineReportV145
    schema: str = "mini-agi-v15.7-sealed-neural-experiment-v1"

    @property
    def digest(self) -> str:
        return digest(self)


class SealedNeuralAdapterEvaluator:
    """Fresh-task A0/A1 harness whose arms are actual governed QW3 states."""

    def __init__(self, *, vault, metadata, cas, gate: FrozenBaselineGateV145,
                 consumer_id: str = "sealed-neural-qw3-a0-a1"):
        self.vault = vault
        self.metadata = metadata
        self.cas = cas
        self.gate = gate
        self.consumer_id = str(consumer_id)

    @staticmethod
    def _score(output: str, expected: str) -> float:
        return 1.0 if str(output) == str(expected) else 0.0

    def evaluate(self, *, commitments: Sequence, a0: QW3NeuralArm, a1: QW3NeuralArm,
                 score_fn: Callable[[str, str], float] | None = None,
                 retention_fn: Callable[[str, str, str], float] | None = None,
                 security_fn: Callable[[str, str, str], int] | None = None) -> SealedNeuralExperiment:
        _assert_only_adapter_delta(a0.manifest, a1.manifest)
        if a0.production_identity_digest != a1.production_identity_digest:
            raise PermissionError("A0/A1 production identity mismatch")
        score_fn = score_fn or self._score
        pairs = []
        records0, records1 = [], []
        for commitment in commitments:
            lease = self.metadata.lease(task_id=commitment.task_id, consumer_id=self.consumer_id)
            task, receipt = self.vault.consume(lease, metadata=self.metadata)
            receipt_digest = self.cas.put_json(asdict(receipt))
            if receipt_digest != receipt.digest:
                raise RuntimeError("fresh-task receipt CAS digest mismatch")
            required = {"family_id", "ring", "input", "expected"}
            if set(task) < required:
                raise ValueError("sealed neural task missing required fields")
            family, ring = str(task["family_id"]), str(task["ring"])
            inp, expected = str(task["input"]), str(task["expected"])
            r0 = a0.run(task_family=family, input_text=inp)
            r1_ret = 1.0 if retention_fn is None else float(retention_fn(family, inp, expected))
            r1_sec = 0 if security_fn is None else int(security_fn(family, inp, expected))
            r1 = a1.run(task_family=family, input_text=inp,
                        retention_score=r1_ret, security_regressions=r1_sec)
            for ev in (r0.evidence, r1.evidence):
                if self.cas.put_json(asdict(ev)) != ev.digest:
                    raise RuntimeError("runtime evidence CAS digest mismatch")
            s0, s1 = float(score_fn(r0.output, expected)), float(score_fn(r1.output, expected))
            if not (0.0 <= s0 <= 1.0 and 0.0 <= s1 <= 1.0):
                raise ValueError("neural evaluation scores must be in [0,1]")
            rec0 = FrozenArmRecordV145(receipt.task_digest, family, ring,
                "sha256:" + a0.manifest.foundation_digest, a0.production_identity_digest,
                "A0", s0, 0, 1.0)
            rec1 = FrozenArmRecordV145(receipt.task_digest, family, ring,
                "sha256:" + a1.manifest.foundation_digest, a1.production_identity_digest,
                "A1", s1, r1.security_regressions, r1.retention_score)
            records0.append(rec0); records1.append(rec1)
            pair = SealedNeuralPair(
                receipt.task_digest, receipt.digest, rec0, rec1,
                r0.evidence.digest, r1.evidence.digest,
            )
            if self.cas.put_json(asdict(pair)) != pair.digest:
                raise RuntimeError("sealed neural pair CAS digest mismatch")
            pairs.append(pair)
        report = self.gate.evaluate(tuple(records0), tuple(records1))
        if self.cas.put_json(asdict(report)) != report.digest:
            raise RuntimeError("neural baseline report CAS digest mismatch")
        experiment = SealedNeuralExperiment(
            a0.manifest.manifest_digest, a1.manifest.manifest_digest,
            a0.manifest.adapter_set_root, a1.manifest.adapter_set_root,
            a0.production_identity_digest, tuple(pairs), report,
        )
        if self.cas.put_json(asdict(experiment)) != experiment.digest:
            raise RuntimeError("sealed neural experiment CAS digest mismatch")
        return experiment
