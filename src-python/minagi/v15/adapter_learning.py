from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
import hashlib
import json
import os
import subprocess
import tempfile
from typing import Callable, Iterable, Mapping, Sequence

from egai.common.canonical import digest, sha256_bytes, validate_digest
from minagi.integration.qw3_state import ServedArtifactManifest
from minagi.v14.experiment_v145 import FrozenArmRecordV145, FrozenBaselineGateV145, FrozenBaselineReportV145


_HEX = set("0123456789abcdef")


def _hex64(value: str, name: str) -> str:
    value = str(value).lower()
    if len(value) != 64 or any(c not in _HEX for c in value):
        raise ValueError(f"{name} must be lowercase sha256 hex")
    return value


def _safe_relpath(value: str) -> str:
    p = Path(value)
    if p.is_absolute() or not value or any(part in {"", ".", ".."} for part in p.parts):
        raise ValueError("adapter file path must be a safe relative path")
    return p.as_posix()


@dataclass(frozen=True)
class AdapterTrainingExample:
    task_family: str
    input_text: str
    target_text: str
    trajectory_digest: str
    evidence_digests: tuple[str, ...]
    verification_receipt_digests: tuple[str, ...]
    schema: str = "mini-agi-v15.2-adapter-training-example-v1"

    def __post_init__(self) -> None:
        validate_digest(self.trajectory_digest)
        if not self.evidence_digests or not self.verification_receipt_digests:
            raise ValueError("adapter training examples require verified evidence")
        for d in (*self.evidence_digests, *self.verification_receipt_digests):
            validate_digest(d)
        if not self.task_family:
            raise ValueError("task_family is required")

    @property
    def digest(self) -> str:
        return digest(self)


@dataclass(frozen=True)
class VerifiedAdapterDataset:
    examples: tuple[AdapterTrainingExample, ...]
    schema: str = "mini-agi-v15.2-verified-adapter-dataset-v1"

    def __post_init__(self) -> None:
        if not self.examples:
            raise ValueError("adapter dataset requires verified examples")
        seen = set()
        for example in self.examples:
            if example.digest in seen:
                raise ValueError("duplicate adapter training example")
            seen.add(example.digest)

    @classmethod
    def from_trajectories(cls, trajectories: Iterable) -> "VerifiedAdapterDataset":
        examples = []
        for t in trajectories:
            evidence = tuple(str(x) for x in getattr(t, "evidence_digests", ()))
            receipts = tuple(str(x) for x in getattr(t, "verification_receipt_digests", ()))
            td = str(getattr(t, "digest"))
            examples.append(AdapterTrainingExample(
                task_family=str(getattr(t, "task_family")),
                input_text=str(getattr(t, "input_text")),
                target_text=str(getattr(t, "repaired_output")),
                trajectory_digest=td,
                evidence_digests=evidence,
                verification_receipt_digests=receipts,
            ))
        return cls(tuple(examples))

    @property
    def digest(self) -> str:
        return digest(self)

    def export_mlx_jsonl(self, directory: str | Path) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "train.jsonl"
        rows = []
        for e in self.examples:
            rows.append({
                "messages": [
                    {"role": "user", "content": e.input_text},
                    {"role": "assistant", "content": e.target_text},
                ],
                "metadata": {
                    "task_family": e.task_family,
                    "trajectory_digest": e.trajectory_digest,
                    "evidence_digests": list(e.evidence_digests),
                    "verification_receipt_digests": list(e.verification_receipt_digests),
                },
            })
        body = "".join(json.dumps(r, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n" for r in rows)
        path.write_text(body, encoding="utf-8")
        return path


@dataclass(frozen=True)
class AdapterTrainingPlan:
    foundation_model_digest: str
    dataset_digest: str
    trainer_identity_digest: str
    backend: str
    rank: int = 16
    scale: float = 20.0
    dropout: float = 0.0
    learning_rate: float = 2e-5
    iterations: int = 200
    seed: int = 0
    target_modules: tuple[str, ...] = ("all-linear",)
    parent_adapter_set_root: str | None = None
    schema: str = "mini-agi-v15.2-adapter-training-plan-v1"

    def __post_init__(self) -> None:
        validate_digest(self.foundation_model_digest)
        validate_digest(self.dataset_digest)
        validate_digest(self.trainer_identity_digest)
        if self.backend not in {"mlx_lm", "peft", "external"}:
            raise ValueError("unsupported adapter backend")
        if self.rank <= 0 or self.scale <= 0 or self.iterations <= 0:
            raise ValueError("adapter training parameters must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("adapter dropout must be in [0,1)")
        if self.learning_rate <= 0:
            raise ValueError("learning rate must be positive")
        if not self.target_modules:
            raise ValueError("target_modules cannot be empty")
        if self.parent_adapter_set_root is not None:
            _hex64(self.parent_adapter_set_root, "parent_adapter_set_root")

    @property
    def digest(self) -> str:
        return digest(self)


@dataclass(frozen=True)
class AdapterFile:
    relative_path: str
    sha256_digest: str
    bytes: int
    schema: str = "mini-agi-v15.2-adapter-file-v1"

    def __post_init__(self) -> None:
        object.__setattr__(self, "relative_path", _safe_relpath(self.relative_path))
        validate_digest(self.sha256_digest)
        if self.bytes < 0:
            raise ValueError("adapter file size cannot be negative")


@dataclass(frozen=True)
class AdapterArtifactManifest:
    training_plan_digest: str
    foundation_model_digest: str
    dataset_digest: str
    backend: str
    files: tuple[AdapterFile, ...]
    schema: str = "mini-agi-v15.2-adapter-artifact-manifest-v1"

    def __post_init__(self) -> None:
        validate_digest(self.training_plan_digest)
        validate_digest(self.foundation_model_digest)
        validate_digest(self.dataset_digest)
        if not self.files:
            raise ValueError("adapter artifact cannot be empty")
        names = [f.relative_path for f in self.files]
        if names != sorted(names) or len(names) != len(set(names)):
            raise ValueError("adapter files must be unique and canonically sorted")

    @property
    def digest(self) -> str:
        return digest(self)


@dataclass(frozen=True)
class AdapterPayloadBinding:
    adapter_manifest_digest: str
    files: tuple[AdapterFile, ...]
    schema: str = "mini-agi-v15.2-adapter-payload-binding-v1"

    def __post_init__(self) -> None:
        validate_digest(self.adapter_manifest_digest)
        if not self.files:
            raise ValueError("adapter payload binding cannot be empty")

    @property
    def digest(self) -> str:
        return digest(self)


@dataclass(frozen=True)
class AdapterSetBundle:
    foundation_model_digest: str
    adapters: tuple[AdapterArtifactManifest, ...]
    schema: str = "mini-agi-v15.2-adapter-set-bundle-v1"

    def __post_init__(self) -> None:
        validate_digest(self.foundation_model_digest)
        if not self.adapters:
            raise ValueError("adapter set requires at least one adapter")
        ds = [a.digest for a in self.adapters]
        if ds != sorted(ds) or len(ds) != len(set(ds)):
            raise ValueError("adapter set must be unique and sorted by digest")
        if any(a.foundation_model_digest != self.foundation_model_digest for a in self.adapters):
            raise ValueError("adapter set cannot mix foundation models")

    @classmethod
    def from_adapters(cls, foundation_model_digest: str,
                      adapters: Iterable[AdapterArtifactManifest]) -> "AdapterSetBundle":
        values = tuple(sorted(tuple(adapters), key=lambda a: a.digest))
        return cls(foundation_model_digest=foundation_model_digest, adapters=values)

    @property
    def digest(self) -> str:
        return digest(self)

    @property
    def root_hex(self) -> str:
        return self.digest.split(":", 1)[1]


class AdapterArtifactBuilder:
    """Materialize an adapter candidate and hash every output byte.

    The supplied runner is deliberately proposal/build authority only.  This
    class neither evaluates nor promotes the output.  It also refuses symlinks
    so a candidate cannot smuggle mutable data outside the artifact directory.
    """

    def __init__(self, *, max_total_bytes: int = 8 * 1024**3):
        self.max_total_bytes = int(max_total_bytes)

    def collect(self, *, output_dir: str | Path, plan: AdapterTrainingPlan) -> AdapterArtifactManifest:
        root = Path(output_dir).resolve()
        if not root.is_dir():
            raise FileNotFoundError(root)
        files: list[AdapterFile] = []
        total = 0
        for p in sorted(root.rglob("*")):
            if p.is_symlink():
                raise ValueError(f"adapter output contains symlink: {p}")
            if not p.is_file():
                continue
            data = p.read_bytes()
            total += len(data)
            if total > self.max_total_bytes:
                raise ValueError("adapter artifact exceeds byte budget")
            files.append(AdapterFile(
                relative_path=p.relative_to(root).as_posix(),
                sha256_digest=sha256_bytes(data),
                bytes=len(data),
            ))
        return AdapterArtifactManifest(
            training_plan_digest=plan.digest,
            foundation_model_digest=plan.foundation_model_digest,
            dataset_digest=plan.dataset_digest,
            backend=plan.backend,
            files=tuple(files),
        )

    def build(self, *, plan: AdapterTrainingPlan,
              runner: Callable[[AdapterTrainingPlan, Path], None],
              output_dir: str | Path) -> AdapterArtifactManifest:
        out = Path(output_dir)
        if out.exists() and any(out.iterdir()):
            raise FileExistsError("adapter output directory must be absent or empty")
        out.mkdir(parents=True, exist_ok=True)
        runner(plan, out)
        return self.collect(output_dir=out, plan=plan)


class MlxLoraCandidateRunner:
    """Operational MLX-LM adapter trainer wrapper.

    It is intentionally injectable/testable.  The default runner executes
    `python -m mlx_lm.lora`; qualification remains external to this class.
    """

    def __init__(self, *, model: str, dataset: VerifiedAdapterDataset,
                 python_executable: str | None = None,
                 run: Callable[..., subprocess.CompletedProcess] | None = None):
        import sys
        self.model = str(model)
        self.dataset = dataset
        self.python_executable = python_executable or sys.executable
        self._run = run or subprocess.run

    def __call__(self, plan: AdapterTrainingPlan, output_dir: Path) -> None:
        if plan.backend != "mlx_lm":
            raise ValueError("MLX runner requires backend=mlx_lm")
        if plan.dataset_digest != self.dataset.digest:
            raise ValueError("training plan dataset digest mismatch")
        data_dir = output_dir.parent / "dataset"
        self.dataset.export_mlx_jsonl(data_dir)
        import yaml
        config = {
            "model": self.model,
            "train": True,
            "fine_tune_type": "lora",
            "data": str(data_dir),
            "adapter_path": str(output_dir),
            "iters": plan.iterations,
            "batch_size": 1,
            "num_layers": 4,
            "learning_rate": plan.learning_rate,
            "seed": plan.seed,
            "mask_prompt": True,
            "grad_checkpoint": True,
            "lora_parameters": {
                "rank": plan.rank,
                "dropout": plan.dropout,
                "scale": plan.scale,
            },
        }
        config_path = output_dir.parent / "mlx_lora_config.yaml"
        config_path.write_text(yaml.safe_dump(config, sort_keys=True), encoding="utf-8")
        cmd = [self.python_executable, "-m", "mlx_lm", "lora", "--config", str(config_path)]
        self._run(cmd, check=True, cwd=str(output_dir.parent), text=True)


@dataclass(frozen=True)
class AdapterArmOutcome:
    output: str
    retention_score: float = 1.0
    security_regressions: int = 0

    def __post_init__(self) -> None:
        if not 0.0 <= float(self.retention_score) <= 1.0:
            raise ValueError("retention_score must be in [0,1]")
        if self.security_regressions < 0:
            raise ValueError("security_regressions cannot be negative")


@dataclass(frozen=True)
class SealedAdapterPairResult:
    task_digest: str
    fresh_task_receipt_digest: str
    a0: FrozenArmRecordV145
    a1: FrozenArmRecordV145
    schema: str = "mini-agi-v15.2-sealed-adapter-pair-result-v1"

    def __post_init__(self) -> None:
        validate_digest(self.task_digest)
        validate_digest(self.fresh_task_receipt_digest)
        if self.a0.task_digest != self.task_digest or self.a1.task_digest != self.task_digest:
            raise ValueError("adapter A0/A1 records are not bound to the same task")

    @property
    def digest(self) -> str:
        return digest(self)


@dataclass(frozen=True)
class SealedAdapterExperiment:
    candidate_adapter_manifest_digest: str
    baseline_adapter_set_root: str
    candidate_adapter_set_root: str
    pairs: tuple[SealedAdapterPairResult, ...]
    baseline_report: FrozenBaselineReportV145
    schema: str = "mini-agi-v15.2-sealed-adapter-experiment-v1"

    def __post_init__(self) -> None:
        validate_digest(self.candidate_adapter_manifest_digest)
        _hex64(self.baseline_adapter_set_root, "baseline_adapter_set_root")
        _hex64(self.candidate_adapter_set_root, "candidate_adapter_set_root")
        if self.baseline_adapter_set_root == self.candidate_adapter_set_root:
            raise ValueError("adapter experiment must compare distinct adapter sets")
        if not self.pairs:
            raise ValueError("adapter experiment requires paired tasks")

    @property
    def digest(self) -> str:
        return digest(self)


class FreshTaskAdapterEvaluator:
    """Sealed paired A0/A1 evaluator for a concrete adapter artifact set."""

    def __init__(self, *, vault, metadata, cas, gate: FrozenBaselineGateV145,
                 consumer_id: str = "sealed-adapter-a0-a1-evaluator"):
        self.vault = vault
        self.metadata = metadata
        self.cas = cas
        self.gate = gate
        self.consumer_id = str(consumer_id)

    @staticmethod
    def _score(output: str, expected: str) -> float:
        return 1.0 if str(output) == str(expected) else 0.0

    @staticmethod
    def _outcome(value) -> AdapterArmOutcome:
        if isinstance(value, AdapterArmOutcome):
            return value
        return AdapterArmOutcome(str(value))

    def evaluate(self, *, commitments: Sequence,
                 candidate_adapter: AdapterArtifactManifest,
                 baseline_adapter_set_root: str,
                 candidate_adapter_set: AdapterSetBundle,
                 foundation_model_digest: str,
                 production_identity_digest: str,
                 a0: Callable[[str, str], AdapterArmOutcome | str],
                 a1: Callable[[str, str], AdapterArmOutcome | str],
                 score_fn: Callable[[str, str], float] | None = None) -> SealedAdapterExperiment:
        validate_digest(foundation_model_digest)
        validate_digest(production_identity_digest)
        _hex64(baseline_adapter_set_root, "baseline_adapter_set_root")
        if candidate_adapter.foundation_model_digest != foundation_model_digest:
            raise PermissionError("candidate adapter foundation identity drift")
        if candidate_adapter_set.foundation_model_digest != foundation_model_digest:
            raise PermissionError("candidate adapter set foundation identity drift")
        if candidate_adapter.digest not in {a.digest for a in candidate_adapter_set.adapters}:
            raise ValueError("candidate adapter is absent from candidate adapter set")
        score_fn = score_fn or self._score
        pairs: list[SealedAdapterPairResult] = []
        for commitment in commitments:
            lease = self.metadata.lease(task_id=commitment.task_id, consumer_id=self.consumer_id)
            task, receipt = self.vault.consume(lease, metadata=self.metadata)
            rd = self.cas.put_json(asdict(receipt))
            if rd != receipt.digest:
                raise RuntimeError("fresh-task receipt CAS digest mismatch")
            required = {"family_id", "ring", "input", "expected"}
            if set(task) < required:
                raise ValueError("sealed adapter task is missing required fields")
            family, ring = str(task["family_id"]), str(task["ring"])
            inp, expected = str(task["input"]), str(task["expected"])
            o0, o1 = self._outcome(a0(family, inp)), self._outcome(a1(family, inp))
            s0, s1 = float(score_fn(o0.output, expected)), float(score_fn(o1.output, expected))
            if not (0.0 <= s0 <= 1.0 and 0.0 <= s1 <= 1.0):
                raise ValueError("adapter evaluation scores must be in [0,1]")
            a0r = FrozenArmRecordV145(receipt.task_digest, family, ring,
                                      foundation_model_digest, production_identity_digest,
                                      "A0", s0, o0.security_regressions, o0.retention_score)
            a1r = FrozenArmRecordV145(receipt.task_digest, family, ring,
                                      foundation_model_digest, production_identity_digest,
                                      "A1", s1, o1.security_regressions, o1.retention_score)
            pair = SealedAdapterPairResult(receipt.task_digest, receipt.digest, a0r, a1r)
            self.cas.put_json(asdict(pair))
            pairs.append(pair)
            self.metadata.close(commitment.task_id)
        report = self.gate.evaluate(tuple(p.a0 for p in pairs), tuple(p.a1 for p in pairs))
        exp = SealedAdapterExperiment(
            candidate_adapter_manifest_digest=candidate_adapter.digest,
            baseline_adapter_set_root=baseline_adapter_set_root,
            candidate_adapter_set_root=candidate_adapter_set.root_hex,
            pairs=tuple(pairs), baseline_report=report,
        )
        ed = self.cas.put_json(asdict(exp))
        if ed != exp.digest:
            raise RuntimeError("sealed adapter experiment CAS digest mismatch")
        return exp


@dataclass(frozen=True)
class QualifiedAdapterSetCandidate:
    training_plan_digest: str
    foundation_model_digest: str
    adapter_manifest_digest: str
    adapter_payload_binding_digest: str
    adapter_set_bundle_digest: str
    sealed_experiment_digest: str
    frozen_baseline_report_digest: str
    adapter_set_root: str
    decision: str
    schema: str = "mini-agi-v15.2-qualified-adapter-set-candidate-v1"

    def __post_init__(self) -> None:
        for d in (
            self.training_plan_digest, self.foundation_model_digest, self.adapter_manifest_digest,
            self.adapter_payload_binding_digest, self.adapter_set_bundle_digest, self.sealed_experiment_digest,
            self.frozen_baseline_report_digest,
        ):
            validate_digest(d)
        _hex64(self.adapter_set_root, "adapter_set_root")
        if self.decision not in {"PASS", "BLOCK"}:
            raise ValueError("invalid adapter qualification decision")

    @property
    def digest(self) -> str:
        return digest(self)


class GovernedAdapterLearningLoop:
    """Build/qualification bridge for in-weight adapter candidates.

    This layer does not possess promotion authority.  A PASS creates a
    content-addressed candidate that *may* be carried into the existing
    qualification/promotion/StateEpoch chain.  BLOCK can never alter serving.
    """

    def __init__(self, *, cas):
        self.cas = cas

    def prepare(self, *, plan: AdapterTrainingPlan,
                adapter: AdapterArtifactManifest,
                adapter_set: AdapterSetBundle,
                experiment: SealedAdapterExperiment,
                artifact_dir: str | Path) -> QualifiedAdapterSetCandidate:
        if adapter.training_plan_digest != plan.digest:
            raise ValueError("adapter artifact is not bound to training plan")
        if adapter.dataset_digest != plan.dataset_digest:
            raise ValueError("adapter artifact dataset binding mismatch")
        if adapter.foundation_model_digest != plan.foundation_model_digest:
            raise ValueError("adapter artifact foundation binding mismatch")
        if experiment.candidate_adapter_manifest_digest != adapter.digest:
            raise ValueError("sealed experiment is bound to a different adapter artifact")
        if experiment.candidate_adapter_set_root != adapter_set.root_hex:
            raise ValueError("sealed experiment is bound to a different adapter set")
        root = Path(artifact_dir).resolve()
        published = []
        for row in adapter.files:
            path = (root / row.relative_path).resolve()
            if not path.is_relative_to(root) or not path.is_file() or path.is_symlink():
                raise ValueError("adapter payload path is missing or unsafe")
            data = path.read_bytes()
            actual = sha256_bytes(data)
            if actual != row.sha256_digest or len(data) != row.bytes:
                raise RuntimeError("adapter payload bytes no longer match qualified manifest")
            if self.cas.put_bytes(data) != row.sha256_digest:
                raise RuntimeError("adapter payload CAS publication mismatch")
            published.append(row)
        payload_binding = AdapterPayloadBinding(adapter.digest, tuple(published))
        payload_binding_digest = self.cas.put_json(asdict(payload_binding))
        if payload_binding_digest != payload_binding.digest:
            raise RuntimeError("adapter payload binding CAS digest mismatch")
        for obj, expected in (
            (plan, plan.digest), (adapter, adapter.digest),
            (adapter_set, adapter_set.digest), (experiment, experiment.digest),
            (experiment.baseline_report, experiment.baseline_report.digest),
        ):
            actual = self.cas.put_json(asdict(obj))
            if actual != expected:
                raise RuntimeError("adapter learning CAS digest mismatch")
        q = QualifiedAdapterSetCandidate(
            training_plan_digest=plan.digest,
            foundation_model_digest=plan.foundation_model_digest,
            adapter_manifest_digest=adapter.digest,
            adapter_payload_binding_digest=payload_binding.digest,
            adapter_set_bundle_digest=adapter_set.digest,
            sealed_experiment_digest=experiment.digest,
            frozen_baseline_report_digest=experiment.baseline_report.digest,
            adapter_set_root=adapter_set.root_hex,
            decision=experiment.baseline_report.decision,
        )
        qd = self.cas.put_json(asdict(q))
        if qd != q.digest:
            raise RuntimeError("qualified adapter candidate CAS digest mismatch")
        return q

    @staticmethod
    def prepare_served_manifest(*, base: ServedArtifactManifest,
                                qualified: QualifiedAdapterSetCandidate) -> ServedArtifactManifest:
        if qualified.decision != "PASS":
            raise PermissionError("blocked adapter experiment cannot alter served state")
        if base.foundation_digest != qualified.foundation_model_digest.split(":", 1)[1]:
            raise PermissionError("qualified adapter foundation does not match served foundation")
        return replace(base, adapter_set_root=qualified.adapter_set_root, native_adapter_bundle_root="0" * 64)
