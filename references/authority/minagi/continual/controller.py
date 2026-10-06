from __future__ import annotations

import json
import os
import shutil
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

import torch

from .adapters import AdapterMetadata, LowRankAdapter
from .artifacts import save_adapter_artifact, load_adapter_artifact, adapter_artifact_digest
from .authority import LearningAuthorityPolicy
from ..authority import WriterLease
from ..candidate_pipeline import (
    CandidateLedger, QualificationPolicy, Ed25519PromotionSigner,
    Ed25519PromotionVerifier, artifact_digest,
)
from ..memory import EpisodicMemory


@dataclass(frozen=True)
class AdapterCandidate:
    candidate_id: str
    artifact_dir: str
    digest: str
    base_generation: str


class ControlledContinualManager:
    """High-level control plane for v6's four-speed learning policy.

    This class intentionally does not expose a method that directly promotes
    core/expert mutations.  Neural changes become candidate artifacts, receive
    evaluation evidence, and only then can an external signer issue a receipt.
    """

    def __init__(self, root: str | os.PathLike, *,
                 authority: LearningAuthorityPolicy | None = None,
                 embedder=None):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.authority = authority or LearningAuthorityPolicy()
        self.memory = EpisodicMemory(str(self.root / "memory.sqlite3"), embedder=embedder)
        self.ledger = CandidateLedger(self.root / "authority")
        self.candidates = self.root / "candidates"
        self.stable = self.root / "stable"
        self.candidates.mkdir(exist_ok=True); self.stable.mkdir(exist_ok=True)

    def remember(self, role: str, text: str, *, source: str = "interaction",
                 importance: float = 1.0, verification: str | None = None,
                 valid_from: float | None = None, valid_to: float | None = None,
                 supersedes: int | None = None, contradiction_of: int | None = None,
                 metadata: dict | None = None) -> int | None:
        self.authority.assert_memory_write()
        return self.memory.append(
            role, text, source=source, importance=importance, metadata=metadata,
            verification=verification, valid_from=valid_from, valid_to=valid_to,
            supersedes=supersedes, contradiction_of=contradiction_of,
        )

    def stage_adapter(self, adapter: LowRankAdapter, metadata: AdapterMetadata, *,
                      base_generation: str, centroid: torch.Tensor | None = None,
                      source: str = "adapter-learning", dataset_digest: str | None = None,
                      extra: dict | None = None, actor: str = "learner") -> AdapterCandidate:
        self.authority.assert_adapter_update()
        cid = "adapter-" + uuid.uuid4().hex
        path = self.candidates / cid
        save_adapter_artifact(path, adapter, metadata, centroid=centroid, extra=extra)
        digest = adapter_artifact_digest(path)
        # Candidate state mutations are serialized by an OS-level writer lease.
        with WriterLease(self.root / "authority", purpose="candidate-ledger"):
            self.ledger = CandidateLedger(self.root / "authority")
            self.ledger.propose(base_generation=str(base_generation), source=source,
                                dataset_digest=dataset_digest, candidate_id=cid, actor=actor)
            self.ledger.append(cid, "trained", actor=actor, evidence={
                "artifact_sha256": digest, "artifact_kind": "directory",
                "adapter_id": metadata.id,
            })
        return AdapterCandidate(cid, str(path), digest, str(base_generation))

    def evaluate_adapter(self, candidate: AdapterCandidate, metrics: dict, *,
                         actor: str = "evaluator") -> dict:
        digest = adapter_artifact_digest(candidate.artifact_dir)
        if digest != candidate.digest:
            raise RuntimeError("candidate artifact changed after staging")
        with WriterLease(self.root / "authority", purpose="candidate-ledger"):
            self.ledger = CandidateLedger(self.root / "authority")
            return self.ledger.append(candidate.candidate_id, "evaluated", actor=actor,
                                      evidence={"metrics": metrics, "artifact_sha256": digest})

    def qualify_adapter(self, candidate: AdapterCandidate, metrics: dict,
                        policy: QualificationPolicy, *, actor: str = "qualifier") -> dict:
        digest = adapter_artifact_digest(candidate.artifact_dir)
        if digest != candidate.digest:
            raise RuntimeError("candidate artifact changed before qualification")
        with WriterLease(self.root / "authority", purpose="candidate-ledger"):
            self.ledger = CandidateLedger(self.root / "authority")
            return self.ledger.qualify(candidate.candidate_id, metrics, policy, actor=actor)

    def promote_adapter(self, candidate: AdapterCandidate, signer: Ed25519PromotionSigner, *,
                        actor: str = "promotion-authority") -> dict:
        if self.authority.require_qualification_for_persistent_neural_state is False:
            raise RuntimeError("v6 refuses unsigned persistent neural promotion in controlled mode")
        src = Path(candidate.artifact_dir)
        digest = adapter_artifact_digest(src)
        if digest != candidate.digest:
            raise RuntimeError("candidate artifact changed before promotion")
        dst = self.stable / candidate.candidate_id
        if dst.exists():
            raise FileExistsError(dst)
        pending = self.stable / (".pending-" + candidate.candidate_id)
        shutil.rmtree(pending, ignore_errors=True)
        # Materialize and verify the production copy *before* consuming the
        # one-way qualified->promoted ledger transition.  A crash after the
        # receipt but before rename leaves a digest-valid pending copy that can
        # be recovered; it does not leave a promoted record with no artifact.
        shutil.copytree(src, pending)
        copied, _ = artifact_digest(pending)
        if copied != digest:
            shutil.rmtree(pending, ignore_errors=True)
            raise RuntimeError("promotion staging copy failed digest verification")
        try:
            with WriterLease(self.root / "authority", purpose="candidate-ledger"):
                self.ledger = CandidateLedger(self.root / "authority")
                receipt = signer.issue(
                    self.ledger, candidate.candidate_id, artifact_path=str(pending),
                    expected_base_generation=candidate.base_generation, actor=actor,
                )
            os.replace(pending, dst)
        except Exception:
            # Preserve a valid pending copy after a successfully issued receipt
            # for operator recovery; otherwise clean up ordinary pre-sign errors.
            if self.ledger.state(candidate.candidate_id) != "promoted":
                shutil.rmtree(pending, ignore_errors=True)
            raise
        active = {
            "version": 1,
            "candidate_id": candidate.candidate_id,
            "artifact_sha256": digest,
            "base_generation": candidate.base_generation,
            "receipt": receipt,
            "published_at": time.time(),
        }
        tmp = self.root / "active.json.tmp"
        tmp.write_text(json.dumps(active, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        os.replace(tmp, self.root / "active.json")
        return active

    def load_active_adapter(self, verifier: Ed25519PromotionVerifier):
        doc = json.loads((self.root / "active.json").read_text(encoding="utf-8"))
        path = self.stable / str(doc["candidate_id"])
        digest = adapter_artifact_digest(path)
        if digest != doc.get("artifact_sha256"):
            raise RuntimeError("active adapter digest mismatch")
        if not verifier.verify_receipt(doc["receipt"], artifact_path=str(path)):
            raise RuntimeError("active adapter promotion receipt is invalid")
        return load_adapter_artifact(path)
