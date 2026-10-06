from __future__ import annotations

from dataclasses import asdict, replace
import json
import os
from pathlib import Path
from typing import Any

from .canonical import atomic_write_json
from .models import Belief, BeliefStatus, PromotionVerdict


class BeliefRepository:
    """Candidate/promoted belief store with signed-promotion enforcement.

    Raw observations enter the EvidenceLedger directly through the trusted
    ingestor. Interpretations of those observations become Belief candidates
    and cannot enter the promoted store without an independent qualification
    authorization.
    """

    def __init__(self, root: str | os.PathLike):
        self.root = Path(root)
        self.candidates = self.root / "candidates"
        self.promoted = self.root / "promoted"
        self.candidates.mkdir(parents=True, exist_ok=True)
        self.promoted.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _doc(belief: Belief) -> dict[str, Any]:
        doc = asdict(belief)
        doc["status"] = belief.status.value
        doc["digest"] = belief.digest
        return doc

    @staticmethod
    def _load_doc(path: Path) -> Belief:
        doc = json.loads(path.read_text(encoding="utf-8"))
        got = str(doc.pop("digest", ""))
        doc["status"] = BeliefStatus(str(doc["status"]))
        belief = Belief(**doc)
        if belief.digest != got:
            raise RuntimeError("belief digest mismatch")
        return belief

    def register_candidate(self, belief: Belief) -> str:
        if belief.status is not BeliefStatus.CANDIDATE:
            raise ValueError("only candidate beliefs may be registered by the learning plane")
        path = self.candidates / f"{belief.digest.removeprefix('sha256:')}.json"
        if path.exists():
            existing = self._load_doc(path)
            if existing != belief:
                raise RuntimeError("belief candidate digest collision")
            return belief.digest
        atomic_write_json(path, self._doc(belief))
        return belief.digest

    def promote(self, candidate_digest: str, authorization: dict[str, Any], verifier: Any) -> Belief:
        stem = str(candidate_digest).removeprefix("sha256:")
        candidate = self._load_doc(self.candidates / f"{stem}.json")
        body = dict(authorization.get("body") or {})
        if not verifier.verify(authorization, expected_body=body):
            raise PermissionError("invalid promotion authorization")
        if body.get("schema") != "mini-agi-egai-promotion-authorization-v2":
            raise PermissionError("unsupported promotion authorization schema")
        if body.get("candidate_digest") != candidate.digest:
            raise PermissionError("promotion authorization candidate mismatch")
        if body.get("verdict") != PromotionVerdict.APPROVE.value:
            raise PermissionError("promotion authorization is not approval")
        promoted = replace(
            candidate,
            status=BeliefStatus.PROMOTED,
            qualification_digest=str(body["qualification_digest"]),
        )
        out = self.promoted / f"{promoted.belief_id}.json"
        atomic_write_json(out, self._doc(promoted))
        return promoted

    def load_promoted(self, belief_id: str) -> Belief:
        belief = self._load_doc(self.promoted / f"{belief_id}.json")
        if belief.status is not BeliefStatus.PROMOTED:
            raise RuntimeError("promoted belief store contains non-promoted record")
        return belief
