from __future__ import annotations

from dataclasses import asdict, replace
import json
import os
from pathlib import Path
from typing import Any

from .canonical import atomic_write_json
from .models import PromotionVerdict, SkillManifest


class SkillRepository:
    """Candidate/promoted skill store.

    Candidate registration is allowed to the learning plane. Promotion requires
    a signed promotion authorization whose body matches the skill candidate.
    """

    def __init__(self, root: str | os.PathLike):
        self.root = Path(root)
        self.candidates = self.root / "candidates"
        self.promoted = self.root / "promoted"
        self.candidates.mkdir(parents=True, exist_ok=True)
        self.promoted.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _doc(skill: SkillManifest) -> dict[str, Any]:
        return {**asdict(skill), "digest": skill.digest}

    @staticmethod
    def _load_doc(path: Path) -> SkillManifest:
        doc = json.loads(path.read_text(encoding="utf-8"))
        got = str(doc.pop("digest", ""))
        skill = SkillManifest(**doc)
        if skill.digest != got:
            raise RuntimeError("skill manifest digest mismatch")
        return skill

    def register_candidate(self, skill: SkillManifest) -> str:
        path = self.candidates / f"{skill.digest.removeprefix('sha256:')}.json"
        if path.exists():
            existing = self._load_doc(path)
            if existing != skill:
                raise RuntimeError("skill candidate digest collision")
            return skill.digest
        atomic_write_json(path, self._doc(skill))
        return skill.digest

    def promote(self, candidate_digest: str, authorization: dict[str, Any], verifier: Any) -> SkillManifest:
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
        qualified = replace(candidate, qualification_digest=str(body["qualification_digest"]))
        out = self.promoted / f"{qualified.skill_id}-v{qualified.version}.json"
        atomic_write_json(out, self._doc(qualified))
        return qualified

    def load_promoted(self, skill_id: str, version: int) -> SkillManifest:
        return self._load_doc(self.promoted / f"{skill_id}-v{int(version)}.json")
