from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, Sequence
import json, os, time
import numpy as np
from .util import canonical_json, sha256_bytes, tree_digest

@dataclass(frozen=True)
class SkillAdapterManifest:
    skill_id: str
    artifact_path: str
    artifact_digest: str
    base_generation: str
    centroid: tuple[float, ...]
    description: str
    evidence_digest: str
    status: str = "candidate"
    created_at: float = 0.0
    parents: tuple[str, ...] = ()

    @classmethod
    def create(cls, *, skill_id: str, artifact_path: str, base_generation: str,
               centroid: Sequence[float], description: str,
               evidence_digest: str = "", status: str = "candidate",
               parents: Iterable[str] = ()):
        return cls(
            skill_id, str(Path(artifact_path).resolve()), tree_digest(artifact_path),
            base_generation, tuple(map(float, centroid)), description,
            evidence_digest, status, time.time(), tuple(sorted(set(parents))),
        )

    @property
    def digest(self) -> str:
        return sha256_bytes(canonical_json(asdict(self)).encode())

class GradientFreeSkillRouter:
    def __init__(self, manifests: Sequence[SkillAdapterManifest]):
        self.manifests = list(manifests)

    @staticmethod
    def _cos(a, b) -> float:
        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        return float(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12))

    def route(self, embedding: Sequence[float], *, top_k: int = 1,
              min_score: float = -1.0):
        rows = [
            (self._cos(embedding, m.centroid), m)
            for m in self.manifests if m.status == "promoted"
        ]
        rows = [x for x in rows if x[0] >= min_score]
        rows.sort(key=lambda x: (-x[0], x[1].skill_id))
        return rows[:max(0, int(top_k))]

class SkillAdapterBank:
    """Immutable adapter manifests + explicit active set.

    v4 defaults to top-1 selection and does not assume arbitrary LoRA
    composition is safe.
    """
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.manifests = self.root / "manifests"
        self.manifests.mkdir(exist_ok=True)
        self.active = self.root / "active.json"

    def register(self, manifest: SkillAdapterManifest):
        if tree_digest(manifest.artifact_path) != manifest.artifact_digest:
            raise ValueError("adapter artifact digest mismatch")
        path = self.manifests / f"{manifest.skill_id}-{manifest.digest[:12]}.json"
        doc = {"manifest": asdict(manifest), "digest": manifest.digest}
        if path.exists() and json.loads(path.read_text()) != doc:
            raise ValueError("manifest collision")
        if not path.exists():
            path.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
        return path

    def promote(self, manifest: SkillAdapterManifest):
        if not manifest.evidence_digest:
            raise ValueError("promotion evidence required")
        promoted = SkillAdapterManifest(**{**asdict(manifest), "status": "promoted"})
        path = self.register(promoted)
        doc = self.active_doc()
        doc[promoted.skill_id] = {
            "manifest_digest": promoted.digest,
            "path": str(path.resolve()),
        }
        tmp = self.active.with_suffix(".tmp")
        tmp.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
        os.replace(tmp, self.active)
        return promoted

    def active_doc(self):
        return json.loads(self.active.read_text()) if self.active.exists() else {}

    def active_manifests(self):
        out = []
        for rec in self.active_doc().values():
            doc = json.loads(Path(rec["path"]).read_text())
            manifest = SkillAdapterManifest(**doc["manifest"])
            if manifest.digest != doc["digest"]:
                raise ValueError("active adapter manifest digest mismatch")
            if tree_digest(manifest.artifact_path) != manifest.artifact_digest:
                raise ValueError("active adapter integrity failure")
            out.append(manifest)
        return out
