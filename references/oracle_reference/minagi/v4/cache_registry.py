from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any
import json, os, time
from .generation import EffectiveModelGeneration, CacheCompatibility
from .util import canonical_json, sha256_bytes, tree_digest

@dataclass(frozen=True)
class CompiledCacheArtifact:
    artifact_id: str
    canonical_record_ids: tuple[str, ...]
    generation_digest: str
    artifact_path: str
    artifact_digest: str
    tier: str
    cache_kind: str
    created_at: float
    metadata: dict[str, Any]

    @classmethod
    def create(cls, *, canonical_record_ids, generation: EffectiveModelGeneration,
               artifact_path: str, tier: str = "warm", cache_kind: str = "rc10",
               metadata: dict[str, Any] | None = None):
        if tier not in {"warm", "hot", "critical", "cold"}:
            raise ValueError("invalid cache tier")
        path = str(Path(artifact_path).resolve())
        digest = tree_digest(path)
        seed = canonical_json({
            "records": list(canonical_record_ids), "generation": generation.digest,
            "artifact": digest, "kind": cache_kind,
        })
        return cls(
            "cache-" + sha256_bytes(seed.encode())[:24],
            tuple(map(str, canonical_record_ids)), generation.digest,
            path, digest, tier, cache_kind, time.time(), dict(metadata or {}),
        )

class NeuralCacheRegistry:
    """Registry for disposable model-versioned cache artifacts.

    Canonical memory is never deleted when this registry evicts/invalidate caches.
    """
    def __init__(self, root: str | Path):
        self.root = Path(root); self.root.mkdir(parents=True, exist_ok=True)
        self.index_path = self.root / "index.json"

    def _load(self):
        return json.loads(self.index_path.read_text()) if self.index_path.exists() else {}

    def _save(self, doc):
        tmp = self.index_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
        os.replace(tmp, self.index_path)

    def register(self, artifact: CompiledCacheArtifact):
        if tree_digest(artifact.artifact_path) != artifact.artifact_digest:
            raise ValueError("compiled cache artifact digest mismatch")
        doc = self._load(); doc[artifact.artifact_id] = asdict(artifact); self._save(doc)
        return artifact.artifact_id

    def compatible(self, generation: EffectiveModelGeneration):
        out = []
        for rec in self._load().values():
            if rec["generation_digest"] == generation.digest:
                art = CompiledCacheArtifact(**rec)
                if tree_digest(art.artifact_path) != art.artifact_digest:
                    raise ValueError(f"compiled cache artifact corrupted: {art.artifact_id}")
                out.append(art)
        return out

    def invalidate_generation(self, generation_digest: str, *, delete_artifacts: bool = False) -> int:
        doc = self._load(); removed = []
        for key, rec in list(doc.items()):
            if rec["generation_digest"] == generation_digest:
                removed.append(rec); del doc[key]
        self._save(doc)
        if delete_artifacts:
            import shutil
            for rec in removed:
                p = Path(rec["artifact_path"])
                if p.is_dir(): shutil.rmtree(p, ignore_errors=True)
                elif p.exists(): p.unlink()
        return len(removed)
