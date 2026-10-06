from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Mapping
import json, os, tempfile, time
from .util import canonical_json, sha256_bytes, tree_digest

@dataclass(frozen=True)
class BaselineManifest:
    base_model: str
    base_model_digest: str
    tokenizer_digest: str
    benchmark_protocol_digest: str
    metrics: Mapping[str, float]
    environment: Mapping[str, Any]
    created_at: float
    schema: str = "mini-agi-baseline-v1"

    @classmethod
    def create(cls, *, base_model: str, base_model_path: str | None = None,
               tokenizer_digest: str, benchmark_protocol: Mapping[str, Any],
               metrics: Mapping[str, float], environment: Mapping[str, Any] | None = None):
        bdig = tree_digest(base_model_path) if base_model_path else sha256_bytes(base_model.encode())
        pdig = sha256_bytes(canonical_json(dict(benchmark_protocol)).encode())
        return cls(base_model, bdig, tokenizer_digest, pdig,
                   dict(metrics), dict(environment or {}), time.time())

    @property
    def digest(self) -> str:
        return sha256_bytes(canonical_json(asdict(self)).encode())

    def write_immutable(self, path: str | Path) -> Path:
        p=Path(path); p.parent.mkdir(parents=True, exist_ok=True)
        doc={"manifest":asdict(self), "digest":self.digest}
        if p.exists():
            old=json.loads(p.read_text())
            if old != doc:
                raise FileExistsError("BaselineManifest already exists with different content")
            return p
        fd,tmp=tempfile.mkstemp(prefix='.baseline-', dir=str(p.parent))
        try:
            with os.fdopen(fd,'w',encoding='utf-8') as f:
                json.dump(doc,f,indent=2,sort_keys=True); f.write('\n'); f.flush(); os.fsync(f.fileno())
            os.replace(tmp,p)
        finally:
            if os.path.exists(tmp): os.unlink(tmp)
        return p

    @classmethod
    def load_verified(cls, path: str | Path):
        doc=json.loads(Path(path).read_text())
        m=cls(**doc['manifest'])
        if m.digest != doc.get('digest'):
            raise ValueError('BaselineManifest digest mismatch')
        return m
