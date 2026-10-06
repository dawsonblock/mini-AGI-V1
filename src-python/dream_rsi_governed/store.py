from __future__ import annotations

import os
from pathlib import Path

from .canonical import parse_sha256, sha256_bytes


class ArtifactStore:
    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def put(self, data: bytes) -> str:
        digest = sha256_bytes(data)
        hexpart = parse_sha256(digest)
        p = self.root / hexpart[:2] / hexpart[2:]
        p.parent.mkdir(parents=True, exist_ok=True)
        if not p.exists():
            tmp = p.with_suffix(".tmp")
            tmp.write_bytes(data)
            os.replace(tmp, p)
        return digest

    def get(self, digest: str) -> bytes:
        hexpart = parse_sha256(digest)
        p = (self.root / hexpart[:2] / hexpart[2:]).resolve()
        if self.root not in p.parents:
            raise ValueError("artifact path escaped store root")
        data = p.read_bytes()
        if sha256_bytes(data) != digest:
            raise ValueError("artifact digest mismatch")
        return data
