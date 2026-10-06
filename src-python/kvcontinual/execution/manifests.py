from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from kvcontinual.execution.digests import sha256_json
from kvcontinual.execution.types import ExecutionIdentity


@dataclass(frozen=True)
class RuntimeManifest:
    execution_identity: ExecutionIdentity
    configuration_digest: str
    created_at: str

    @classmethod
    def create(cls, *, execution_identity: ExecutionIdentity, config: dict) -> "RuntimeManifest":
        return cls(execution_identity, sha256_json(config), datetime.now(timezone.utc).isoformat())

    @property
    def digest(self) -> str:
        payload = {"execution_identity": self.execution_identity.to_dict(), "configuration_digest": self.configuration_digest, "created_at": self.created_at}
        return sha256_json(payload)
