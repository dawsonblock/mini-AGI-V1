from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from kvcontinual.continual.digests import sha256_json


@dataclass(frozen=True)
class RuntimeManifest:
    base_model_digest: str
    adapter_set_digest: str
    tokenizer_digest: str
    recurrence_impl: str
    position_scheme: str
    configuration_digest: str
    created_at: str

    @classmethod
    def create(cls, *, base_model_digest: str, adapter_set_digest: str, tokenizer_digest: str, recurrence_impl: str, position_scheme: str, config: dict) -> "RuntimeManifest":
        return cls(
            base_model_digest=base_model_digest,
            adapter_set_digest=adapter_set_digest,
            tokenizer_digest=tokenizer_digest,
            recurrence_impl=recurrence_impl,
            position_scheme=position_scheme,
            configuration_digest=sha256_json(config),
            created_at=datetime.now(timezone.utc).isoformat(),
        )

    @property
    def digest(self) -> str:
        return sha256_json(asdict(self))
