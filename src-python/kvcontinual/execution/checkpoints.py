from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import uuid

from kvcontinual.execution.types import ExecutionIdentity


@dataclass
class ExactPrefixCheckpoint:
    conversation_id: str
    token_position: int
    identity: ExecutionIdentity
    prefix_token_digest: str
    recurrent_state_ref: str
    conv_state_ref: str
    attention_prefix_ref: str
    checkpoint_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class PrefixCheckpointStore:
    def __init__(self):
        self._items: list[ExactPrefixCheckpoint] = []

    def put(self, checkpoint: ExactPrefixCheckpoint) -> None:
        self._items.append(checkpoint)

    def deepest(self, conversation_id: str, token_position: int, identity: ExecutionIdentity) -> ExactPrefixCheckpoint | None:
        xs = [x for x in self._items if x.conversation_id == conversation_id and x.identity == identity and x.token_position <= token_position]
        return max(xs, key=lambda x: x.token_position) if xs else None
