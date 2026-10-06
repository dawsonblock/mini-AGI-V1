from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
import uuid


@dataclass
class Episode:
    prompt: str
    response: str
    retrieved_memories: list[str] = field(default_factory=list)
    source_blocks: list[str] = field(default_factory=list)
    tools_used: list[str] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    reward: float | None = None
    user_feedback: str | None = None
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class EpisodeLog:
    def __init__(self, path: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, e: Episode) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(e.__dict__, sort_keys=True) + "\n")
