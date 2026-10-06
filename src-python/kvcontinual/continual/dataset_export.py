from __future__ import annotations

import json
from pathlib import Path

from kvcontinual.continual.digests import sha256_file
from kvcontinual.continual.experience_store import ExperienceStore


def export_mlx_dataset(store: ExperienceStore, out_dir: str, limit: int = 1000, mark_exported: bool = True) -> tuple[Path, str, list[str]]:
    rows = store.list_training_ready(limit=limit)
    if not rows:
        raise ValueError("no verified training-ready episodes")
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    train = root / "train.jsonl"
    ids: list[str] = []
    with train.open("w", encoding="utf-8") as f:
        for row in rows:
            record = {
                "messages": [
                    {"role": "user", "content": row.prompt},
                    {"role": "assistant", "content": row.response},
                ],
                "metadata": {"episode_id": row.id, "importance": row.importance},
            }
            f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            ids.append(row.id)
    digest = sha256_file(train)
    if mark_exported:
        store.mark_exported(ids)
    return train, digest, ids
