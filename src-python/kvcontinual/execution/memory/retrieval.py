from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class VectorItem:
    id: str
    vector: np.ndarray
    utility: float = 0.0
    confidence: float = 0.5
    importance: float = 0.0
    recency: float = 0.0
    recurrence: float = 0.0


class NumpyVectorIndex:
    """Deterministic local fallback index. Replace with FAISS for scale."""
    def __init__(self):
        self.items: list[VectorItem] = []

    def add(self, item: VectorItem) -> None:
        v = np.asarray(item.vector, dtype=np.float32)
        n = np.linalg.norm(v)
        if n == 0:
            raise ValueError("zero vector")
        item.vector = v / n
        self.items.append(item)

    def search(self, query: np.ndarray, k: int = 12) -> list[tuple[str, float]]:
        q = np.asarray(query, dtype=np.float32)
        n = np.linalg.norm(q)
        if n == 0:
            return []
        q = q / n
        scored = []
        for x in self.items:
            semantic = float(np.dot(q, x.vector))
            score = (
                0.40 * semantic
                + 0.20 * x.utility
                + 0.15 * x.confidence
                + 0.10 * x.importance
                + 0.10 * x.recency
                + 0.05 * x.recurrence
            )
            scored.append((x.id, score))
        scored.sort(key=lambda z: z[1], reverse=True)
        return scored[:k]
