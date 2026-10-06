from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, TypeVar

T = TypeVar("T")


def approximate_tokens(text: str) -> int:
    """Deterministic fallback estimate when the backend tokenizer is unavailable."""
    if not text:
        return 0
    # Conservative for English/code mixed prompts; qualification should use the
    # real tokenizer when exact context accounting matters.
    return max(1, (len(text) + 3) // 4)


@dataclass(frozen=True)
class BudgetSelection:
    selected: tuple[T, ...]
    estimated_tokens: int
    dropped_count: int


def select_by_token_budget(
    items: Iterable[T],
    *,
    render: Callable[[T], str],
    budget_tokens: int,
    max_items: int,
) -> BudgetSelection:
    if budget_tokens < 0 or max_items < 0:
        raise ValueError("budgets must be non-negative")
    chosen: list[T] = []
    used = 0
    total = 0
    for item in items:
        total += 1
        if len(chosen) >= max_items:
            continue
        cost = approximate_tokens(render(item))
        if used + cost > budget_tokens:
            continue
        chosen.append(item)
        used += cost
    return BudgetSelection(tuple(chosen), used, max(0, total - len(chosen)))
