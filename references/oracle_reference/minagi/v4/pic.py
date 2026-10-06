from __future__ import annotations

"""Position-independent cache metadata helpers for full-attention RC10 planes.

The actual attention kernel remains model/runtime specific; this module owns the
position relocation contract so cached and fresh planes cannot silently disagree
about logical history.
"""

from dataclasses import dataclass
from typing import Sequence
import torch


@dataclass(frozen=True)
class FAPICSegment:
    key: torch.Tensor
    value: torch.Tensor
    source_positions: tuple[int, ...]
    token_digest: str = ""

    def validate(self) -> None:
        if self.key.shape[:-1] != self.value.shape[:-1]:
            raise ValueError("K/V leading dimensions mismatch")
        if self.key.shape[-2] != len(self.source_positions):
            raise ValueError("position count does not match cached sequence")


@dataclass(frozen=True)
class RelocatedFAPIC:
    key: torch.Tensor
    value: torch.Tensor
    logical_positions: tuple[int, ...]
    source_positions: tuple[int, ...]


def relocate(segment: FAPICSegment, start_position: int, *, skip: int = 0) -> RelocatedFAPIC:
    segment.validate()
    skip = int(skip)
    if skip < 0 or skip > len(segment.source_positions):
        raise ValueError("invalid skip")
    k = segment.key[..., skip:, :]
    v = segment.value[..., skip:, :]
    n = k.shape[-2]
    logical = tuple(range(int(start_position), int(start_position) + n))
    return RelocatedFAPIC(k, v, logical, segment.source_positions[skip:])


def append_segments(parts: Sequence[RelocatedFAPIC]) -> RelocatedFAPIC:
    if not parts:
        raise ValueError("non-empty PIC parts required")
    expected = parts[0].logical_positions[0] if parts[0].logical_positions else 0
    for part in parts:
        if part.logical_positions and part.logical_positions[0] != expected:
            raise ValueError("logical positions must be contiguous")
        expected += len(part.logical_positions)
    return RelocatedFAPIC(
        torch.cat([p.key for p in parts], dim=-2),
        torch.cat([p.value for p in parts], dim=-2),
        tuple(x for p in parts for x in p.logical_positions),
        tuple(x for p in parts for x in p.source_positions),
    )
