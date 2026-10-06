"""Deterministic sparse-routing record for activation recomputation.

v5.1 records both the approximate candidate set and the final discrete top-k.
This matters once hierarchical retrieval is used by the virtual runtime: replay
must not be allowed to silently change the candidate graph before replaying the
final expert IDs.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import hashlib
import json
import torch


@dataclass
class RoutingTape:
    entries: list[dict] = field(default_factory=list)
    cursor: int = 0
    replay: bool = False

    def reset_replay(self):
        self.cursor = 0
        self.replay = True

    def stop_replay(self):
        self.replay = False
        self.cursor = 0

    def clear(self):
        self.entries.clear()
        self.cursor = 0
        self.replay = False

    def _choose(self, kind: str, site: int, indices: torch.Tensor) -> torch.Tensor:
        shape = tuple(indices.shape)
        if self.replay:
            if self.cursor >= len(self.entries):
                raise RuntimeError("routing tape exhausted during replay")
            ent = self.entries[self.cursor]
            self.cursor += 1
            ent_kind = str(ent.get("kind", "topk"))  # v5.0 compatibility
            if (ent_kind != kind or int(ent["site"]) != int(site)
                    or tuple(ent["shape"]) != shape):
                raise RuntimeError(
                    f"routing replay mismatch at {self.cursor-1}: expected "
                    f"{ent_kind}/site {ent['site']}/shape {tuple(ent['shape'])}, "
                    f"got {kind}/site {site}/shape {shape}")
            return ent["indices"].to(indices.device)
        self.entries.append({"kind": kind, "site": int(site), "shape": shape,
                             "indices": indices.detach().cpu().clone()})
        return indices

    def next_kind(self):
        if not self.replay or self.cursor >= len(self.entries):
            return None
        return str(self.entries[self.cursor].get("kind", "topk"))

    def replay_next(self, kind: str, site: int, device=None) -> torch.Tensor:
        """Consume a recorded decision without recomputing a proposal.

        Hierarchical candidate retrieval and stochastic top-k must be skipped
        during activation replay; otherwise an audit fallback or a fresh RNG
        draw could alter control flow before the tape gets a chance to restore
        the original graph.
        """
        if not self.replay:
            raise RuntimeError("routing tape is not in replay mode")
        if self.cursor >= len(self.entries):
            raise RuntimeError("routing tape exhausted during replay")
        ent = self.entries[self.cursor]
        self.cursor += 1
        ent_kind = str(ent.get("kind", "topk"))
        if ent_kind != str(kind) or int(ent["site"]) != int(site):
            raise RuntimeError(
                f"routing replay mismatch at {self.cursor-1}: expected "
                f"{ent_kind}/site {ent['site']}, got {kind}/site {site}")
        return ent["indices"].to(device=device)

    def choose(self, site: int, indices: torch.Tensor) -> torch.Tensor:
        return self._choose("topk", site, indices)

    def choose_candidates(self, site: int, indices: torch.Tensor) -> torch.Tensor:
        return self._choose("candidates", site, indices)

    def digest(self) -> str:
        """Stable SHA-256 of the discrete routing metadata."""
        h = hashlib.sha256()
        for ent in self.entries:
            meta = {
                "kind": str(ent.get("kind", "topk")),
                "site": int(ent["site"]),
                "shape": list(ent["shape"]),
            }
            h.update(json.dumps(meta, sort_keys=True, separators=(",", ":")).encode())
            t = ent["indices"].detach().cpu().to(torch.int64).contiguous()
            h.update(t.numpy().tobytes())
        return h.hexdigest()
