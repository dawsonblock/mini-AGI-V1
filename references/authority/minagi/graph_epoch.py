"""Graph-wide expert-version barrier.

A graph epoch pins each logical expert to the SHA-256 first observed in the
forward.  Backward must reload the same digest.  External expert updates are
forbidden until ``finish_backward``.  This is the v5 replacement for treating a
mutable VRAM slot as parameter identity.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import itertools
from .routing_tape import RoutingTape

_ids = itertools.count(1)


@dataclass
class GraphEpoch:
    store: object
    training: bool = True
    id: int = field(default_factory=lambda: next(_ids))
    versions: dict[int, str] = field(default_factory=dict)
    routing: RoutingTape = field(default_factory=RoutingTape)
    state: str = "open"

    def pin(self, uid: int) -> str:
        if self.state not in ("open", "backward", "backward_done"):
            raise RuntimeError(f"graph epoch {self.id} is {self.state}")
        uid = int(uid)
        now = self.store.version(uid).sha256
        old = self.versions.setdefault(uid, now)
        if old != now:
            raise RuntimeError(
                f"expert {uid} mutated inside graph epoch {self.id}: {old} -> {now}")
        return old

    def begin_backward(self):
        if self.state != "open":
            raise RuntimeError(f"cannot begin backward from {self.state}")
        self.state = "backward"

    def finish_backward(self):
        if self.state not in ("open", "backward"):
            raise RuntimeError(f"cannot finish backward from {self.state}")
        # A direct loss.backward() may not have called begin_backward; allowing
        # open->backward_done keeps the API ergonomic without weakening the barrier.
        self.state = "backward_done"

    def assert_step_allowed(self):
        if self.training and self.state != "backward_done":
            raise RuntimeError("expert optimizer step requires completed backward")

    def close(self):
        if self.state == "aborted":
            return
        if self.training and self.state != "backward_done":
            raise RuntimeError("cannot close training graph before backward")
        self.state = "closed"

    def abort(self):
        self.state = "aborted"
