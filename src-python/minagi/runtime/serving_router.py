"""v16.4.3 serving router (HARDENING_PLAN WP8 / OPS-003).

A durable pointer plus a resident handle does not by itself make
protected inference routing. `ServingRouter` is the runtime service's
routing table: it exposes only the activation that is active, healthy,
and committed — a lock/versioned table serializes activation, rollback,
quarantine, and inference requests so a concurrent transition can never
be observed mid-flight.

The router never accepts a raw model path or a client-supplied backend
object: `activate` is called only by the supervisor with a handle that
already passed load + health probes, and `route` dispatches to the
backend's own inference method on the live handle.

Draining: `deactivate` removes the routing entry immediately (new
requests refuse), waits for in-flight leases to drain up to
`drain_timeout`, then returns — the supervisor unloads after drain.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass


class RoutingRefused(PermissionError):
    """No authorized live model may serve this request."""


@dataclass
class _Route:
    activation_id: str
    backend: object
    handle: object
    version: int
    activated_at: int


class ServingRouter:
    """Versioned, lock-guarded routing table for the active model."""

    def __init__(self, *, drain_timeout: float = 5.0):
        self._lock = threading.RLock()
        self._route: _Route | None = None
        self._version = 0
        self._inflight = 0
        self._drained = threading.Condition(self._lock)
        self._drain_timeout = float(drain_timeout)

    # --- supervisor-driven control plane -----------------------------
    def activate(self, activation_id: str, backend, handle,
                 *, at: int = 0) -> int:
        """Point traffic at a verified live handle. Returns the new
        routing-table version."""
        with self._lock:
            self._version += 1
            self._route = _Route(activation_id=activation_id,
                                 backend=backend, handle=handle,
                                 version=self._version,
                                 activated_at=int(at))
            return self._version

    def deactivate(self, activation_id: str | None = None) -> None:
        """Stop routing to `activation_id` (or whatever is active).
        Waits briefly for in-flight requests to finish."""
        deadline = time.monotonic() + self._drain_timeout
        with self._drained:
            if activation_id is None or (
                    self._route is not None and
                    self._route.activation_id == activation_id):
                self._route = None
            while self._inflight > 0:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self._drained.wait(timeout=remaining)

    def drain(self, activation_id: str) -> None:
        """Alias for deactivate — drain in-flight requests and remove
        the routing entry."""
        self.deactivate(activation_id)

    # --- data plane ---------------------------------------------------
    def route(self, request):
        """Dispatch one inference request to the active handle. Refuses
        when nothing healthy+committed is serving or the backend lacks
        an inference method."""
        with self._lock:
            route = self._route
            if route is None:
                raise RoutingRefused(
                    "no live committed activation is serving — the "
                    "service is unavailable")
            self._inflight += 1
        try:
            infer = getattr(route.backend, "infer", None) or \
                getattr(route.backend, "generate", None)
            if infer is None:
                raise RoutingRefused(
                    f"backend {getattr(route.backend, 'backend_id', None)!r} "
                    "exposes no inference method on the routed handle")
            return {"activation_id": route.activation_id,
                    "routing_version": route.version,
                    "result": infer(route.handle, request)}
        finally:
            with self._drained:
                self._inflight -= 1
                if self._inflight == 0:
                    self._drained.notify_all()

    # --- introspection -------------------------------------------------
    def status(self) -> dict:
        with self._lock:
            if self._route is None:
                return {"serving": False, "version": self._version}
            r = self._route
            return {"serving": True, "version": r.version,
                    "activation_id": r.activation_id,
                    "backend_id": getattr(r.backend, "backend_id", ""),
                    "activated_at": r.activated_at,
                    "inflight": self._inflight}
