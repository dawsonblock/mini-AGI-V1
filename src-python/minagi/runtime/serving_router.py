"""v16.4.4 serving router — per-activation request leases
(HARDENING_PLAN WP8 / OPS-003 / v16.4.4 WP-B / SEC-303).

v16.4.3 used one global ``_inflight`` counter and a ``deactivate`` that
returned after a timeout while requests were still running — the
supervisor then unloaded a model that was actively serving. v16.4.4
replaces that with per-activation lease accounting:

  * every request acquires a lease under the same lock that controls
    route replacement; the lease names the activation and route epoch
    it was issued against, so a release can never be misapplied to a
    re-prepared route;
  * routes have explicit gates: ``prepare_route`` registers a
    non-serving entry, ``publish_route`` opens traffic for exactly the
    authorized deployment generation, ``stop_accepting`` withdraws new
    traffic, ``drain_until_idle`` waits for the activation's OWN
    in-flight count, ``cancel_requests`` asks the backend for
    cooperative cancellation, and ``safe_unload`` releases resources
    only when the activation's lease count is zero;
  * a drain timeout is NOT permission to unload: on timeout the retired
    route stays non-accepting and its resources are retained until its
    leases actually reach zero (``safe_unload`` refuses while
    ``inflight > 0``);
  * generation checking: ``publish_route`` refuses a generation that is
    not newer than the last published one — a stale transition cannot
    overwrite a completed deployment.

The required unload ordering is:

    stop_accepting -> drain_until_idle -> cancel_requests (if needed)
                   -> terminate (worker backends, after cancel grace)
                   -> confirm inflight == 0 -> safe_unload

The router never accepts a raw model path or a client-supplied backend
object: routes are registered only by the supervisor with handles that
already passed load + health probes, and ``route`` dispatches to the
backend's own inference method on the live handle.

Compatibility: ``activate``/``deactivate``/``drain``/``route``/``status``
keep their v16.4.3 signatures, implemented over the lease machinery.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass


class RoutingRefused(PermissionError):
    """No authorized live model may serve this request."""


class RouteConflict(RuntimeError):
    """A control-plane operation collided with a live route state."""


@dataclass
class RouteLeaseState:
    """External snapshot of one activation's routing state."""
    activation_id: str
    generation: int
    accepting: bool
    inflight: int
    draining: bool


@dataclass
class _Lease:
    """A held request lease: the activation + route epoch it counts
    against, so release can never decrement a re-prepared route."""
    activation_id: str
    epoch: int
    principal: str
    acquired_at: float
    released: bool = False


@dataclass
class _RouteEntry:
    activation_id: str
    backend: object
    handle: object
    version: int
    generation: int
    epoch: int
    activated_at: int
    accepting: bool = False
    inflight: int = 0
    draining: bool = False
    retired_at: float = 0.0

    def __post_init__(self):
        self.cancelled = threading.Event()


class ServingRouter:
    """Versioned, lock-guarded routing table with per-activation
    request leases and bounded-concurrency admission."""

    def __init__(self, *, drain_timeout: float = 5.0, budget=None,
                 cancel_grace: float = 2.0, terminate_grace: float = 5.0):
        self._lock = threading.RLock()
        self._routes: dict[str, _RouteEntry] = {}
        self._current: str | None = None
        self._version = 0
        self._published_generation = 0
        self._epoch = 0
        self._drained = threading.Condition(self._lock)
        self._drain_timeout = float(drain_timeout)
        self._cancel_grace = float(cancel_grace)
        self._terminate_grace = float(terminate_grace)
        self._budget = budget
        self._principal_inflight: dict[str, int] = {}

    # --- supervisor-driven control plane -----------------------------
    def prepare_route(self, activation_id: str, backend, handle,
                      *, at: int = 0) -> RouteLeaseState:
        """Register a route entry for a verified live handle WITHOUT
        accepting traffic. A route that still accepts requests or holds
        in-flight leases cannot be prepared over — publish the new
        generation or retire the old route first."""
        with self._lock:
            entry = self._routes.get(activation_id)
            if entry is not None:
                if entry.accepting or entry.inflight > 0:
                    raise RouteConflict(
                        f"route {activation_id!r} is still live "
                        f"(accepting={entry.accepting}, inflight="
                        f"{entry.inflight}) — cannot prepare over it")
                self._epoch += 1
                entry.epoch = self._epoch
                entry.backend = backend
                entry.handle = handle
                entry.activated_at = int(at)
                entry.draining = False
                entry.cancelled = threading.Event()
            else:
                self._epoch += 1
                entry = _RouteEntry(
                    activation_id=activation_id, backend=backend,
                    handle=handle, version=0, generation=0,
                    epoch=self._epoch, activated_at=int(at))
                self._routes[activation_id] = entry
            return self._lease_state(entry)

    def publish_route(self, activation_id: str, *, generation=None,
                      at: int = 0) -> int:
        """Open traffic for exactly one prepared route. When
        `generation` is supplied it must be newer than every previously
        published generation — a stale transition loses (CAS). Returns
        the routing-table version."""
        with self._lock:
            entry = self._routes.get(activation_id)
            if entry is None:
                raise RouteConflict(
                    f"publish of unknown route {activation_id!r} — "
                    "prepare_route first")
            if generation is not None and \
                    int(generation) <= self._published_generation:
                raise RouteConflict(
                    f"deployment generation {generation} is not newer "
                    f"than the last published generation "
                    f"{self._published_generation} — stale transition "
                    "refused")
            current = self._routes.get(self._current or "")
            if current is not None and current is not entry:
                current.accepting = False
                current.draining = True
                current.retired_at = time.monotonic()
            self._version += 1
            entry.version = self._version
            entry.accepting = True
            entry.draining = False
            entry.activated_at = int(at)
            self._current = activation_id
            if generation is not None:
                self._published_generation = int(generation)
                entry.generation = int(generation)
            return self._version

    def stop_accepting(self, activation_id: str) -> RouteLeaseState:
        """Withdraw new traffic from one activation. In-flight leases
        are unaffected — they keep their backend + handle alive."""
        with self._lock:
            entry = self._routes.get(activation_id)
            if entry is None:
                return RouteLeaseState(activation_id, 0, False, 0, False)
            entry.accepting = False
            entry.draining = True
            entry.retired_at = time.monotonic()
            if self._current == activation_id:
                self._current = None
            return self._lease_state(entry)

    def drain_until_idle(self, activation_id: str, *,
                         timeout: float | None = None) -> bool:
        """Wait until THIS activation's in-flight leases reach zero.
        Returns False on timeout — never a claim that the backend is
        idle, so a timeout is not permission to unload."""
        deadline = time.monotonic() + (
            self._drain_timeout if timeout is None else float(timeout))
        with self._drained:
            entry = self._routes.get(activation_id)
            if entry is None:
                return True
            while entry.inflight > 0:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._drained.wait(timeout=remaining)
            return True

    def cancel_requests(self, activation_id: str) -> int:
        """Cooperatively cancel in-flight requests: signal the route's
        cancel event (passed to the backend on each request) and call
        the backend's own ``cancel(handle)`` hook when it exposes one.
        Returns the number of in-flight leases signalled."""
        with self._lock:
            entry = self._routes.get(activation_id)
            if entry is None:
                return 0
            count = entry.inflight
            entry.cancelled.set()
            backend, handle = entry.backend, entry.handle
        cancel = getattr(backend, "cancel", None)
        if callable(cancel):
            try:
                cancel(handle)
            except Exception:  # noqa: BLE001 - cooperative best effort
                pass
        return count

    def safe_unload(self, activation_id: str) -> bool:
        """Release the backend resources of a retired route. Refuses
        (returns False) while the route still accepts requests or holds
        in-flight leases — Unload(M) requires InFlight(M) == 0."""
        with self._lock:
            entry = self._routes.get(activation_id)
            if entry is None:
                return True
            if entry.accepting or entry.inflight > 0:
                return False
            backend, handle = entry.backend, entry.handle
            del self._routes[activation_id]
        try:
            backend.unload(handle)
        except Exception:  # noqa: BLE001 - unload is best effort
            pass
        return True

    def retire(self, activation_id: str, *,
               timeout: float | None = None) -> dict:
        """The full withdrawal sequence for one activation:
        stop_accepting -> drain -> cancel on timeout -> terminate a
        terminable backend whose requests ignore cancellation ->
        re-drain -> safe_unload when actually idle.

        A backend exposing ``terminate(handle)`` (a worker-process
        backend) is killed once drain timeout + cancel grace have both
        elapsed — its in-flight requests fail fast and the OS reclaims
        the model's resources (RUN-401). A backend without ``terminate``
        keeps the v16.4.4 semantics: resources are RETAINED while
        leases remain and retire() may be retried later."""
        self.stop_accepting(activation_id)
        drained = self.drain_until_idle(activation_id, timeout=timeout)
        cancelled = 0
        terminated = False
        if not drained:
            cancelled = self.cancel_requests(activation_id)
            drained = self.drain_until_idle(
                activation_id, timeout=self._cancel_grace)
        if not drained:
            terminated = self._terminate_route(activation_id)
            if terminated:
                drained = self.drain_until_idle(
                    activation_id, timeout=self._terminate_grace)
        unloaded = self.safe_unload(activation_id) if drained else False
        return {"activation_id": activation_id, "drained": drained,
                "cancelled": cancelled, "terminated": terminated,
                "unloaded": unloaded,
                "inflight": self.inflight(activation_id)}

    def _terminate_route(self, activation_id: str) -> bool:
        """Preemptive resource release for backends that support it —
        the remedy for requests that ignored cooperative cancellation.
        Invoked only after the drain timeout and the cancel grace have
        both elapsed; never as a first resort."""
        with self._lock:
            entry = self._routes.get(activation_id)
            if entry is None or entry.inflight <= 0:
                # Nothing holds the backend anymore — the drain landed
                # between checks; do not kill a now-idle worker.
                return False
            backend, handle = entry.backend, entry.handle
        terminate = getattr(backend, "terminate", None)
        if not callable(terminate):
            return False
        try:
            terminate(handle)
        except Exception:  # noqa: BLE001 - the kill path is best effort
            return False
        return True

    # --- compat control plane ----------------------------------------
    def activate(self, activation_id: str, backend, handle,
                 *, at: int = 0) -> int:
        """v16.4.3-compatible prepare+publish in one call. Re-
        activating an existing (retired) route re-opens it — the
        supervisor only ever re-publishes the same verified handle."""
        with self._lock:
            if self._routes.get(activation_id) is None:
                self.prepare_route(activation_id, backend, handle, at=at)
        return self.publish_route(activation_id, at=at)

    def deactivate(self, activation_id: str | None = None) -> None:
        """Stop routing to `activation_id` (or whatever is current) and
        wait up to `drain_timeout` for ITS leases. Never unloads."""
        target = activation_id
        with self._lock:
            if target is None:
                target = self._current
        if target is None:
            return
        self.stop_accepting(target)
        self.drain_until_idle(target, timeout=self._drain_timeout)

    def drain(self, activation_id: str) -> None:
        self.deactivate(activation_id)

    # --- data plane ---------------------------------------------------
    def acquire_lease(self, activation_id: str | None = None,
                      *, principal: str = "") -> _Lease:
        """Acquire a request lease under the same lock that controls
        route replacement — a publish/stop can never tear a held lease.
        Enforces the configured concurrency and per-principal budgets."""
        with self._lock:
            aid = activation_id or self._current
            entry = self._routes.get(aid or "")
            if entry is None or not entry.accepting:
                raise RoutingRefused(
                    "no live committed activation is serving — the "
                    "service is unavailable")
            if self._budget is not None:
                total = sum(e.inflight for e in self._routes.values())
                if total >= int(getattr(
                        self._budget, "max_concurrent_requests",
                        1 << 30)):
                    raise RoutingRefused(
                        "concurrency budget exhausted — the request "
                        "would exceed the authorized simultaneous-"
                        "request limit")
                if principal:
                    per = self._principal_inflight.get(principal, 0)
                    if per >= int(getattr(
                            self._budget, "max_queued_per_principal",
                            1 << 30)):
                        raise RoutingRefused(
                            f"per-principal request budget exhausted "
                            f"for {principal!r}")
            entry.inflight += 1
            if principal:
                self._principal_inflight[principal] = \
                    self._principal_inflight.get(principal, 0) + 1
            return _Lease(activation_id=entry.activation_id,
                          epoch=entry.epoch, principal=str(principal),
                          acquired_at=time.monotonic())

    def release_lease(self, lease: _Lease) -> None:
        with self._drained:
            if lease.released:
                return
            lease.released = True
            entry = self._routes.get(lease.activation_id)
            if entry is not None and entry.epoch == lease.epoch and \
                    entry.inflight > 0:
                entry.inflight -= 1
                if entry.inflight == 0:
                    self._drained.notify_all()
            if lease.principal:
                left = self._principal_inflight.get(lease.principal, 0) - 1
                if left <= 0:
                    self._principal_inflight.pop(lease.principal, None)
                else:
                    self._principal_inflight[lease.principal] = left

    def route(self, request, *, principal: str = "") -> dict:
        """Dispatch one inference request under a held lease. The lease
        is released in `finally` — including when inference raises or is
        cancelled — so in-flight accounting cannot leak."""
        lease = self.acquire_lease(principal=principal)
        try:
            with self._lock:
                entry = self._routes.get(lease.activation_id)
                if entry is None or entry.epoch != lease.epoch:
                    raise RoutingRefused(
                        "route retired between lease acquisition and "
                        "dispatch")
                backend, handle = entry.backend, entry.handle
                version = entry.version
                cancelled = entry.cancelled
            if callable(getattr(backend, "infer", None)):
                infer = getattr(backend, "infer")
            elif callable(getattr(backend, "generate", None)):
                infer = getattr(backend, "generate")
            else:
                raise RoutingRefused(
                    f"backend {getattr(backend, 'backend_id', None)!r} "
                    "exposes no inference method on the routed handle")
            if isinstance(request, dict):
                request = dict(request)
                request.setdefault("_cancel_event", cancelled)
            started = time.monotonic()
            try:
                result = infer(handle, request)
            except Exception as exc:  # noqa: BLE001
                if cancelled.is_set():
                    raise RoutingRefused(
                        "request cancelled — the route was withdrawn "
                        "while inference was in flight") from exc
                raise
            out = {"activation_id": lease.activation_id,
                   "routing_version": version, "result": result}
            metrics = result.get("metrics") \
                if isinstance(result, dict) else None
            out["metrics"] = dict(metrics or {})
            out["metrics"]["request_seconds"] = round(
                time.monotonic() - started, 6)
            return out
        finally:
            self.release_lease(lease)

    # --- introspection -------------------------------------------------
    def inflight(self, activation_id: str | None = None) -> int:
        with self._lock:
            if activation_id is not None:
                entry = self._routes.get(activation_id)
                return 0 if entry is None else entry.inflight
            return sum(e.inflight for e in self._routes.values())

    def lease_states(self) -> list[RouteLeaseState]:
        with self._lock:
            return [self._lease_state(e) for e in self._routes.values()]

    def route_entry(self, activation_id: str) -> RouteLeaseState | None:
        """The lease state of one activation's route, or None if no
        route entry exists."""
        with self._lock:
            entry = self._routes.get(activation_id)
            return None if entry is None else self._lease_state(entry)

    @staticmethod
    def _lease_state(entry: _RouteEntry) -> RouteLeaseState:
        return RouteLeaseState(activation_id=entry.activation_id,
                               generation=entry.generation,
                               accepting=entry.accepting,
                               inflight=entry.inflight,
                               draining=entry.draining)

    def status(self) -> dict:
        with self._lock:
            entry = self._routes.get(self._current or "")
            if entry is None:
                return {"serving": False, "version": self._version,
                        "routes": [vars(self._lease_state(e))
                                   for e in self._routes.values()]}
            return {"serving": True, "version": entry.version,
                    "activation_id": entry.activation_id,
                    "backend_id": getattr(entry.backend, "backend_id", ""),
                    "activated_at": entry.activated_at,
                    "generation": entry.generation,
                    "inflight": entry.inflight,
                    "routes": [vars(self._lease_state(e))
                               for e in self._routes.values()]}
