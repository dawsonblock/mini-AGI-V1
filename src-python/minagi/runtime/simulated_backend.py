"""Deterministic simulated serving backend (tests + development).

Exercises the full supervised-backend contract — ``load`` /
``health_probe`` / ``infer`` / ``cancel`` / ``unload`` — without model
weights, a tokenizer, or an accelerator. Every behaviour is configured
through picklable constructor kwargs, so the backend can run inside a
``WorkerBackend`` child process (`python -m minagi.runtime.worker_backend`)
as well as in-process:

  * ``latency_seconds`` — per-request generation latency, served
    cooperatively: the request's ``_cancel_event`` short-circuits it;
  * ``respect_cancel=False`` — finishes the latency while ignoring the
    cancel event (cancellation-capable but uncooperative);
  * ``wedge=True`` — an uninterruptible generation: the simulation of
    a native-level wedge that cooperative cancellation cannot reach.
    Only ``terminate`` (or process death) ends it;
  * ``die_on_infer=True`` — ``os._exit`` mid-request: the simulation
    of a crash or OOM kill;
  * ``load_seconds`` / ``probe_seconds`` — startup latency injection.

It still requires a real ``MeasuredSnapshot`` at load — the isolation
boundary carries measured evidence, never a raw path.
"""
from __future__ import annotations

import os
import time

from minagi.v161.immutable_snapshot import MeasuredSnapshot


class SimulatedServingBackend:
    """Reference backend for qualification tests and local dev."""

    backend_id = "simulated"

    def __init__(self, *, latency_seconds: float = 0.0,
                 wedge: bool = False, die_on_infer: bool = False,
                 respect_cancel: bool = True, load_seconds: float = 0.0,
                 probe_seconds: float = 0.0):
        self.latency_seconds = float(latency_seconds)
        self.wedge = bool(wedge)
        self.die_on_infer = bool(die_on_infer)
        self.respect_cancel = bool(respect_cancel)
        self.load_seconds = float(load_seconds)
        self.probe_seconds = float(probe_seconds)

    def load(self, snapshot):
        if not isinstance(snapshot, MeasuredSnapshot):
            raise PermissionError(
                "SimulatedServingBackend.load() requires a "
                "MeasuredSnapshot produced by the trusted staging "
                "path — a raw path is not a loadable artifact")
        if self.load_seconds:
            time.sleep(self.load_seconds)
        return {"snapshot_root": str(snapshot.root),
                "manifest_digest": snapshot.manifest_digest,
                "artifact_digests": dict(snapshot.artifact_digests)}

    def health_probe(self, handle) -> None:
        if self.probe_seconds:
            time.sleep(self.probe_seconds)

    def infer(self, handle, request) -> dict:
        if self.die_on_infer:
            # Unannounced process death — the crash/OOM-kill case. The
            # parent learns of it through channel EOF, not a reply.
            os._exit(1)
        req = dict(request or {})
        ev = req.get("_cancel_event")
        if self.wedge:
            while True:  # uninterruptible — only terminate() ends it
                time.sleep(60.0)
        cancelled = False
        deadline = time.monotonic() + self.latency_seconds
        while time.monotonic() < deadline:
            if self.respect_cancel and ev is not None and ev.is_set():
                cancelled = True
                break
            time.sleep(min(0.01, max(0.0, deadline - time.monotonic())))
        return {"completion": "echo:" + str(req.get("prompt", "")),
                "manifest_digest": handle.get("manifest_digest", ""),
                "metrics": {"request_id": str(req.get("request_id", "")),
                            "cancelled": bool(cancelled),
                            "worker_pid": os.getpid()}}

    def cancel(self, handle) -> None:
        # The route/worker cancel event the request carries is
        # authoritative; this hook exists for parity with real backends.
        pass

    def unload(self, handle) -> None:
        if hasattr(handle, "update"):
            handle["unloaded"] = True


__all__ = ["SimulatedServingBackend"]
