"""v16.4.5 worker-process backend isolation (RUN-401).

v16.4.4's per-activation leases guarantee the router never FREES a
model while a request references it — but with an in-process backend
the hard case stayed open: a model wedged past its cooperative
cancellation point (a stuck accelerator kernel, a deadlocked native
call, a generate loop ignoring ``stopping_criteria``) could only be
stopped by killing the supervisor itself. Cancellation was
cooperative only.

This module runs each loaded model in its own interpreter process:

  * ``load()`` spawns a fresh worker which imports the authorized
    backend class and loads the staged snapshot inside itself — model
    memory belongs to the worker, never the supervisor;
  * ``infer`` / ``health_probe`` / ``unload`` are request-reply frames
    over the child's private pipes — concurrent in-flight requests are
    demultiplexed by request id;
  * ``cancel()`` sets a worker-local cancellation event injected into
    every in-flight request — the same cooperative contract as the
    in-process path, now across the wire (and stronger: a client can
    no longer smuggle its own ``_cancel_event`` into the backend);
  * ``terminate()`` is the preemptive remedy: SIGKILL the worker. The
    OS reclaims the model's memory regardless of what it was doing,
    pending requests fail fast with :class:`WorkerDied`, their leases
    release, and the drain completes — the supervisor survives to
    reconcile, restore a predecessor, or keep serving other models;
  * unexpected worker death (crash, OOM kill) is detected by pipe EOF
    and fails every pending request — a request can never hang waiting
    on a dead process;
  * parent death is symmetric: the child's command channel hits EOF
    the moment the parent's end closes (orderly exit or SIGKILL
    alike), and the worker exits rather than keep a model resident
    that nobody supervises.

Trust boundary: the backend *spec* is operator configuration (the
service's factory registry), never client input — the same trust
level as the registry itself. The snapshot is pickled as an object:
a MeasuredSnapshot was already produced by the trusted staging path,
so serializing it carries proof-of-measurement, not the ability to
construct one.

The protocol channel is fd 1 duplicated before any backend import —
the worker's own fd 1 is then redirected to stderr, so library
``print`` calls can never corrupt the frame stream.
"""
from __future__ import annotations

import atexit
import importlib
import os
import pickle
import subprocess
import sys
import threading
import traceback
from dataclasses import dataclass, field


class WorkerBackendError(RuntimeError):
    """A worker-process backend operation failed."""


class WorkerDied(WorkerBackendError):
    """The backend worker exited — crashed, OOM-killed, or terminated.
    Every request it held fails fast; the supervisor is intact and may
    reconcile or roll back."""


class WorkerUnresponsive(WorkerBackendError):
    """The worker produced no response inside the watchdog window.
    The handle is marked stalled: further operations fail fast while
    the control plane retires or terminates the worker."""


class RemoteBackendError(WorkerBackendError):
    """The in-worker backend raised and its exception type could not
    be mapped to a local class; the remote message and traceback are
    preserved on ``traceback_text``."""


_SAFE_EXC_MODULES = ("builtins", "minagi.", "egai.")


def _remote_exception_class(module: str, name: str):
    """Re-raise worker exceptions as their real type when the class is
    safely resolvable (builtins or project modules) — so a backend-side
    ``BudgetExceeded`` still arrives as ``BudgetExceeded`` across the
    wire instead of being flattened into a generic error."""
    try:
        if not isinstance(module, str) or not isinstance(name, str):
            return None
        if not (module == "builtins" or
                module.startswith(_SAFE_EXC_MODULES)):
            return None
        cls = importlib.import_module(module)
        for part in name.split("."):
            cls = getattr(cls, part)
        if isinstance(cls, type) and issubclass(cls, BaseException):
            return cls
    except Exception:  # noqa: BLE001 - mapping is best effort
        pass
    return None


@dataclass(frozen=True)
class BackendSpec:
    """Which backend the worker instantiates — operator configuration
    equivalent in trust to the service's factory registry. ``kwargs``
    must be picklable; they cross the process boundary verbatim."""
    module: str
    qualname: str
    kwargs: dict = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self, "module", str(self.module))
        object.__setattr__(self, "qualname", str(self.qualname))
        object.__setattr__(self, "kwargs", dict(self.kwargs))
        if not self.module or not self.qualname:
            raise WorkerBackendError("backend spec requires module+qualname")

    def resolve(self):
        """Import the backend class — in the parent this verifies the
        spec is real and reads its ``backend_id`` without instantiating
        it; in the worker it is the class actually constructed."""
        try:
            obj = importlib.import_module(self.module)
            for part in self.qualname.split("."):
                obj = getattr(obj, part)
        except Exception as exc:  # noqa: BLE001 - spec errors are refused
            raise WorkerBackendError(
                f"backend spec {self.module}:{self.qualname} does not "
                f"resolve: {exc}") from exc
        if not callable(obj):
            raise WorkerBackendError(
                f"backend spec {self.module}:{self.qualname} is not "
                "callable")
        return obj

    def backend_id(self) -> str:
        bid = getattr(self.resolve(), "backend_id", None)
        if not isinstance(bid, str) or not bid:
            raise WorkerBackendError(
                f"backend spec {self.module}:{self.qualname} exposes "
                "no backend_id")
        return bid


class _Pending:
    """One outstanding RPC: resolved by the reader thread (reply) or
    by death marking (error)."""
    __slots__ = ("event", "reply", "error")

    def __init__(self):
        self.event = threading.Event()
        self.reply: dict | None = None
        self.error: BaseException | None = None


class _WorkerHandle:
    """Parent-side state for one worker process — the opaque `handle`
    the router and supervisor carry."""
    __slots__ = ("owner", "proc", "reader", "pending", "send_lock",
                 "seq", "dead", "stalled", "unloaded")

    def __init__(self, *, owner, proc):
        self.owner = owner
        self.proc = proc
        self.reader: threading.Thread | None = None
        self.pending: dict[str, _Pending] = {}
        self.send_lock = threading.RLock()
        self.seq = 0
        self.dead: WorkerDied | None = None
        self.stalled = False
        self.unloaded = False

    @property
    def pid(self) -> int:
        return int(self.proc.pid or 0)


class WorkerBackend:
    """The supervised-backend contract over a dedicated worker
    process — terminable where the in-process backend is not.

    One proxy instance supervises one loaded model. ``terminate()`` is
    the remedy cooperative cancellation cannot reach; the router
    invokes it only after drain timeout + cancel grace have elapsed
    (never as a first resort)."""

    def __init__(self, spec: BackendSpec, *, executable=None,
                 start_timeout: float = 120.0, probe_timeout: float = 60.0,
                 shutdown_grace: float = 5.0,
                 request_watchdog: float | None = None,
                 env: dict | None = None):
        self.spec = spec
        # Resolve in the parent now: backend_id must answer before the
        # worker exists, and a broken spec fails at construction.
        self.backend_id = spec.backend_id()
        self.executable = str(executable or sys.executable)
        self.start_timeout = float(start_timeout)
        self.probe_timeout = float(probe_timeout)
        self.shutdown_grace = float(shutdown_grace)
        self.request_watchdog = (None if request_watchdog is None
                                 else float(request_watchdog))
        self.env = dict(env) if env else None
        self._loaded = False
        self._load_lock = threading.Lock()
        self._handles: list[_WorkerHandle] = []
        self._handles_lock = threading.Lock()
        atexit.register(self._atexit_cleanup)

    # --- serving-backend contract ------------------------------------
    def load(self, snapshot) -> _WorkerHandle:
        """Spawn the worker, boot the backend class inside it, run its
        ``load()`` on the staged snapshot. Any failure reaps the worker
        before raising — a failed prepare cannot leave a resident
        model behind."""
        with self._load_lock:
            if self._loaded:
                raise WorkerBackendError(
                    "this WorkerBackend already loaded a model — one "
                    "proxy supervises one worker; build another for a "
                    "second activation")
            h = self._spawn()
            try:
                self._rpc(h, {"op": "load", "snapshot": snapshot},
                          timeout=self.start_timeout, what="load")
            except Exception:
                self._kill_and_reap(h)
                raise
            self._loaded = True
            with self._handles_lock:
                self._handles.append(h)
            return h

    def health_probe(self, handle) -> None:
        h = self._check(handle)
        self._rpc(h, {"op": "probe"}, timeout=self.probe_timeout,
                  what="health probe")

    def infer(self, handle, request) -> dict:
        h = self._check(handle)
        req = dict(request or {})
        # The route's threading.Event cannot cross a process boundary —
        # the worker injects its own cancel event, which additionally
        # refuses a client-supplied cancel object (an in-process gap).
        req.pop("_cancel_event", None)
        return self._rpc(h, {"op": "infer", "request": req},
                         timeout=self.request_watchdog, what="inference")

    def cancel(self, handle) -> None:
        """Cooperative cancellation: set the worker-local event every
        in-flight request holds. Advisory — errors are swallowed; the
        preemptive remedy is ``terminate``."""
        h = self._check(handle)
        if h.dead is not None or h.unloaded:
            return
        try:
            self._rpc(h, {"op": "cancel"},
                      timeout=min(self.shutdown_grace, 2.0),
                      what="cancel")
        except Exception:  # noqa: BLE001 - advisory only
            pass

    def unload(self, handle) -> None:
        """Graceful release with kill escalation: ask the worker to run
        its backend's ``unload`` and exit; if it has not exited inside
        ``shutdown_grace`` it is terminated. Idempotent — safe under
        abort and recovery paths that may call it twice."""
        h = self._check(handle)
        if h.unloaded:
            return
        h.unloaded = True
        if h.dead is None:
            try:
                self._rpc(h, {"op": "unload"},
                          timeout=self.shutdown_grace, what="unload")
            except Exception:  # noqa: BLE001 - the kill below is authoritative
                pass
        try:
            h.proc.wait(timeout=self.shutdown_grace)
        except subprocess.TimeoutExpired:
            self.terminate(h)
            return
        except Exception:  # noqa: BLE001
            self.terminate(h)
            return
        if h.proc.poll() is None:
            self.terminate(h)
            return
        self._close(h)

    def terminate(self, handle) -> None:
        """The RUN-401 remedy: SIGKILL the worker regardless of what it
        is doing. Pending requests fail fast with WorkerDied, their
        leases release, and the OS — not the backend — reclaims the
        model's resources."""
        h = self._check(handle)
        try:
            if h.proc.poll() is None:
                h.proc.kill()
        except OSError:
            pass
        self._mark_dead(h, WorkerDied(
            f"backend worker pid {h.pid} terminated"))
        try:
            h.proc.wait(timeout=10)
        except Exception:  # noqa: BLE001 - reap is best effort
            pass
        self._close(h)

    # --- introspection -------------------------------------------------
    def alive(self, handle) -> bool:
        h = self._check(handle)
        return h.dead is None and h.proc.poll() is None

    def worker_pid(self, handle) -> int:
        return self._check(handle).pid

    # --- internals ------------------------------------------------------
    def _check(self, handle) -> _WorkerHandle:
        if not isinstance(handle, _WorkerHandle) or handle.owner is not self:
            raise WorkerBackendError(
                "foreign handle — the supervisor only ever passes "
                "handles produced by this backend's load()")
        return handle

    def _spawn(self) -> _WorkerHandle:
        env = dict(os.environ)
        if self.env:
            env.update({str(k): str(v) for k, v in self.env.items()})
        pythonpath = os.pathsep.join(p for p in sys.path if p)
        env["PYTHONPATH"] = (pythonpath + os.pathsep + env["PYTHONPATH"]
                             if env.get("PYTHONPATH") else pythonpath)
        try:
            proc = subprocess.Popen(
                [self.executable, "-m", "minagi.runtime.worker_backend"],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, env=env)
        except OSError as exc:
            raise WorkerDied(
                f"worker interpreter could not start: {exc}") from exc
        h = _WorkerHandle(owner=self, proc=proc)
        reader = threading.Thread(
            target=self._reader, args=(h,),
            name=f"minagi-worker-rx-{proc.pid}", daemon=True)
        h.reader = reader
        reader.start()
        try:
            self._rpc(
                h, {"op": "boot", "module": self.spec.module,
                    "qualname": self.spec.qualname,
                    "kwargs": dict(self.spec.kwargs),
                    "sys_path": [p for p in sys.path if p],
                    "request_id": "__boot__"},
                timeout=self.start_timeout, what="boot")
        except Exception:
            self._kill_and_reap(h)
            raise
        return h

    def _rid(self, h: _WorkerHandle) -> str:
        h.seq += 1
        return f"r{h.seq}"

    def _rpc(self, h: _WorkerHandle, frame: dict, *,
             timeout: float | None, what: str):
        if h.dead is not None:
            raise h.dead
        if h.stalled and frame.get("op") not in ("unload", "cancel"):
            raise WorkerUnresponsive(
                f"{what}: worker pid {h.pid} is stalled — a wedged "
                "backend is retired or terminated, not given more work")
        frame = dict(frame)
        p = _Pending()
        with h.send_lock:
            # the request id is allocated under the same lock that owns
            # the pending map — concurrent in-flight requests can never
            # collide and misroute each other's replies
            rid = str(frame.setdefault("request_id", self._rid(h)))
            try:
                blob = pickle.dumps(frame)
            except Exception as exc:  # noqa: BLE001 - caller error, not death
                raise WorkerBackendError(
                    f"{what}: request frame is not picklable: {exc}") \
                    from exc
            h.pending[rid] = p
            try:
                h.proc.stdin.write(blob)
                h.proc.stdin.flush()
            except Exception as exc:  # noqa: BLE001 - broken pipe = dead
                h.pending.pop(rid, None)
                self._mark_dead(h, WorkerDied(
                    f"worker pid {h.pid} command channel broken: {exc}"))
                raise h.dead from exc
        if not p.event.wait(None if timeout is None else float(timeout)):
            with h.send_lock:
                h.pending.pop(rid, None)
            if h.dead is not None:
                raise h.dead
            if frame.get("op") == "infer":
                h.stalled = True
            raise WorkerUnresponsive(
                f"{what}: worker pid {h.pid} produced no response "
                f"within {timeout}s")
        if p.error is not None:
            raise p.error
        reply = p.reply or {}
        if reply.get("ok"):
            return reply.get("result")
        self._raise_remote(reply, what=what)

    def _raise_remote(self, reply: dict, *, what: str):
        msg = str(reply.get("error") or "remote backend error")
        tb = str(reply.get("traceback") or "")
        detail = (f"{what} failed in worker: {msg}"
                  + (f"\n--- worker traceback ---\n{tb}" if tb else ""))
        cls = _remote_exception_class(str(reply.get("error_module")),
                                      str(reply.get("error_type")))
        exc = None
        if cls is not None:
            try:
                exc = cls(msg)
            except Exception:  # noqa: BLE001 - signature mismatch
                exc = None
        if exc is not None:
            raise exc from RemoteBackendError(detail)
        raise RemoteBackendError(detail)

    def _reader(self, h: _WorkerHandle) -> None:
        try:
            while True:
                reply = pickle.load(h.proc.stdout)
                if not isinstance(reply, dict):
                    continue
                rid = str(reply.get("request_id") or "")
                with h.send_lock:
                    p = h.pending.pop(rid, None)
                if p is not None:
                    p.reply = reply
                    p.event.set()
        except Exception:  # noqa: BLE001 - any channel end means death
            pass
        code = None
        try:
            code = h.proc.wait(timeout=10)
        except Exception:  # noqa: BLE001 - reap is best effort here
            pass
        self._mark_dead(h, WorkerDied(
            f"backend worker pid {h.pid} exited"
            + (f" (status {code})" if code is not None else "")))

    def _mark_dead(self, h: _WorkerHandle, err: WorkerDied) -> None:
        with h.send_lock:
            if h.dead is None:
                h.dead = err
            pending = list(h.pending.values())
            h.pending = {}
        for p in pending:
            p.error = h.dead
            p.event.set()

    def _close(self, h: _WorkerHandle) -> None:
        self._mark_dead(h, WorkerDied(f"backend worker pid {h.pid} closed"))
        for stream in (h.proc.stdin, h.proc.stdout):
            try:
                stream.close()
            except Exception:  # noqa: BLE001
                pass
        with self._handles_lock:
            if h in self._handles:
                self._handles.remove(h)

    def _kill_and_reap(self, h: _WorkerHandle) -> None:
        try:
            if h.proc.poll() is None:
                h.proc.kill()
        except OSError:
            pass
        self._mark_dead(h, WorkerDied(
            f"backend worker pid {h.pid} killed"))
        try:
            h.proc.wait(timeout=10)
        except Exception:  # noqa: BLE001
            pass
        self._close(h)

    def _atexit_cleanup(self) -> None:
        """The supervisor process must never leave a resident model
        behind on the way out."""
        with self._handles_lock:
            handles = list(self._handles)
        for h in handles:
            try:
                self.terminate(h)
            except Exception:  # noqa: BLE001
                pass


# ---------------------------------------------------------------------------
# Worker child — executed by `python -m minagi.runtime.worker_backend`.
# ---------------------------------------------------------------------------

class _WorkerLoop:
    """In-worker command loop. Lifecycle ops (cancel/unload/shutdown)
    are handled on the main thread so they are always reachable;
    ``load``/``probe``/``infer`` run on daemon threads so a wedged
    request can never take the command channel with it."""

    def __init__(self, backend, out, send_lock):
        self.backend = backend
        self.handle = None
        self.cancel_event = threading.Event()
        self.out = out
        self.send_lock = send_lock

    def _reply(self, rid, ok, result=None, error=None):
        msg = {"request_id": rid, "ok": bool(ok)}
        if ok:
            msg["result"] = result
        else:
            msg.update(error=str(error),
                       error_type=type(error).__name__,
                       error_module=type(error).__module__,
                       traceback=traceback.format_exc())
        try:
            blob = pickle.dumps(msg)
        except Exception as exc:  # noqa: BLE001 - report honestly
            # A result that cannot cross the wire (a live tensor, an
            # unpicklable value) must not silently drop the frame —
            # the caller would hang until watchdog. Substitute an
            # error reply instead of corrupting the stream with a
            # partial pickle.
            blob = pickle.dumps({
                "request_id": rid, "ok": False,
                "error": f"backend result is not picklable: {exc}",
                "error_type": "WorkerBackendError",
                "error_module": "minagi.runtime.worker_backend",
                "traceback": ""})
        try:
            with self.send_lock:
                self.out.write(blob)
                self.out.flush()
        except Exception:  # noqa: BLE001 - parent is gone
            pass

    def serve(self) -> None:
        inp = sys.stdin.buffer
        while True:
            try:
                msg = pickle.load(inp)
            except Exception:  # noqa: BLE001 - EOF = parent is gone
                return
            if not isinstance(msg, dict):
                continue
            op = msg.get("op")
            rid = msg.get("request_id")
            if op == "shutdown":
                return
            if op == "load":
                threading.Thread(target=self._do_load, args=(msg,),
                                 daemon=True).start()
            elif op == "probe":
                threading.Thread(target=self._do_probe, args=(rid,),
                                 daemon=True).start()
            elif op == "infer":
                threading.Thread(target=self._do_infer, args=(msg,),
                                 daemon=True).start()
            elif op == "cancel":
                self.cancel_event.set()
                try:
                    cancel = getattr(self.backend, "cancel", None)
                    if callable(cancel) and self.handle is not None:
                        cancel(self.handle)
                except Exception:  # noqa: BLE001 - cooperative
                    pass
                self._reply(rid, True, result={"cancelled": True})
            elif op == "unload":
                try:
                    unload = getattr(self.backend, "unload", None)
                    if callable(unload) and self.handle is not None:
                        unload(self.handle)
                    self._reply(rid, True, result={"unloaded": True})
                except Exception as exc:  # noqa: BLE001
                    self._reply(rid, False, error=exc)
                return
            else:
                self._reply(rid, False, error=WorkerBackendError(
                    f"unknown worker op {op!r}"))

    def _do_load(self, msg) -> None:
        rid = msg.get("request_id")
        try:
            self.handle = self.backend.load(msg["snapshot"])
            self._reply(rid, True, result={"loaded": True})
        except Exception as exc:  # noqa: BLE001 - reported, then child dies
            self._reply(rid, False, error=exc)

    def _do_probe(self, rid) -> None:
        try:
            self.backend.health_probe(self.handle)
            self._reply(rid, True, result={"healthy": True})
        except Exception as exc:  # noqa: BLE001
            self._reply(rid, False, error=exc)

    def _do_infer(self, msg) -> None:
        rid = msg.get("request_id")
        try:
            if self.handle is None:
                raise WorkerBackendError("infer before load")
            req = dict(msg.get("request") or {})
            # The worker's own cancel event is injected into every
            # request — a client-supplied value can never reach the
            # backend through this boundary.
            req.pop("_cancel_event", None)
            req["_cancel_event"] = self.cancel_event
            infer = getattr(self.backend, "infer", None)
            if not callable(infer):
                infer = getattr(self.backend, "generate", None)
            if not callable(infer):
                raise WorkerBackendError(
                    "backend exposes no inference method")
            self._reply(rid, True, result=infer(self.handle, req))
        except Exception as exc:  # noqa: BLE001
            self._reply(rid, False, error=exc)


def _worker_entry() -> None:
    """Worker child entrypoint. The FIRST frame on stdin is the boot
    spec (module/qualname/kwargs/sys_path); the worker then replies
    '__boot__' and serves commands until shutdown, unload, or channel
    EOF (parent death — the worker must not outlive its supervisor)."""
    # Protocol safety: fd 1 is the reply channel. Duplicate it BEFORE
    # any backend code can print, then redirect fd 1 to fd 2 so stray
    # output lands on stderr instead of corrupting the frame stream.
    proto = os.fdopen(os.dup(1), "wb")
    os.dup2(2, 1)
    out_lock = threading.Lock()
    try:
        boot = pickle.load(sys.stdin.buffer)
    except Exception:  # noqa: BLE001 - nothing can be reported
        os._exit(2)
    sys.stdout = sys.stderr  # library print() must not hit the protocol
    try:
        for p in boot.get("sys_path") or ():
            if p and p not in sys.path:
                sys.path.append(p)
        mod = importlib.import_module(str(boot["module"]))
        obj = mod
        for part in str(boot["qualname"]).split("."):
            obj = getattr(obj, part)
        backend = obj(**dict(boot.get("kwargs") or {}))
    except Exception as exc:  # noqa: BLE001 - report and die
        with out_lock:
            pickle.dump({"request_id": "__boot__", "ok": False,
                         "error": str(exc),
                         "error_type": type(exc).__name__,
                         "error_module": type(exc).__module__,
                         "traceback": traceback.format_exc()}, proto)
            proto.flush()
        os._exit(2)
    with out_lock:
        pickle.dump({"request_id": "__boot__", "ok": True,
                     "result": {"pid": os.getpid()}}, proto)
        proto.flush()
    loop = _WorkerLoop(backend, proto, out_lock)
    code = 0
    try:
        loop.serve()
    except Exception:  # noqa: BLE001 - a broken loop is not a clean exit
        code = 3
    finally:
        try:
            proto.flush()
        except Exception:  # noqa: BLE001
            pass
        os._exit(code)


if __name__ == "__main__":
    _worker_entry()


__all__ = ["BackendSpec", "RemoteBackendError", "WorkerBackend",
           "WorkerBackendError", "WorkerDied", "WorkerUnresponsive"]
