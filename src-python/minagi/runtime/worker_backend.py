"""v16.4.5 worker-process backend isolation (RUN-401 + SEC-401).

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

Trust boundary (v16.4.5): the channel is the
``minagi-worker-v2`` framed-JSON protocol
(`worker_protocol`) — never pickle, in either direction. A
compromised worker can send malformed, hostile, or enormous payloads;
the supervisor validates structure before acting and executes nothing
the worker constructs. The snapshot crosses as a *descriptor* (root +
authorized digests) that the worker re-measures into a real
``MeasuredSnapshot`` — serialized measurement evidence is not trusted,
measurement is repeated. The worker's environment is constructed, not
inherited (`worker_isolation`): no signing credentials, a private
scratch directory, POSIX resource bounds where the platform enforces
them, and an optional explicit identity demotion.

The protocol channel is fd 1 duplicated before any backend import —
the worker's own fd 1 is then redirected to stderr, so library
``print`` calls can never corrupt the frame stream.
"""
from __future__ import annotations

import atexit
import importlib
import os
import subprocess
import sys
import threading
import traceback
from dataclasses import dataclass, field

from .worker_isolation import (DEFAULT_ISOLATION, IsolationError,
                               WorkerIsolationPolicy,
                               WorkerProcessController, build_env,
                               descendants_of, make_private_tmp,
                               make_worker_scratch, new_worker_unit,
                               worker_preexec, worker_unit_env,
                               wrap_argv)
from .worker_protocol import (MAX_PENDING_REQUESTS, OverflowRef,
                              ProtocolRefused, WorkerProtocolError,
                              decode_typed, encode_request,
                              encode_response, encode_typed,
                              read_frame, read_overflow, write_overflow)


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
    """The in-worker backend raised. ``code`` is one of the bounded
    WORKER_* error codes — worker-controlled module/class names are
    never resolved into live exception types in the supervisor
    (v16.4.6 WP3/SEC-503), so a hostile worker cannot select
    ``SystemExit``/``KeyboardInterrupt`` and alter privileged control
    flow here."""

    def __init__(self, message, *, code: str = "WORKER_INTERNAL_ERROR"):
        self.code = code
        super().__init__(message)


class WorkerProtocolViolation(WorkerDied):
    """The worker emitted bytes that are not a valid protocol v2
    message — a compromised or corrupted worker is terminated, never
    negotiated with. Subclasses WorkerDied: a violated channel is a
    dead worker for every caller that only needs to know the model is
    gone."""


#: Bounded error vocabulary (WP3). The worker reports a CODE; only the
#: supervisor decides which local handling a code maps to. Unknown
#: or missing codes collapse to WORKER_INTERNAL_ERROR.
WORKER_ERROR_CODES = frozenset({
    "WORKER_LOAD_FAILED", "WORKER_INFERENCE_FAILED", "WORKER_CANCELLED",
    "WORKER_BUDGET_EXCEEDED", "WORKER_PROTOCOL_VIOLATION",
    "WORKER_UNAVAILABLE", "WORKER_INTERNAL_ERROR"})

_MAX_REMOTE_TRACEBACK = 4000


def _local_error_class(code: str):
    """The STATIC code -> local exception map. Only ordinary
    ``Exception`` subclasses may appear here — control-flow classes
    (``SystemExit``, ``KeyboardInterrupt``, ``GeneratorExit``) can
    never be selected by worker data."""
    if code == "WORKER_BUDGET_EXCEEDED":
        from .inference_policy import BudgetExceeded
        return BudgetExceeded
    return None


_OP_ERROR_CODES = {
    "load": "WORKER_LOAD_FAILED",
    "infer": "WORKER_INFERENCE_FAILED",
    "probe": "WORKER_UNAVAILABLE",
}


def _error_code(error, op: str | None = None) -> str:
    """Worker side: the exception maps to one bounded CODE. Its real
    class name travels only as inert text for the detail message."""
    try:
        from .inference_policy import BudgetExceeded
        if isinstance(error, BudgetExceeded):
            return "WORKER_BUDGET_EXCEEDED"
    except Exception:  # noqa: BLE001
        pass
    if isinstance(error, (ProtocolRefused, WorkerProtocolError)):
        return "WORKER_PROTOCOL_VIOLATION"
    return _OP_ERROR_CODES.get(op or "", "WORKER_INTERNAL_ERROR")


@dataclass(frozen=True)
class BackendSpec:
    """Which backend the worker instantiates — operator configuration
    equivalent in trust to the service's factory registry. ``kwargs``
    must be JSON-safe or implement ``to_doc`` (``worker_protocol``
    typed values); they cross the process boundary as data only."""
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
                 "seq", "dead", "stalled", "unloaded", "private_tmp",
                 "overflow_dir", "overflow_fd", "tracked_pids", "unit")

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
        self.private_tmp = None
        self.overflow_dir = None
        self.overflow_fd = None
        self.tracked_pids = frozenset()
        self.unit = None

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
                 env: dict | None = None,
                 isolation: WorkerIsolationPolicy | None = None,
                 backend_id: str | None = None):
        self.spec = spec
        # backend_id: supplied explicitly, it comes from the verified
        # signed manifest (production — the parent never imports the
        # backend module itself); otherwise resolve the pinned,
        # trusted spec in the parent so a broken spec fails at
        # construction, before the worker exists.
        if backend_id is not None:
            if not isinstance(backend_id, str) or not backend_id:
                raise WorkerBackendError(
                    "an explicitly supplied backend_id must be a "
                    "non-empty string")
            self.backend_id = backend_id
        else:
            self.backend_id = spec.backend_id()
        self.executable = str(executable or sys.executable)
        self.start_timeout = float(start_timeout)
        self.probe_timeout = float(probe_timeout)
        self.shutdown_grace = float(shutdown_grace)
        self.request_watchdog = (None if request_watchdog is None
                                 else float(request_watchdog))
        self.isolation = isolation or DEFAULT_ISOLATION
        self.controller = WorkerProcessController()
        if env:
            # Backward-compatible surface: explicit env additions are
            # still refused when they look credential-bearing. Every
            # policy field — including v16.4.6 enforcement flags — is
            # preserved across the merge.
            merged = dict(self.isolation.env)
            merged.update({str(k): str(v) for k, v in env.items()})
            self.isolation = WorkerIsolationPolicy(
                env_passthrough=self.isolation.env_passthrough,
                env=merged,
                cpu_seconds=self.isolation.cpu_seconds,
                memory_bytes=self.isolation.memory_bytes,
                max_processes=self.isolation.max_processes,
                max_open_files=self.isolation.max_open_files,
                max_file_bytes=self.isolation.max_file_bytes,
                demote_to=self.isolation.demote_to,
                sandbox_profile=self.isolation.sandbox_profile,
                require_separate_identity=(
                    self.isolation.require_separate_identity),
                require_filesystem_confinement=(
                    self.isolation.require_filesystem_confinement),
                require_process_containment=(
                    self.isolation.require_process_containment),
                require_network_isolation=(
                    self.isolation.require_network_isolation))
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
        model behind. The snapshot crosses as a descriptor; the worker
        re-measures it into a real MeasuredSnapshot."""
        from minagi.v161.immutable_snapshot import snapshot_descriptor
        with self._load_lock:
            if self._loaded:
                raise WorkerBackendError(
                    "this WorkerBackend already loaded a model — one "
                    "proxy supervises one worker; build another for a "
                    "second activation")
            h = self._spawn()
            try:
                self._rpc(h, {"op": "load",
                              "snapshot": snapshot_descriptor(snapshot)},
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
        """The RUN-401 remedy: terminate the worker's whole containment
        unit regardless of what it is doing — v16.4.6 WP2. The worker
        owns its process group (``start_new_session``); the controller
        snapshots live descendants FIRST, signals the group, then
        verifies via the unit marker that no owned process survived.
        Pending requests fail fast with WorkerDied, their leases
        release, and the OS — not the backend — reclaims the model's
        resources."""
        h = self._check(handle)
        if h.proc.poll() is None:
            h.tracked_pids = frozenset(
                descendants_of(h.pid) | {h.pid})
        survivors = self.controller.terminate_tree(
            h.proc, leader_pid=h.pid,
            tracked=set(h.tracked_pids), unit=h.unit)
        if survivors:
            self._mark_dead(h, WorkerDied(
                f"backend worker pid {h.pid} terminated; descendant "
                f"processes {sorted(survivors)} could not be confirmed "
                "dead — containment is not proven"))
        else:
            self._mark_dead(h, WorkerDied(
                f"backend worker pid {h.pid} and its process tree "
                "terminated"))
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
        # Worker-owned scratch when the policy demotes (SEC-501): the
        # demoted uid must still be able to enter its own workspace —
        # group is the supervisor's so result collection stays
        # supervisor-readable without world access.
        if self.isolation.demote_to is not None:
            private_tmp = make_worker_scratch(
                self.isolation.demote_to, group_gid=os.getgid())
        else:
            private_tmp = make_private_tmp()
        overflow_dir = private_tmp / "overflow"
        try:
            overflow_dir.mkdir(mode=0o750)
            if self.isolation.demote_to is not None:
                os.chown(overflow_dir, int(self.isolation.demote_to[0]),
                         os.getgid())
            # SEC-504: pin the directory's descriptor NOW, before the
            # worker can influence the path — later result reads open
            # the leaf relative to this inode, so renaming the dir or
            # planting a symlink cannot redirect the read.
            overflow_fd = os.open(
                str(overflow_dir),
                os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_NOFOLLOW", 0))
        except OSError as exc:
            import shutil
            shutil.rmtree(private_tmp, ignore_errors=True)
            raise IsolationError(
                f"could not establish the worker result directory: "
                f"{exc}") from exc
        pythonpath = os.pathsep.join(p for p in sys.path if p)
        env = build_env(self.isolation, pythonpath=pythonpath,
                        private_tmp=private_tmp)
        # The containment-unit token — inherited by every descendant,
        # lets the supervisor prove nothing owned survives teardown.
        unit = new_worker_unit()
        env.update(worker_unit_env(unit))
        argv = wrap_argv(
            [self.executable, "-m", "minagi.runtime.worker_backend"],
            self.isolation)
        try:
            proc = subprocess.Popen(
                argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                env=env, cwd=str(private_tmp),
                start_new_session=True,
                preexec_fn=worker_preexec(self.isolation))
        except (OSError, IsolationError) as exc:
            os.close(overflow_fd)
            import shutil
            shutil.rmtree(private_tmp, ignore_errors=True)
            raise WorkerDied(
                f"worker interpreter could not start: {exc}") from exc
        h = _WorkerHandle(owner=self, proc=proc)
        h.private_tmp = private_tmp
        h.overflow_dir = overflow_dir
        h.overflow_fd = overflow_fd
        h.unit = unit
        reader = threading.Thread(
            target=self._reader, args=(h,),
            name=f"minagi-worker-rx-{proc.pid}", daemon=True)
        h.reader = reader
        reader.start()
        try:
            self._rpc(
                h, {"op": "boot", "module": self.spec.module,
                    "qualname": self.spec.qualname,
                    "kwargs": encode_typed(dict(self.spec.kwargs)),
                    "sys_path": [p for p in sys.path if p],
                    "overflow_dir": str(overflow_dir),
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
        p = _Pending()
        with h.send_lock:
            # the request id is allocated under the same lock that owns
            # the pending map — concurrent in-flight requests can never
            # collide and misroute each other's replies
            rid = str(frame.get("request_id") or self._rid(h))
            if len(h.pending) >= MAX_PENDING_REQUESTS:
                raise WorkerBackendError(
                    f"{what}: {len(h.pending)} requests already "
                    f"outstanding — the per-worker bound "
                    f"({MAX_PENDING_REQUESTS}) is refused")
            op = frame["op"]
            payload = {k: v for k, v in frame.items()
                       if k not in ("op", "request_id")}
            try:
                blob = encode_request(rid, op, payload)
            except Exception as exc:  # noqa: BLE001 - caller error, not death
                raise WorkerBackendError(
                    f"{what}: request frame is not encodable: {exc}") \
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
            payload = reply.get("payload") or {}
            ref = payload.get("overflow")
            if ref is not None:
                try:
                    return read_overflow(
                        h.overflow_dir,
                        OverflowRef.from_doc(ref),
                        dir_fd=h.overflow_fd)
                except (WorkerProtocolError, ProtocolRefused) as exc:
                    raise WorkerBackendError(
                        f"{what}: overflow result refused: {exc}") \
                        from exc
            return payload.get("result")
        self._raise_remote(reply.get("payload") or {}, what=what)

    def _raise_remote(self, payload: dict, *, what: str):
        """v16.4.6 WP3/SEC-503: the worker reports an error CODE plus
        informational strings; only the supervisor's static map
        decides what local type (if any) represents it. Worker-
        supplied module/class names are data in the detail text —
        never resolved into live classes, so nothing the worker
        sends can instantiate SystemExit or its kin here."""
        msg = str(payload.get("error") or "remote backend error")
        tb = str(payload.get("traceback") or "")[:_MAX_REMOTE_TRACEBACK]
        code = str(payload.get("error_code") or "")
        if code not in WORKER_ERROR_CODES:
            code = "WORKER_INTERNAL_ERROR"
        remote_kind = ".".join(
            p for p in (str(payload.get("error_module") or "")[:120],
                        str(payload.get("error_type") or "")[:80]) if p)
        detail = (f"{what} failed in worker"
                  + (f" [{remote_kind}]" if remote_kind else "")
                  + f": {msg}"
                  + (f"\n--- worker traceback ---\n{tb}" if tb else ""))
        cls = _local_error_class(code)
        if cls is not None:
            raise cls(msg) from RemoteBackendError(detail, code=code)
        raise RemoteBackendError(detail, code=code)

    def _reader(self, h: _WorkerHandle) -> None:
        """Demultiplex worker replies. The ONLY thing this thread does
        with worker-controlled bytes is validate them against the
        protocol — a violation marks the worker dead and fails every
        pending request; nothing worker-supplied is ever executed."""
        from .worker_protocol import RESPONSE_TYPES
        violation: str | None = None
        try:
            while True:
                reply = read_frame(h.proc.stdout, types=RESPONSE_TYPES)
                if reply is None:
                    break
                rid = str(reply.get("request_id") or "")
                with h.send_lock:
                    p = h.pending.pop(rid, None)
                if p is not None:
                    p.reply = reply
                    p.event.set()
        except (ProtocolRefused, WorkerProtocolError) as exc:
            violation = str(exc)
        except Exception as exc:  # noqa: BLE001 - any channel end means death
            violation = f"channel error: {exc}"
        code = None
        try:
            code = h.proc.wait(timeout=10)
        except Exception:  # noqa: BLE001 - reap is best effort here
            pass
        if violation is not None:
            self._mark_dead(h, WorkerProtocolViolation(
                f"backend worker pid {h.pid} violated the IPC "
                f"protocol: {violation}"))
        else:
            self._mark_dead(h, WorkerDied(
                f"backend worker pid {h.pid} exited"
                + (f" (status {code})" if code is not None else "")))

    def _mark_dead(self, h: _WorkerHandle, err: WorkerDied) -> None:
        with h.send_lock:
            if h.dead is not None:
                return
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
        if h.overflow_fd is not None:
            try:
                os.close(h.overflow_fd)
            except OSError:
                pass
            h.overflow_fd = None
        if h.private_tmp is not None:
            import shutil
            shutil.rmtree(h.private_tmp, ignore_errors=True)
        with self._handles_lock:
            if h in self._handles:
                self._handles.remove(h)

    def _kill_and_reap(self, h: _WorkerHandle) -> None:
        if h.proc.poll() is None:
            h.tracked_pids = frozenset(
                descendants_of(h.pid) | {h.pid})
        survivors = self.controller.terminate_tree(
            h.proc, leader_pid=h.pid,
            tracked=set(h.tracked_pids), unit=h.unit)
        if survivors:
            self._mark_dead(h, WorkerDied(
                f"backend worker pid {h.pid} killed; descendants "
                f"{sorted(survivors)} could not be confirmed dead"))
        else:
            self._mark_dead(h, WorkerDied(
                f"backend worker pid {h.pid} killed"))
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

    def __init__(self, backend, out, send_lock, overflow_dir=None):
        self.backend = backend
        self.handle = None
        self.cancel_event = threading.Event()
        self.out = out
        self.send_lock = send_lock
        self.overflow_dir = overflow_dir

    def _reply(self, rid, ok, result=None, error=None, op=None):
        payload: dict = {}
        if ok:
            try:
                blob = encode_response(rid, True, {"result": result})
            except WorkerProtocolError:
                # A result that cannot fit a control frame travels
                # content-addressed: written into the parent-provided
                # overflow dir, referenced by name+digest.
                if self.overflow_dir is None:
                    blob = encode_response(
                        rid, False, {
                            "error": "result exceeds the frame bound "
                                     "and no overflow directory was "
                                     "provided",
                            "error_type": "WorkerBackendError",
                            "error_module": "minagi.runtime.worker_backend",
                            "traceback": ""})
                else:
                    try:
                        ref = write_overflow(
                            self.overflow_dir, f"{rid}.result.json",
                            result)
                        blob = encode_response(
                            rid, True, {"overflow": ref.to_doc()})
                    except Exception as exc:  # noqa: BLE001
                        blob = encode_response(
                            rid, False, {
                                "error": f"result overflow failed: {exc}",
                                "error_type": "WorkerBackendError",
                                "error_module":
                                    "minagi.runtime.worker_backend",
                                "traceback": ""})
        else:
            blob = encode_response(
                rid, False, {
                    "error": str(error)[:2000],
                    "error_code": _error_code(error, op),
                    "error_type": type(error).__name__[:80],
                    "error_module": type(error).__module__[:120],
                    "traceback": traceback.format_exc()[-4000:]})
        try:
            with self.send_lock:
                self.out.write(blob)
                self.out.flush()
        except Exception:  # noqa: BLE001 - parent is gone
            pass

    def serve(self) -> None:
        from .worker_protocol import REQUEST_TYPES
        inp = sys.stdin.buffer
        while True:
            try:
                msg = read_frame(inp, types=REQUEST_TYPES)
            except Exception:  # noqa: BLE001 - EOF or violation = done
                return
            if msg is None:
                return
            op = msg.get("type")
            rid = msg.get("request_id")
            payload = msg.get("payload") or {}
            if op == "shutdown":
                return
            if op == "load":
                threading.Thread(target=self._do_load, args=(rid, payload),
                                 daemon=True).start()
            elif op == "probe":
                threading.Thread(target=self._do_probe, args=(rid,),
                                 daemon=True).start()
            elif op == "infer":
                threading.Thread(target=self._do_infer,
                                 args=(rid, payload),
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

    def _do_load(self, rid, payload) -> None:
        try:
            from minagi.v161.immutable_snapshot import (
                snapshot_from_descriptor)
            self.handle = self.backend.load(
                snapshot_from_descriptor(payload.get("snapshot")))
            self._reply(rid, True, result={"loaded": True})
        except Exception as exc:  # noqa: BLE001 - reported, then child dies
            self._reply(rid, False, error=exc, op="load")

    def _do_probe(self, rid) -> None:
        try:
            self.backend.health_probe(self.handle)
            self._reply(rid, True, result={"healthy": True})
        except Exception as exc:  # noqa: BLE001
            self._reply(rid, False, error=exc, op="probe")

    def _do_infer(self, rid, payload) -> None:
        try:
            if self.handle is None:
                raise WorkerBackendError("infer before load")
            req = dict(payload.get("request") or {})
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
    spec (module/qualname/kwargs/sys_path/overflow_dir); the worker
    then replies '__boot__' and serves commands until shutdown,
    unload, or channel EOF (parent death — the worker must not
    outlive its supervisor)."""
    from .worker_protocol import REQUEST_TYPES
    # Protocol safety: fd 1 is the reply channel. Duplicate it BEFORE
    # any backend code can print, then redirect fd 1 to fd 2 so stray
    # output lands on stderr instead of corrupting the frame stream.
    proto = os.fdopen(os.dup(1), "wb")
    os.dup2(2, 1)
    out_lock = threading.Lock()
    try:
        boot = read_frame(sys.stdin.buffer, types=REQUEST_TYPES)
    except Exception:  # noqa: BLE001 - nothing can be reported
        os._exit(2)
    sys.stdout = sys.stderr  # library print() must not hit the protocol
    if boot is None or boot.get("type") != "boot":
        os._exit(2)
    bpl = boot.get("payload") or {}
    overflow_dir = bpl.get("overflow_dir")
    try:
        for p in bpl.get("sys_path") or ():
            if p and p not in sys.path:
                sys.path.append(p)
        mod = importlib.import_module(str(bpl["module"]))
        obj = mod
        for part in str(bpl["qualname"]).split("."):
            obj = getattr(obj, part)
        backend = obj(**dict(decode_typed(bpl.get("kwargs") or {})))
    except Exception as exc:  # noqa: BLE001 - report and die
        with out_lock:
            proto.write(encode_response(
                "__boot__", False, {
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                    "error_module": type(exc).__module__,
                    "traceback": traceback.format_exc()}))
            proto.flush()
        os._exit(2)
    with out_lock:
        proto.write(encode_response(
            "__boot__", True, {"result": {"pid": os.getpid()}}))
        proto.flush()
    loop = _WorkerLoop(backend, proto, out_lock,
                       overflow_dir=str(overflow_dir)
                       if overflow_dir else None)
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
           "WorkerBackendError", "WorkerDied", "WorkerProtocolViolation",
           "WorkerUnresponsive"]
