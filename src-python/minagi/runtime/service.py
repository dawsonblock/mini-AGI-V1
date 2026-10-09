"""v16.4.3 supervised launch service (HARDENING_PLAN WP2/WP3/WP7/WP8).

A signed Python object is not a security boundary against malicious code
in the same interpreter. This module wraps `ServingSupervisor` +
`TrustedRuntimeLauncher` + `ServingRouter` in a dedicated service
process that owns the runtime signing keys, the authority store, the
snapshot root, and the serving routing table, speaking to untrusted
callers over authenticated Unix sockets:

  * requests and responses are newline-delimited JSON, size-bounded
    BEFORE parse (`readline(MAX_REQUEST_BYTES + 1)`), with read/write
    and total-request deadlines (OPS-001);
  * every connection is peer-credential checked (SO_PEERCRED on Linux,
    `socket.getpeereid` where available); the authenticated uid is
    mapped to a role (`--research-uid`, `--operator-uid`) — a
    client-supplied `"role"` field is data, not identity (SEC-203);
  * the dispatcher enforces `require(principal, op)` BEFORE any side
    effect; privileged ops emit a signed audit event;
  * two endpoints: `research.sock` (proposals/status/inference only)
    and `operator.sock` (lifecycle operations). Socket permissions do
    not substitute for dispatcher authorization;
  * a fixed worker pool with a bounded queue and per-principal
    connection quotas; auth failures are rate-limited; `stop()` drains;
  * activation ids, snapshot destinations, and receipt paths are
    generated inside the service — clients receive artifact ids, never
    write locations (SEC-204);
  * platforms without peer-credential extraction fail closed
    (`--allow-insecure-dev` is development-only).

Deployment topology: run under a dedicated OS identity owning
`storage/{snapshots,receipts,state,revocations,quarantine}` (0700);
grant the research uid `--research-uid` (research socket 0660/group or
firewall-bounded) and operators `--operator-uid`.
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import struct
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .access_policy import (RESEARCH_OPS,
                            PrincipalContext, PolicyRefused,
                            audit_decision, principal_for_uid, require,
                            secure_dir)


class ServiceRefused(PermissionError):
    """The service refused a connection or request."""


MAX_REQUEST_BYTES = 1 << 20          # 1 MiB request bound
IO_TIMEOUT_SECONDS = 30.0            # read/write + total deadline
MAX_WORKERS = 4
MAX_CONNECTIONS = 32
PER_UID_CONNECTIONS = 8
AUTH_FAILURE_LIMIT = 5
AUTH_FAILURE_WINDOW_SECONDS = 60.0


def peer_uid(conn: socket.socket) -> int | None:
    """Extract the authenticated peer uid, or None when the platform
    cannot prove it (in which case the connection must be refused)."""
    if hasattr(socket, "getpeereid"):
        try:
            return int(conn.getpeereid()[0])
        except (AttributeError, OSError):
            return None
    if hasattr(socket, "SO_PEERCRED"):
        try:
            creds = conn.getsockopt(socket.SOL_SOCKET,
                                    socket.SO_PEERCRED,
                                    struct.calcsize("3i"))
            _pid, uid, _gid = struct.unpack("3i", creds)
            return int(uid)
        except OSError:
            return None
    return None


class SupervisorService:
    """The protected end of the launch boundary."""

    def __init__(self, socket_path, *, launcher, supervisor,
                 backend_factories: dict, uid_roles: dict[int, str],
                 router=None, audit_signer=None,
                 allow_insecure_dev: bool = False,
                 max_request_bytes: int = MAX_REQUEST_BYTES,
                 io_timeout: float = IO_TIMEOUT_SECONDS,
                 max_workers: int = MAX_WORKERS,
                 max_connections: int = MAX_CONNECTIONS,
                 per_uid_connections: int = PER_UID_CONNECTIONS,
                 socket_mode: int = 0o600, socket_group: int | None = None,
                 audit_sink=None):
        self.socket_path = Path(socket_path)
        self.launcher = launcher
        self.supervisor = supervisor
        self.backend_factories = dict(backend_factories)
        self.router = router
        self.uid_roles = {int(u): str(r) for u, r in uid_roles.items()}
        self.audit_signer = audit_signer
        self.allow_insecure_dev = bool(allow_insecure_dev)
        self.max_request_bytes = int(max_request_bytes)
        self.io_timeout = float(io_timeout)
        self.max_connections = int(max_connections)
        self.per_uid_connections = int(per_uid_connections)
        self.socket_mode = int(socket_mode)
        self.socket_group = socket_group
        self._audit_sink = audit_sink   # callable(audit_doc) or None
        self._sock: socket.socket | None = None
        self._stop = threading.Event()
        self._pool = ThreadPoolExecutor(max_workers=int(max_workers))
        self._conn_sem = threading.BoundedSemaphore(self.max_connections)
        self._uid_conn_count: dict[int, int] = {}
        self._uid_lock = threading.Lock()
        self._auth_failures: dict[int, list[float]] = {}

    # --- request handling -------------------------------------------
    def _handle_request(self, req: dict, principal: PrincipalContext,
                        allowed_ops: frozenset | None = None) -> dict:
        if not isinstance(req, dict):
            raise ServiceRefused("request must be a JSON object")
        op = req.get("op")
        if not isinstance(op, str):
            raise ServiceRefused("request requires an 'op' string")
        if allowed_ops is not None and op not in allowed_ops:
            raise ServiceRefused(
                f"operation {op!r} is not dispatched on this endpoint")
        # Authorization precedes any side effect (SEC-203): the role
        # comes from the authenticated uid map, never the request body.
        try:
            require(principal, op)
        except PolicyRefused as exc:
            self._audit(principal, op, reason=str(exc), refused=True)
            raise ServiceRefused(str(exc)) from exc
        if op == "ping":
            return {"ok": True, "service": "supervised-launch",
                    "runtime_identity": self.supervisor.runtime_identity,
                    "serving_state": self.supervisor.serving_state.value}
        if op == "status":
            return {"ok": True,
                    "active": self.supervisor.active_pointer(),
                    "serving_state": self.supervisor.serving_state.value,
                    "router": (self.router.status()
                               if self.router is not None else None)}
        if op == "infer":
            if self.router is None:
                raise ServiceRefused(
                    "inference routing is not configured")
            try:
                out = self.router.route(req.get("request"))
            except Exception as exc:  # noqa: BLE001
                raise ServiceRefused(f"inference refused: {exc}") from exc
            self._audit(principal, op,
                        target=str(out.get("activation_id", "")))
            return {"ok": True, **out}
        if op == "recover":
            self._audit(principal, op)
            return {"ok": True,
                    "report": self.supervisor.recover()}
        if op == "quarantine":
            self.supervisor.quarantine_active(
                reason=str(req.get("reason") or "operator quarantine"))
            self._audit(principal, op, reason=str(req.get("reason") or ""))
            return {"ok": True,
                    "active": self.supervisor.active_pointer()}
        if op == "rollback":
            try:
                target = self.supervisor.rollback()
            except Exception as exc:  # noqa: BLE001
                raise ServiceRefused(f"rollback: {exc}") from exc
            self._audit(principal, op, target=str(target or ""))
            return {"ok": True, "active": target}
        if op == "launch":
            result = self._launch(req)
            self._audit(principal, op,
                        target=str(result.get("activation_id", "")))
            return result
        raise ServiceRefused(f"unknown op {op!r}")

    def _audit(self, principal: PrincipalContext, op: str, *,
               target: str = "", reason: str = "",
               refused: bool = False) -> None:
        if self.audit_signer is None:
            return
        try:
            doc = audit_decision(
                self.audit_signer, principal=principal, operation=op,
                target=target,
                reason=reason or ("refused" if refused else "ok"),
                at=int(time.time()))
        except Exception:  # noqa: BLE001 - audit must not break dispatch
            return
        if self._audit_sink is not None:
            try:
                self._audit_sink(doc)
            except Exception:  # noqa: BLE001
                pass

    def _launch(self, req: dict) -> dict:
        from minagi.v161.trusted_launcher import (LaunchReplay,
                                                  LaunchRequest)
        spec = str(req.get("backend") or "hf-peft")
        factory = self.backend_factories.get(spec)
        if factory is None:
            raise ServiceRefused(
                f"no service-registered backend factory for {spec!r} — "
                "clients name backends, they do not supply them")
        body = dict(req.get("request") or {})
        body["expected_backend"] = spec
        try:
            request = LaunchRequest(**body)
        except (TypeError, ValueError) as exc:
            raise ServiceRefused(f"launch request invalid: {exc}") from exc
        request_id = str(req.get("request_id") or "")
        try:
            result = self.launcher.launch(request, backend=factory(),
                                          request_id=request_id or None)
        except LaunchReplay as replay:
            # Idempotent retry: return the recorded outcome — no second
            # activation, no second grant.
            return {"ok": True, "replayed": True, **replay.prior}
        return {"ok": True, "backend_id": result.backend_id,
                "receipt_digest": result.receipt_doc["digest"],
                "activation_id": result.activation_id,
                "receipt_path": str(result.receipt_path),
                "loaded_artifact_digests":
                    dict(result.receipt.loaded_artifact_digests),
                "activation_nonce": result.receipt.activation_nonce}

    # --- transport ---------------------------------------------------
    def _authenticate(self, conn: socket.socket) -> PrincipalContext:
        uid = peer_uid(conn)
        if uid is None:
            if not self.allow_insecure_dev:
                raise ServiceRefused(
                    "peer credentials unavailable on this platform — "
                    "connections cannot be authenticated")
            return principal_for_uid(None, {}, insecure_dev=True)
        principal = principal_for_uid(uid, self.uid_roles)
        if principal.role == "unauthenticated":
            self._auth_failure(uid)
            raise ServiceRefused(
                f"peer uid {uid} is not an allowed runtime client")
        return principal

    def _auth_failure(self, uid: int) -> None:
        """Rate-limit repeated unauthenticated connection attempts."""
        now = time.monotonic()
        with self._uid_lock:
            fails = [t for t in self._auth_failures.get(uid, [])
                     if now - t < AUTH_FAILURE_WINDOW_SECONDS]
            fails.append(now)
            self._auth_failures[uid] = fails
            if len(fails) >= AUTH_FAILURE_LIMIT:
                # soft-penalty: the connection still refuses; the delay
                # discourages credential-bounce loops
                time.sleep(min(2.0, 0.2 * len(fails)))

    def _admit_conn(self, uid: int | None) -> None:
        key = -1 if uid is None else int(uid)
        with self._uid_lock:
            count = self._uid_conn_count.get(key, 0)
            if count >= self.per_uid_connections:
                raise ServiceRefused(
                    "per-principal connection quota exhausted")
            self._uid_conn_count[key] = count + 1

    def _release_conn(self, uid: int | None) -> None:
        key = -1 if uid is None else int(uid)
        with self._uid_lock:
            left = self._uid_conn_count.get(key, 0) - 1
            if left <= 0:
                self._uid_conn_count.pop(key, None)
            else:
                self._uid_conn_count[key] = left

    def _serve_conn(self, conn: socket.socket,
                    allowed_ops: frozenset | None) -> None:
        uid_for_quota: int | None = None
        admitted = False
        try:
            conn.settimeout(self.io_timeout)
            principal = self._authenticate(conn)
            uid_for_quota = None if principal.uid < 0 else principal.uid
            self._admit_conn(uid_for_quota)
            admitted = True
            f = conn.makefile("rwb")
            line = f.readline(self.max_request_bytes + 1)
            if not line:
                raise ServiceRefused("empty request")
            if len(line) > self.max_request_bytes:
                raise ServiceRefused("request exceeds the size bound")
            try:
                req = json.loads(line)
            except ValueError as exc:
                raise ServiceRefused(f"request is not JSON: {exc}")
            try:
                resp = self._handle_request(req, principal,
                                            allowed_ops=allowed_ops)
            except ServiceRefused as exc:
                resp = {"ok": False, "refused": str(exc)}
            except Exception as exc:  # noqa: BLE001 - hygiene
                resp = {"ok": False,
                        "error": f"{type(exc).__name__}: {exc}"}
            f.write((json.dumps(resp) + "\n").encode())
            f.flush()
        except ServiceRefused as exc:
            try:
                f = conn.makefile("rwb")
                f.write((json.dumps(
                    {"ok": False, "refused": str(exc)}) + "\n").encode())
                f.flush()
            except OSError:
                pass
        except (OSError, socket.timeout):
            pass
        finally:
            if admitted:
                self._release_conn(uid_for_quota)
            self._conn_sem.release()
            try:
                conn.close()
            except OSError:
                pass

    def serve_forever(self, *,
                      allowed_ops: frozenset | None = None) -> None:
        self.socket_path.parent.mkdir(parents=True, exist_ok=True)
        if self.socket_path.exists():
            self.socket_path.unlink()
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.bind(str(self.socket_path))
        os.chmod(self.socket_path, self.socket_mode)
        if self.socket_group is not None:
            try:
                os.chown(self.socket_path, -1, int(self.socket_group))
            except OSError:
                pass
        sock.listen(16)
        self._sock = sock
        try:
            while not self._stop.is_set():
                try:
                    sock.settimeout(0.5)
                    conn, _ = sock.accept()
                except socket.timeout:
                    continue
                except OSError:
                    break  # stop() closed the socket
                if not self._conn_sem.acquire(blocking=False):
                    try:
                        conn.close()
                    except OSError:
                        pass
                    continue
                self._pool.submit(self._serve_conn, conn,
                                  allowed_ops)
        finally:
            sock.close()
            try:
                self.socket_path.unlink()
            except OSError:
                pass

    def stop(self) -> None:
        self._stop.set()
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass

    def shutdown(self) -> None:
        """Graceful stop: refuse new connections, drain the pool."""
        self.stop()
        self._pool.shutdown(wait=True)


def main(argv=None) -> int:
    """Run the supervised launch service.

        python -m minagi.runtime.service \
            --socket /run/minagi/operator.sock \
            --research-socket /run/minagi/research.sock \
            --storage-root STORAGE --operator-uid 0 --research-uid 1000
    """
    ap = argparse.ArgumentParser(prog="minagi.runtime.service")
    ap.add_argument("--socket", required=True,
                    help="operator endpoint (lifecycle operations)")
    ap.add_argument("--research-socket", default=None,
                    help="optional separate research endpoint "
                         "(proposals/status/inference only)")
    ap.add_argument("--storage-root", required=True)
    ap.add_argument("--operator-uid", action="append", type=int,
                    default=[], help="peer uid with operator role")
    ap.add_argument("--research-uid", action="append", type=int,
                    default=[], help="peer uid with research role")
    ap.add_argument("--allowed-uid", action="append", type=int,
                    default=[], help="deprecated alias for "
                                     "--research-uid")
    ap.add_argument("--state-dir", default=None,
                    help="authority store directory (default "
                         "<storage>/state)")
    ap.add_argument("--socket-mode", default="0600")
    ap.add_argument("--socket-group", type=int, default=None)
    ap.add_argument("--allow-insecure-dev", action="store_true",
                    help="development only: serve when peer credentials "
                         "cannot be extracted")
    args = ap.parse_args(argv)

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from egai.common.crypto import Ed25519Signer
    from minagi.runtime.authority_store import AuthorityStore
    from minagi.runtime.serving_router import ServingRouter
    from minagi.runtime.supervisor import ServingSupervisor
    from minagi.v161.authority import AuthorityRegistry
    from minagi.v161.peft_serving import PeftServingBackend
    from minagi.v161.trusted_launcher import TrustedRuntimeLauncher
    from minagi.security.signed_revocations import RevocationStore

    storage = Path(args.storage_root).resolve()
    snapshots = secure_dir(storage / "snapshots")
    receipts = secure_dir(storage / "receipts")
    state_dir = secure_dir(
        args.state_dir or (storage / "state"))
    secure_dir(storage / "quarantine")

    registry = AuthorityRegistry.load(storage / "trust_root.json")
    runtime_key = Ed25519Signer.from_private_bytes(
        (storage / ".keys" / "runtime.pem").read_bytes())
    admission_key = Ed25519Signer.from_private_bytes(
        (storage / ".keys" / "admission.pem").read_bytes())

    store = AuthorityStore(state_dir / "authority.sqlite")
    supervisor = ServingSupervisor(
        store, runtime_signer=runtime_key, registry=registry)
    report = supervisor.recover()
    router = ServingRouter()

    launcher = TrustedRuntimeLauncher(
        registry,
        revocation_store=RevocationStore(storage / "revocations"),
        runtime_signer=runtime_key, admission_signer=admission_key,
        supervisor=supervisor,
        snapshot_root=snapshots,
        receipts_dir=receipts,
        authority_store=store)
    supervisor.router = router

    uid_roles = {u: "operator" for u in args.operator_uid}
    for u in args.research_uid + args.allowed_uid:
        uid_roles.setdefault(u, "research")
    uid_roles.setdefault(os.getuid(), "operator")

    mode = int(args.socket_mode, 8)
    service = SupervisorService(
        args.socket, launcher=launcher, supervisor=supervisor,
        backend_factories={"hf-peft": PeftServingBackend},
        uid_roles=uid_roles, router=router, audit_signer=runtime_key,
        socket_mode=mode, socket_group=args.socket_group,
        allow_insecure_dev=args.allow_insecure_dev)
    print(f"[supervised-launch] operator endpoint on {args.socket} "
          f"(roles: {sorted(set(uid_roles.values()))}) "
          f"state={report.get('serving_state')}", flush=True)
    try:
        if args.research_socket:
            research = SupervisorService(
                args.research_socket, launcher=launcher,
                supervisor=supervisor,
                backend_factories={"hf-peft": PeftServingBackend},
                uid_roles=uid_roles, router=router,
                audit_signer=runtime_key, socket_mode=0o660,
                socket_group=args.socket_group,
                allow_insecure_dev=args.allow_insecure_dev)
            threading.Thread(
                target=research.serve_forever,
                kwargs={"allowed_ops": RESEARCH_OPS}, daemon=True).start()
        service.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        service.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
