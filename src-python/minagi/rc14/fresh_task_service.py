from __future__ import annotations

from dataclasses import asdict
import json
import os
from pathlib import Path
import secrets
import socket
import threading
import time
from typing import Any

from .fresh_tasks import DurableFreshTaskAuthority, FreshTaskLease, HiddenTaskCommitment


_EVALUATOR_OPS = {"lease", "consume", "status"}
_ADMIN_OPS = _EVALUATOR_OPS | {"seal", "reveal", "close"}


def _token_text(value: str | bytes) -> str:
    if isinstance(value, bytes):
        return value.hex()
    value = str(value).strip()
    if not value:
        raise ValueError("fresh-task service token cannot be empty")
    return value


class FreshTaskAuthorityServer:
    """Authenticated Unix-domain-socket boundary for hidden evaluation tasks.

    Evaluator credentials can only lease/consume tasks. Administrative operations that
    can introduce or reveal hidden material require a different token. This prevents a
    research/evaluator process from using the same socket to bypass task secrecy.
    """

    def __init__(self, socket_path: str | Path, authority: DurableFreshTaskAuthority, *,
                 evaluator_token: str | bytes, admin_token: str | bytes):
        self.socket_path = Path(socket_path)
        self.authority = authority
        self._evaluator_token = _token_text(evaluator_token)
        self._admin_token = _token_text(admin_token)
        if secrets.compare_digest(self._evaluator_token, self._admin_token):
            raise ValueError("evaluator and admin fresh-task tokens must differ")
        self._stop = threading.Event()
        self._sock: socket.socket | None = None

    @staticmethod
    def _reply(conn: socket.socket, payload: dict[str, Any]) -> None:
        conn.sendall((json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8"))

    def _authorize(self, req: dict[str, Any], op: str) -> str:
        role = str(req.get("role") or "")
        token = str(req.get("token") or "")
        if role == "admin":
            if not secrets.compare_digest(token, self._admin_token):
                raise PermissionError("invalid fresh-task admin capability")
            if op not in _ADMIN_OPS:
                raise PermissionError("operation not permitted for fresh-task admin")
            return role
        if role == "evaluator":
            if not secrets.compare_digest(token, self._evaluator_token):
                raise PermissionError("invalid fresh-task evaluator capability")
            if op not in _EVALUATOR_OPS:
                raise PermissionError("operation requires fresh-task admin capability")
            return role
        raise PermissionError("fresh-task role/capability required")

    def _dispatch(self, req: dict[str, Any]) -> dict[str, Any]:
        op = str(req.get("op") or "")
        self._authorize(req, op)
        if op == "seal":
            obj = self.authority.seal(req.get("task"), generation=int(req["generation"]))
            return {"ok": True, "result": asdict(obj)}
        if op == "lease":
            obj = self.authority.lease(
                task_id=str(req["task_id"]),
                consumer_id=str(req["consumer_id"]),
                ttl_seconds=float(req.get("ttl_seconds", 300.0)),
            )
            return {"ok": True, "result": asdict(obj)}
        if op == "consume":
            obj = FreshTaskLease(**dict(req["lease"]))
            return {"ok": True, "result": self.authority.consume(obj)}
        if op == "reveal":
            return {"ok": True, "result": self.authority.reveal(str(req["task_id"]))}
        if op == "close":
            self.authority.close(str(req["task_id"]))
            return {"ok": True, "result": None}
        if op == "status":
            boundary = dict(self.authority.storage_boundary())
            return {
                "ok": True,
                "result": {
                    "encrypted_at_rest": bool(boundary.get("encrypted_at_rest")),
                    "service_isolation_supported": True,
                    "role_scoped_capabilities": True,
                },
            }
        raise ValueError("unsupported fresh-task service operation")

    def serve_forever(self) -> None:
        self.socket_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            if self.socket_path.exists():
                self.socket_path.unlink()
        except FileNotFoundError:
            pass
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._sock = sock
        sock.bind(str(self.socket_path))
        os.chmod(self.socket_path, 0o600)
        sock.listen(32)
        sock.settimeout(0.25)
        try:
            while not self._stop.is_set():
                try:
                    conn, _ = sock.accept()
                except socket.timeout:
                    continue
                with conn:
                    try:
                        data = b""
                        while b"\n" not in data:
                            chunk = conn.recv(65536)
                            if not chunk:
                                break
                            data += chunk
                        if not data:
                            continue
                        req = json.loads(data.split(b"\n", 1)[0].decode("utf-8"))
                        self._reply(conn, self._dispatch(req))
                    except Exception as exc:
                        try:
                            self._reply(conn, {"ok": False, "error": type(exc).__name__, "message": str(exc)})
                        except OSError:
                            pass
        finally:
            sock.close()
            self._sock = None
            try:
                self.socket_path.unlink()
            except FileNotFoundError:
                pass

    def stop(self) -> None:
        self._stop.set()
        try:
            s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            s.connect(str(self.socket_path))
            s.close()
        except OSError:
            pass


class _FreshTaskClientBase:
    def __init__(self, socket_path: str | Path, *, token: str | bytes, role: str, timeout: float = 10.0):
        self.socket_path = Path(socket_path)
        self._token = _token_text(token)
        self._role = str(role)
        self.timeout = float(timeout)

    def _call(self, payload: dict[str, Any]):
        payload = dict(payload)
        payload["role"] = self._role
        payload["token"] = self._token
        deadline = time.monotonic() + self.timeout
        data = b""
        while True:
            s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            remaining = max(0.01, deadline - time.monotonic())
            s.settimeout(remaining)
            try:
                s.connect(str(self.socket_path))
                break
            except (ConnectionRefusedError, FileNotFoundError):
                s.close()
                if time.monotonic() >= deadline:
                    raise
                time.sleep(0.01)
        try:
            s.sendall((json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8"))
            while b"\n" not in data:
                chunk = s.recv(65536)
                if not chunk:
                    break
                data += chunk
        finally:
            s.close()
        if not data:
            raise ConnectionError("fresh-task authority returned no response")
        reply = json.loads(data.split(b"\n", 1)[0].decode("utf-8"))
        if not reply.get("ok"):
            raise PermissionError(f"fresh-task service rejected request: {reply.get('error')}: {reply.get('message')}")
        return reply.get("result")

    def storage_boundary(self) -> dict:
        result = dict(self._call({"op": "status"}))
        result["transport"] = "unix-domain-socket"
        result["client_has_key_material"] = False
        result["client_has_database_path"] = False
        result["client_role"] = self._role
        return result


class FreshTaskAuthorityClient(_FreshTaskClientBase):
    """Evaluator client: can lease/consume but cannot seal/reveal/close tasks."""

    def __init__(self, socket_path: str | Path, *, token: str | bytes, timeout: float = 10.0):
        super().__init__(socket_path, token=token, role="evaluator", timeout=timeout)

    def lease(self, *, task_id: str, consumer_id: str, ttl_seconds: float = 300.0) -> FreshTaskLease:
        return FreshTaskLease(**self._call({"op": "lease", "task_id": task_id, "consumer_id": consumer_id, "ttl_seconds": float(ttl_seconds)}))

    def consume(self, lease: FreshTaskLease):
        return self._call({"op": "consume", "lease": asdict(lease)})

    # Explicit denial methods make accidental privilege assumptions fail locally as well
    # as at the daemon boundary.
    def seal(self, *args, **kwargs):
        raise PermissionError("evaluator client cannot seal hidden tasks")

    def reveal(self, *args, **kwargs):
        raise PermissionError("evaluator client cannot reveal hidden tasks")

    def close(self, *args, **kwargs):
        raise PermissionError("evaluator client cannot close hidden tasks")


class FreshTaskAuthorityAdminClient(_FreshTaskClientBase):
    """Administrative client for task provisioning and post-consumption audit reveal."""

    def __init__(self, socket_path: str | Path, *, token: str | bytes, timeout: float = 10.0):
        super().__init__(socket_path, token=token, role="admin", timeout=timeout)

    def seal(self, task, *, generation: int) -> HiddenTaskCommitment:
        return HiddenTaskCommitment(**self._call({"op": "seal", "task": task, "generation": int(generation)}))

    def lease(self, *, task_id: str, consumer_id: str, ttl_seconds: float = 300.0) -> FreshTaskLease:
        return FreshTaskLease(**self._call({"op": "lease", "task_id": task_id, "consumer_id": consumer_id, "ttl_seconds": float(ttl_seconds)}))

    def consume(self, lease: FreshTaskLease):
        return self._call({"op": "consume", "lease": asdict(lease)})

    def reveal(self, task_id: str) -> dict:
        return dict(self._call({"op": "reveal", "task_id": task_id}))

    def close(self, task_id: str) -> None:
        self._call({"op": "close", "task_id": task_id})
