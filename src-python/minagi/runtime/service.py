"""v16.4.2 supervised launch service (UPGRADE_PLAN §3.1/§3.2).

A signed Python object is not a security boundary against malicious
code in the same interpreter. This module wraps `ServingSupervisor` +
`TrustedRuntimeLauncher` in a dedicated service process that owns the
runtime signing keys, the snapshot root, and the durable journal, and
speaks to untrusted callers over an authenticated Unix socket:

  * requests and responses are newline-delimited JSON;
  * every connection is peer-credential checked (SO_PEERCRED on Linux,
    `socket.getpeereid` on macOS/BSD) against an explicit allowed-uid
    policy — the research plane may submit *documents*, but it never
    receives a signing capability, a backend object, or a writable
    path into the protected artifact store;
  * the backend is constructed INSIDE the service from its own
    registry of factories — a client cannot inject a loader object;
  * platforms without peer-credential extraction fail closed
    (`--allow-insecure-dev` exists only for development and never in
    the production path).

Deployment topology: run the service under a dedicated OS identity
with exclusive ownership of `.keys/`, `snapshots/`, and the journal;
grant the research uid an entry in `--allowed-uid`. Same-uid operation
is supported for tests/development but confers no OS boundary.
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import struct
import sys
import threading
from pathlib import Path


class ServiceRefused(PermissionError):
    """The service refused a connection or request."""


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
                 backend_factories: dict, allowed_peer_uids,
                 allow_insecure_dev: bool = False):
        self.socket_path = Path(socket_path)
        self.launcher = launcher
        self.supervisor = supervisor
        self.backend_factories = dict(backend_factories)
        self.allowed = {int(u) for u in allowed_peer_uids}
        self.allow_insecure_dev = bool(allow_insecure_dev)
        self._sock: socket.socket | None = None
        self._stop = threading.Event()

    # --- request handling -------------------------------------------
    def _handle_request(self, req: dict) -> dict:
        if not isinstance(req, dict):
            raise ServiceRefused("request must be a JSON object")
        op = req.get("op")
        if op == "ping":
            return {"ok": True, "service": "supervised-launch",
                    "runtime_identity": self.supervisor.runtime_identity}
        if op == "status":
            return {"ok": True,
                    "active": self.supervisor.active_pointer()}
        if op == "recover":
            return {"ok": True,
                    "report": self.supervisor.recover_from_journal()}
        if op == "quarantine":
            self.supervisor.quarantine_active(
                reason=str(req.get("reason") or "operator quarantine"))
            return {"ok": True,
                    "active": self.supervisor.active_pointer()}
        if op == "rollback":
            try:
                target = self.supervisor.rollback()
            except Exception as exc:  # noqa: BLE001
                raise ServiceRefused(f"rollback: {exc}") from exc
            return {"ok": True, "active": target}
        if op == "launch":
            return self._launch(req)
        raise ServiceRefused(f"unknown op {op!r}")

    def _launch(self, req: dict) -> dict:
        from minagi.v161.trusted_launcher import LaunchRequest
        spec = str(req.get("backend") or "hf-peft")
        factory = self.backend_factories.get(spec)
        if factory is None:
            raise ServiceRefused(
                f"no service-registered backend factory for {spec!r} — "
                "clients name backends, they do not supply them")
        body = dict(req.get("request") or {})
        body["expected_backend"] = spec
        request = LaunchRequest(**body)
        backend = factory()
        result = self.launcher.launch(request, backend,
                                      receipt_path=req.get(
                                          "receipt_path"))
        return {"ok": True, "backend_id": result.backend_id,
                "receipt_digest": result.receipt_doc["digest"],
                "activation_id": getattr(result, "activation_id", ""),
                "loaded_artifact_digests":
                    dict(result.receipt.loaded_artifact_digests),
                "activation_nonce": result.receipt.activation_nonce}

    # --- transport --------------------------------------------------
    def _serve_conn(self, conn: socket.socket) -> None:
        try:
            try:
                uid = peer_uid(conn)
                if uid is None:
                    if not self.allow_insecure_dev:
                        raise ServiceRefused(
                            "peer credentials unavailable on this "
                            "platform — connections cannot be "
                            "authenticated")
                elif uid not in self.allowed:
                    raise ServiceRefused(
                        f"peer uid {uid} is not an allowed runtime "
                        "client")
                f = conn.makefile("rwb")
                line = f.readline()
                if len(line) > 1 << 22:
                    raise ServiceRefused(
                        "request exceeds the size bound")
                try:
                    req = json.loads(line)
                except ValueError as exc:
                    raise ServiceRefused(f"request is not JSON: {exc}")
                try:
                    resp = self._handle_request(req)
                except ServiceRefused as exc:
                    resp = {"ok": False, "refused": str(exc)}
                except Exception as exc:  # noqa: BLE001 - hygiene
                    resp = {"ok": False,
                            "error": f"{type(exc).__name__}: {exc}"}
            except ServiceRefused as exc:
                resp = {"ok": False, "refused": str(exc)}
                f = conn.makefile("rwb")
            f.write((json.dumps(resp) + "\n").encode())
            f.flush()
        except Exception:  # noqa: BLE001 - a bad conn must not kill svc
            pass
        finally:
            conn.close()

    def serve_forever(self) -> None:
        self.socket_path.parent.mkdir(parents=True, exist_ok=True)
        if self.socket_path.exists():
            self.socket_path.unlink()
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.bind(str(self.socket_path))
        os.chmod(self.socket_path, 0o600)
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
                threading.Thread(target=self._serve_conn,
                                 args=(conn,), daemon=True).start()
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


def main(argv=None) -> int:
    """Run the supervised launch service.

        python -m minagi.runtime.service \
            --socket /run/minagi/supervisor.sock \
            --storage-root STORAGE --allowed-uid 0 --allowed-uid 1000
    """
    ap = argparse.ArgumentParser(prog="minagi.runtime.service")
    ap.add_argument("--socket", required=True)
    ap.add_argument("--storage-root", required=True)
    ap.add_argument("--allowed-uid", action="append", type=int,
                    default=[], help="peer uid permitted to connect "
                                     "(repeatable)")
    ap.add_argument("--journal-dir", default=None)
    ap.add_argument("--allow-insecure-dev", action="store_true",
                    help="development only: serve when peer credentials "
                         "cannot be extracted")
    args = ap.parse_args(argv)

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from egai.common.crypto import Ed25519Signer
    from minagi.runtime.supervisor import ServingSupervisor
    from minagi.v161.authority import AuthorityRegistry
    from minagi.v161.peft_serving import PeftServingBackend
    from minagi.v161.trusted_launcher import TrustedRuntimeLauncher
    from minagi.security.signed_revocations import RevocationStore

    storage = Path(args.storage_root).resolve()
    registry = AuthorityRegistry.load(storage / "trust_root.json")
    runtime_key = Ed25519Signer.from_private_bytes(
        (storage / ".keys" / "runtime.pem").read_bytes())
    admission_key = Ed25519Signer.from_private_bytes(
        (storage / ".keys" / "admission.pem").read_bytes())

    supervisor = ServingSupervisor(
        args.journal_dir or (storage / "runtime_journal"),
        runtime_signer=runtime_key, registry=registry)
    supervisor.recover_from_journal()

    launcher = TrustedRuntimeLauncher(
        registry,
        revocation_store=RevocationStore(storage / "revocations"),
        runtime_signer=runtime_key, admission_signer=admission_key,
        supervisor=supervisor,
        snapshot_root=storage / "snapshots",
        nonce_journal=storage / "activation_nonces.jsonl",
        ledger_path=storage / "AUTHORITY_LEDGER.jsonl")

    service = SupervisorService(
        args.socket, launcher=launcher, supervisor=supervisor,
        backend_factories={"hf-peft": PeftServingBackend},
        allowed_peer_uids=args.allowed_uid or [os.getuid()],
        allow_insecure_dev=args.allow_insecure_dev)
    print(f"[supervised-launch] listening on {args.socket} "
          f"(allowed uids: {sorted(service.allowed)})", flush=True)
    try:
        service.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
