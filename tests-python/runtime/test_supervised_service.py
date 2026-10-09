"""v16.4.3 — the supervised launch service is an OS-enforced boundary.

The service owns the runtime signing keys, the authority store, and the
routing table; clients submit documents over the authenticated socket.
Peer-credential checks fail closed on platforms that cannot prove a uid;
operation authorization derives from the authenticated uid, never the
request body (SEC-203); request size, connection count, and I/O are
bounded (OPS-001).
"""
import os
import socket
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src-python"))

from minagi.runtime.authority_store import AuthorityStore  # noqa: E402
from minagi.runtime.service import (  # noqa: E402
    SupervisorService, peer_uid)
from minagi.runtime.supervisor import ServingSupervisor  # noqa: E402
from minagi.security.trusted_authority_client import (  # noqa: E402
    RequestRefused, ServiceUnavailable, SupervisorClient)
from egai.common.crypto import Ed25519Signer  # noqa: E402
from minagi.v161.authority import (AuthorityRegistry,  # noqa: E402
                                   write_trust_root)

NOW = datetime(2026, 10, 8, 12, 0, tzinfo=timezone.utc)


def _service(tmp_path, *, uid_roles=None, insecure=False):
    storage = tmp_path / "storage"
    write_trust_root(storage / ".keys", storage / "trust_root.json")
    registry = AuthorityRegistry.load(storage / "trust_root.json")
    signer = Ed25519Signer.from_private_bytes(
        (storage / ".keys" / "runtime.pem").read_bytes())
    supervisor = ServingSupervisor(
        AuthorityStore(tmp_path / "journal" / "authority.sqlite"),
        runtime_signer=signer, registry=registry, now=NOW)
    import secrets
    # AF_UNIX path length is limited (~104 chars on macOS) — keep the
    # socket outside the deep pytest tmp tree
    sock = Path("/tmp") / f"miniagi-svc-{secrets.token_hex(6)}.sock"
    roles = uid_roles if uid_roles is not None else \
        {os.getuid(): "operator"}
    svc = SupervisorService(
        sock, launcher=None, supervisor=supervisor,
        backend_factories={}, uid_roles=roles,
        audit_signer=signer, allow_insecure_dev=insecure)
    thread = threading.Thread(target=svc.serve_forever, daemon=True)
    thread.start()
    # the socket file appears at bind(); wait until listen() is live so
    # a connect cannot race it and refuse
    for _ in range(200):
        if not sock.exists():
            time.sleep(0.01)
            continue
        try:
            probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            probe.connect(str(sock))
            probe.close()
            break
        except OSError:
            time.sleep(0.01)
    return svc, sock


def test_peer_uid_extractable_on_socketpair():
    a, b = socket.socketpair()
    try:
        uid = peer_uid(a)
        if uid is not None:  # platform supports it
            assert uid == os.getuid()
    finally:
        a.close()
        b.close()


def test_ping_and_status_over_socket(tmp_path):
    # no peer-credential mechanism on this platform — dev mode
    svc, sock = _service(tmp_path, insecure=True)
    try:
        client = SupervisorClient(sock)
        resp = client.ping()
        assert resp["ok"] and resp["service"] == "supervised-launch"
        status = client.status()
        assert status["ok"] and "serving_state" in status
    finally:
        svc.stop()


def test_unlisted_peer_uid_refused(tmp_path):
    """A client running under a non-allowlisted uid is refused at the
    credential check — the refusal arrives as a refused frame (or a
    closed connection where the platform cannot authenticate peers);
    documents never reach request handling."""
    svc, sock = _service(tmp_path, uid_roles={0: "operator"})
    try:
        client = SupervisorClient(sock)
        if os.getuid() == 0:
            pytest.skip("running as the allowlisted uid")
        with pytest.raises((RequestRefused, ServiceUnavailable)):
            client._call({"op": "ping"})
    finally:
        svc.stop()


def test_unknown_op_refused_cleanly(tmp_path):
    svc, sock = _service(tmp_path, insecure=True)
    try:
        client = SupervisorClient(sock)
        with pytest.raises(RequestRefused,
                           match="unknown operation|not permitted"):
            client._call({"op": "exec", "code": "import os"})
    finally:
        svc.stop()


def test_research_principal_cannot_quarantine(tmp_path):
    """SEC-203: a research principal may propose launches but cannot
    drive protected lifecycle operations — the dispatcher enforces the
    uid-derived role, not a client-supplied claim."""
    svc, _sock = _service(tmp_path, insecure=True)
    try:
        from minagi.runtime.access_policy import PrincipalContext
        from minagi.runtime.service import ServiceRefused
        research = PrincipalContext(uid=1000, role="research",
                                    principal_id="uid:1000")
        operator = PrincipalContext(uid=1001, role="operator",
                                    principal_id="uid:1001")
        anon = PrincipalContext(uid=-1, role="unauthenticated",
                                principal_id="uid:-1")
        for op in ("quarantine", "rollback", "recover"):
            with pytest.raises(ServiceRefused, match="not permitted"):
                svc._handle_request({"op": op}, research)
            with pytest.raises(ServiceRefused, match="not permitted"):
                svc._handle_request({"op": op}, anon)
        # a client-supplied "role" field changes nothing
        with pytest.raises(ServiceRefused, match="not permitted"):
            svc._handle_request({"op": "quarantine",
                                 "role": "operator"}, research)
        resp = svc._handle_request({"op": "ping"}, research)
        assert resp["ok"]
        resp = svc._handle_request({"op": "recover"}, operator)
        assert resp["ok"]
    finally:
        svc.stop()


def test_launch_without_registered_factory_refused(tmp_path):
    """Clients name backends; they never supply loader objects — an
    unregistered backend name is refused."""
    svc, sock = _service(tmp_path, insecure=True)
    try:
        client = SupervisorClient(sock)
        with pytest.raises(RequestRefused, match="no service-registered"):
            client.launch({"campaign_id": "c", "seed": "s"},
                          backend="attacker-loader")
    finally:
        svc.stop()


def test_recover_op_reconciles_store(tmp_path):
    svc, sock = _service(tmp_path, insecure=True)
    try:
        client = SupervisorClient(sock)
        resp = client.recover()
        assert resp["ok"] and "report" in resp
        assert resp["report"]["serving_state"] == "UNAVAILABLE"
    finally:
        svc.stop()


def test_oversized_request_refused(tmp_path):
    """A request line beyond the size bound is refused before parsing
    (OPS-001): the bounded read never buffers the full payload."""
    svc, sock = _service(tmp_path, insecure=True)
    try:
        conn = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        conn.connect(str(sock))
        try:
            conn.sendall(b'{"op": "ping", "pad": "'
                         + b"x" * ((1 << 22) + 8))
            conn.sendall(b'"\n')
        except BrokenPipeError:
            pass  # server closed after the bounded read — correct
        time.sleep(0.1)
        try:
            resp = conn.recv(4096)
        except OSError:
            resp = b""
        conn.close()
        assert not resp or b'"ok": false' in resp.lower() or \
            b'"ok":false' in resp.lower()
    finally:
        svc.stop()


def test_empty_request_refused(tmp_path):
    svc, sock = _service(tmp_path, insecure=True)
    try:
        conn = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        conn.connect(str(sock))
        conn.sendall(b"\n")
        resp = conn.recv(4096)
        conn.close()
        assert b"refused" in resp or b'"ok":false' in resp or not resp
    finally:
        svc.stop()
