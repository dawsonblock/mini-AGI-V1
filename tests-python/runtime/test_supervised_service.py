"""v16.4.2 — the supervised launch service is an OS-enforced boundary.

The service owns the runtime signing keys and the journal; clients can
only submit documents over the authenticated socket. Peer-credential
checks fail closed on platforms that cannot prove a uid.
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

from minagi.runtime.service import (  # noqa: E402
    SupervisorService, peer_uid)
from minagi.runtime.supervisor import ServingSupervisor  # noqa: E402
from minagi.security.trusted_authority_client import (  # noqa: E402
    RequestRefused, ServiceUnavailable, SupervisorClient)
from egai.common.crypto import Ed25519Signer  # noqa: E402
from minagi.v161.authority import (AuthorityRegistry,  # noqa: E402
                                   write_trust_root)

NOW = datetime(2026, 10, 8, 12, 0, tzinfo=timezone.utc)


def _service(tmp_path, *, allowed=None, insecure=False):
    storage = tmp_path / "storage"
    write_trust_root(storage / ".keys", storage / "trust_root.json")
    registry = AuthorityRegistry.load(storage / "trust_root.json")
    signer = Ed25519Signer.from_private_bytes(
        (storage / ".keys" / "runtime.pem").read_bytes())
    supervisor = ServingSupervisor(
        tmp_path / "journal", runtime_signer=signer, registry=registry,
        now=NOW)
    import secrets
    # AF_UNIX path length is limited (~104 chars on macOS) — keep the
    # socket outside the deep pytest tmp tree
    sock = Path("/tmp") / f"miniagi-svc-{secrets.token_hex(6)}.sock"
    svc = SupervisorService(
        sock, launcher=None, supervisor=supervisor,
        backend_factories={}, allowed_peer_uids=(
            [os.getuid()] if allowed is None else allowed),
        allow_insecure_dev=insecure)
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
        assert status["ok"] and status["active"] is None
    finally:
        svc.stop()


def test_unlisted_peer_uid_refused(tmp_path):
    """A client running under a non-allowlisted uid is refused at the
    credential check — the refusal arrives as a refused frame (or a
    closed connection where the platform cannot authenticate peers);
    documents never reach request handling."""
    svc, sock = _service(tmp_path, allowed=[0])
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
        with pytest.raises(RequestRefused, match="unknown op"):
            client._call({"op": "exec", "code": "import os"})
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


def test_recover_op_reconciles_journal(tmp_path):
    svc, sock = _service(tmp_path, insecure=True)
    try:
        client = SupervisorClient(sock)
        resp = client.recover()
        assert resp["ok"] and "report" in resp
    finally:
        svc.stop()


def test_oversized_request_refused(tmp_path):
    """A request line beyond the size bound is refused before parsing."""
    svc, sock = _service(tmp_path, insecure=True)
    try:
        conn = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        conn.connect(str(sock))
        conn.sendall(b'{"op": "ping", "pad": "' + b"x" * ((1 << 22) + 8))
        conn.sendall(b'"\n')
        time.sleep(0.2)
        resp = conn.recv(4096)
        conn.close()
        # refused conns are closed without a response or with an error
        assert not resp or b'"ok": false' in resp.lower() or not resp
    finally:
        svc.stop()
