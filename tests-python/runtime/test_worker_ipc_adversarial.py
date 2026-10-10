"""v16.4.5 worker IPC adversarial tests (SEC-401 / G2).

A compromised worker controls every byte it sends. The supervisor
must fail closed on ANY protocol violation — and must itself survive:
pending requests fail fast, the worker is terminable, and nothing the
worker emitted is ever deserialized into objects or code.
"""
import io
import json
import struct
import sys
import threading
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src-python"))
sys.path.insert(0, str(ROOT / "tests-python" / "helpers"))

from minagi.runtime.worker_backend import (  # noqa: E402
    BackendSpec, WorkerBackend, WorkerBackendError, WorkerDied,
    WorkerProtocolViolation, WorkerUnresponsive, _WorkerHandle)
from minagi.runtime.worker_protocol import (  # noqa: E402
    MAX_FRAME_BYTES, MAX_PENDING_REQUESTS, PROTOCOL)


def _frame_bytes(payload: bytes) -> bytes:
    return struct.pack(">I", len(payload)) + payload


def _json_frame(obj) -> bytes:
    return _frame_bytes(json.dumps(obj).encode())


class _FakeProc:
    """A worker stand-in exposing the proc surface _reader/_rpc use.
    With ``hold_open`` the reply channel is a pipe whose write end is
    kept open — the worker appears alive but silent (no EOF)."""
    def __init__(self, incoming: bytes = b"", *, hold_open=False):
        import os
        self.stdin = io.BytesIO()
        if hold_open:
            r, self._w = os.pipe()
            self.stdout = os.fdopen(r, "rb")
        else:
            self._w = None
            self.stdout = io.BytesIO(incoming)
        self.pid = 4242
        self._exit = None

    def poll(self):
        return self._exit

    def wait(self, timeout=None):
        return 0 if self._exit is not None else 0

    def kill(self):
        self._exit = -9
        if self._w is not None:
            try:
                import os
                os.close(self._w)
            except OSError:
                pass
            self._w = None


def _backend() -> WorkerBackend:
    return WorkerBackend(BackendSpec(
        module="minagi.v161.peft_serving",
        qualname="PeftServingBackend", kwargs={}))


def _handle(backend, incoming: bytes, *, hold_open=False
            ) -> _WorkerHandle:
    h = _WorkerHandle(owner=backend,
                      proc=_FakeProc(incoming, hold_open=hold_open))
    h.reader = threading.Thread(
        target=backend._reader, args=(h,), daemon=True)
    h.reader.start()
    return h


def _await_dead(h, timeout=10.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if h.dead is not None:
            return h.dead
        time.sleep(0.01)
    raise AssertionError("worker handle was never marked dead")


# ---------- hostile worker bytes -------------------------------------------

def test_pickle_payload_tears_down_channel():
    """The canonical v16.4.4 exploit: pickle bytes on the reply
    channel. They are refused as malformed frames — never handed to
    any deserializer — and the pending request fails."""
    import pickle
    backend = _backend()
    h = _handle(backend, _frame_bytes(
        pickle.dumps({"evil": "object"})))
    with pytest.raises(WorkerProtocolViolation):
        backend._rpc(h, {"op": "probe"}, timeout=10, what="probe")


def test_giant_frame_announcement_refused():
    backend = _backend()
    h = _handle(backend, struct.pack(">I", MAX_FRAME_BYTES + 1))
    with pytest.raises(WorkerProtocolViolation):
        backend._rpc(h, {"op": "probe"}, timeout=10, what="probe")


def test_unknown_reply_type_violates():
    backend = _backend()
    h = _handle(backend, _json_frame({
        "protocol": PROTOCOL, "request_id": "r1",
        "type": "exec_request", "payload": {"code": "x()"}}))
    with pytest.raises(WorkerProtocolViolation):
        backend._rpc(h, {"op": "probe"}, timeout=10, what="probe")


def test_worker_eof_mid_response_fails_pending_not_supervisor():
    """Channel ends mid-frame: pending request fails with WorkerDied;
    the supervisor object is fully usable afterwards."""
    backend = _backend()
    good = _json_frame({"protocol": PROTOCOL, "request_id": "r1",
                        "type": "result", "ok": True,
                        "payload": {"result": {"x": 1}}})
    h = _handle(backend, good[:-20])      # truncated mid-payload
    with pytest.raises(WorkerDied):
        backend._rpc(h, {"op": "probe"}, timeout=10, what="probe")
    # supervisor still governs the handle
    backend.terminate(h)
    assert not backend.alive(h)


def test_reply_for_unknown_request_id_is_ignored_then_eof():
    backend = _backend()
    h = _handle(backend, _json_frame({
        "protocol": PROTOCOL, "request_id": "r999",
        "type": "result", "ok": True, "payload": {"result": 1}}))
    with pytest.raises(WorkerDied):
        backend._rpc(h, {"op": "probe"}, timeout=10, what="probe")


def test_pending_request_bound():
    backend = _backend()
    h = _handle(backend, b"", hold_open=True)  # alive but never replies
    p = backend
    for i in range(MAX_PENDING_REQUESTS):
        threading.Thread(
            target=lambda: _swallow(
                lambda: p._rpc(h, {"op": "probe"}, timeout=30,
                               what="probe")), daemon=True).start()
    deadline = time.monotonic() + 10
    while len(h.pending) < MAX_PENDING_REQUESTS:
        if time.monotonic() > deadline:
            pytest.fail("pending map never filled")
        time.sleep(0.01)
    with pytest.raises(WorkerBackendError, match="outstanding"):
        p._rpc(h, {"op": "probe"}, timeout=1, what="probe")


def _swallow(fn):
    try:
        fn()
    except Exception:  # noqa: BLE001
        pass


def test_stalled_infer_marks_handle_unusable():
    """A worker that never answers inference marks the handle stalled;
    subsequent work refuses fast rather than queueing behind a wedge."""
    backend = _backend()
    h = _handle(backend, b"", hold_open=True)
    with pytest.raises(WorkerUnresponsive):
        backend._rpc(h, {"op": "infer", "request": {}},
                     timeout=0.2, what="inference")
    assert h.stalled
    with pytest.raises(WorkerUnresponsive):
        backend.infer(h, {"prompt": "x"})


# ---------- real spawned hostile worker -------------------------------------

def test_spawned_attacker_cannot_inject_objects(tmp_path):
    """A real worker process whose backend writes hostile bytes to
    every fd it owns: the supervisor refuses the channel, fails the
    load, and terminates the process — nothing is deserialized."""
    spec = BackendSpec(module="evil_backends",
                       qualname="ProtocolAttacker", kwargs={})
    backend = WorkerBackend(spec, start_timeout=30.0)
    from minagi.v161.immutable_snapshot import stage_snapshot
    from minagi.v161.artifact_closure import close_tree
    adir = tmp_path / "ad"
    adir.mkdir()
    (adir / "adapter_config.json").write_text("{}")
    (adir / "w.bin").write_bytes(b"w")
    snap = stage_snapshot(
        tmp_path / "snap" / "s", {"adapter": str(adir)},
        expected_digests={"adapter": close_tree(adir).digest},
        manifest_digest="sha256:" + "0" * 64)
    with pytest.raises((WorkerDied, WorkerProtocolViolation,
                        WorkerBackendError)):
        backend.load(snap)


def test_spawned_silent_worker_killed_by_deadline(tmp_path):
    """A worker that wedges on load is killed at the startup deadline —
    the supervisor does not wait forever."""
    spec = BackendSpec(module="evil_backends",
                       qualname="SilentBackend", kwargs={})
    backend = WorkerBackend(spec, start_timeout=1.0)
    from minagi.v161.immutable_snapshot import stage_snapshot
    from minagi.v161.artifact_closure import close_tree
    adir = tmp_path / "ad"
    adir.mkdir()
    (adir / "adapter_config.json").write_text("{}")
    (adir / "w.bin").write_bytes(b"w")
    snap = stage_snapshot(
        tmp_path / "snap" / "s", {"adapter": str(adir)},
        expected_digests={"adapter": close_tree(adir).digest},
        manifest_digest="sha256:" + "0" * 64)
    started = time.monotonic()
    with pytest.raises((WorkerUnresponsive, WorkerDied,
                        WorkerBackendError)):
        backend.load(snap)
    assert time.monotonic() - started < 30
