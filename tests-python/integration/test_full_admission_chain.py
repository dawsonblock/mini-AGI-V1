"""v16.4.3 — end-to-end: proposal -> supervised activation -> routed
inference -> rollback, through the authenticated service socket
(OPS-003 minimum end-to-end test, with the fake backend standing in for
the real HF/PEFT model — real-model qualification is a v16.5 gate).
"""
import os
import socket
import sys
import threading
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src-python"))
sys.path.insert(0, str(ROOT / "tests-python" / "v161"))

from minagi.runtime.authority_store import AuthorityStore  # noqa: E402
from minagi.runtime.service import SupervisorService  # noqa: E402
from minagi.runtime.serving_router import ServingRouter  # noqa: E402
from minagi.runtime.supervisor import ServingSupervisor  # noqa: E402
from minagi.security.signed_revocations import RevocationStore  # noqa: E402
from minagi.security.trusted_authority_client import (  # noqa: E402
    RequestRefused, SupervisorClient)
from minagi.v161.trusted_launcher import (  # noqa: E402
    TrustedRuntimeLauncher)

from test_v1641_trusted_launcher import (  # noqa: E402
    FakeBackend, _build_chain, _revocation_snapshot, _request, NOW)


class InferBackend(FakeBackend):
    """A serving backend that answers inference on its handle."""

    def infer(self, handle, request):
        return {"completion": f"{request.get('prompt', '')}!",
                "handle": str(handle.get("loaded_from", ""))[:24]}


def _service(tmp_path, *, insecure=True):
    chain = _build_chain(tmp_path)
    store = AuthorityStore(tmp_path / "state" / "authority.sqlite")
    supervisor = ServingSupervisor(
        store, runtime_signer=chain.signers["runtime"],
        registry=chain.registry, now=NOW)
    router = ServingRouter()
    supervisor.router = router
    rev_dir = tmp_path / "revocations"
    RevocationStore(rev_dir).publish(_revocation_snapshot(chain))
    launcher = TrustedRuntimeLauncher(
        chain.registry,
        revocation_store=RevocationStore(rev_dir),
        runtime_signer=chain.signers["runtime"],
        admission_signer=chain.signers["admission"],
        supervisor=supervisor, authority_store=store,
        snapshot_root=tmp_path / "snapshots",
        receipts_dir=tmp_path / "receipts", now=NOW)
    import secrets
    sock = Path("/tmp") / f"miniagi-e2e-{secrets.token_hex(6)}.sock"
    svc = SupervisorService(
        sock, launcher=launcher, supervisor=supervisor,
        backend_factories={"hf-peft": InferBackend},
        uid_roles={os.getuid(): "operator"},
        router=router, audit_signer=chain.signers["runtime"],
        allow_insecure_dev=insecure)
    t = threading.Thread(target=svc.serve_forever, daemon=True)
    t.start()
    for _ in range(200):
        if not sock.exists():
            time.sleep(0.01)
            continue
        try:
            p = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            p.connect(str(sock))
            p.close()
            break
        except OSError:
            time.sleep(0.01)
    return svc, sock, chain


def _request_fields(chain):
    req = _request(chain)
    return {"campaign_id": req.campaign_id, "seed": req.seed,
            "decision_doc": req.decision_doc,
            "qualification_doc": req.qualification_doc,
            "plan_doc": req.plan_doc,
            "runtime_manifest": req.runtime_manifest,
            "adapter_dir": req.adapter_dir,
            "model_path": req.model_path,
            "tokenizer_path": req.tokenizer_path}


def test_launch_infer_and_route_through_socket(tmp_path):
    """The same admission authority that gates activation also gates
    inference — a committed activation becomes routable, and route()
    reaches the live handle."""
    svc, sock, chain = _service(tmp_path)
    try:
        client = SupervisorClient(sock)
        out = client.launch(_request_fields(chain),
                            request_id="req-e2e-1")
        assert out["ok"] and out["activation_id"]
        assert len(out["activation_id"]) == 32   # server-generated
        status = client.status()
        assert status["serving_state"] == "SERVING"
        assert status["router"]["serving"] is True
        assert status["router"]["activation_id"] == out["activation_id"]
        infer = client.infer({"prompt": "probe"})
        assert infer["ok"]
        assert infer["activation_id"] == out["activation_id"]

        # idempotent retry: same request id returns the recorded
        # outcome — no second activation, no second grant
        again = client.launch(_request_fields(chain),
                              request_id="req-e2e-1")
        assert again["ok"] and again.get("replayed") is True
        assert again["activation_id"] == out["activation_id"]
    finally:
        svc.stop()


def test_unknown_backend_name_refused_through_socket(tmp_path):
    svc, sock, chain = _service(tmp_path)
    try:
        client = SupervisorClient(sock)
        with pytest.raises(RequestRefused,
                           match="no service-registered"):
            client.launch(_request_fields(chain),
                          backend="qwen-native-cuda")
    finally:
        svc.stop()


def test_infer_refused_when_nothing_serving(tmp_path):
    svc, sock, chain = _service(tmp_path)
    try:
        client = SupervisorClient(sock)
        with pytest.raises(RequestRefused, match="unavailable|no live"):
            client.infer({"prompt": "x"})
    finally:
        svc.stop()
