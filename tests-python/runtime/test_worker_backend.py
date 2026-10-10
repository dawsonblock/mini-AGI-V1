"""v16.4.5 — worker-process backend isolation (RUN-401).

A model that ignores cooperative cancellation must be terminable
without stopping the supervisor. These tests spawn real worker
processes over the SimulatedServingBackend (deterministic: wedge,
die, latency, cancel-response modes) and drive the full
stop -> drain -> cancel -> terminate -> unload ordering through the
v16.4.4 lease machinery.
"""
import os
import pickle
import signal
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src-python"))

from egai.common.canonical import digest  # noqa: E402
from egai.common.crypto import Ed25519Signer  # noqa: E402
from minagi.runtime.authority_store import AuthorityStore  # noqa: E402
from minagi.runtime.serving_router import (  # noqa: E402
    RoutingRefused, ServingRouter)
from minagi.runtime.supervisor import (  # noqa: E402
    ServingState, ServingSupervisor)
from minagi.runtime.worker_backend import (  # noqa: E402
    BackendSpec, WorkerBackend, WorkerBackendError, WorkerDied,
    WorkerUnresponsive)
from minagi.security.admission_grants import issue_grant  # noqa: E402
from minagi.v161.artifact_closure import close_tree  # noqa: E402
from minagi.v161.authority import (AUTHORITY_ROLES, AuthorityRegistry,  # noqa: E402
                                   write_trust_root)
from minagi.v161.immutable_snapshot import (  # noqa: E402
    stage_snapshot, verify_snapshot)

NOW = datetime(2026, 10, 9, 12, 0, tzinfo=timezone.utc)
SIM_SPEC = ("minagi.runtime.simulated_backend", "SimulatedServingBackend")


def _worker(spec_kwargs=None, **kw):
    spec = BackendSpec(module=SIM_SPEC[0], qualname=SIM_SPEC[1],
                       kwargs=dict(spec_kwargs or {}))
    return WorkerBackend(spec, start_timeout=30.0, probe_timeout=10.0,
                         shutdown_grace=3.0, **kw)


def _snapshot(tmp_path, tag="m"):
    model = tmp_path / f"model-{tag}"
    if not model.exists():
        model.mkdir()
        (model / "config.json").write_text('{"model_type": "sim"}')
    adir = tmp_path / f"adapter-{tag}"
    if not adir.exists():
        adir.mkdir()
        (adir / "adapter_model.safetensors").write_bytes(b"w-" + tag.encode())
    digests = {"model": close_tree(model).digest,
               "adapter": close_tree(adir).digest}
    return stage_snapshot(
        tmp_path / "snap" / tag, {"model": str(model), "adapter": str(adir)},
        expected_digests=digests, manifest_digest=digest({"m": tag}))


def _wait_inflight(router, aid, want=1, timeout=5.0):
    deadline = time.monotonic() + timeout
    while router.inflight(aid) != want and time.monotonic() < deadline:
        time.sleep(0.01)
    return router.inflight(aid) == want


def _route_bg(router, request, results, errors):
    def hit():
        try:
            results.append(router.route(request))
        except Exception as exc:  # noqa: BLE001 - asserted by caller
            errors.append(exc)
    t = threading.Thread(target=hit, daemon=True)
    t.start()
    return t


# ---------- snapshot transport -------------------------------------------

def test_measured_snapshot_survives_pickling(tmp_path):
    """RUN-401 transport: the snapshot crossing into the worker keeps
    its measurement — and re-verification on the restored object
    still passes."""
    snap = _snapshot(tmp_path)
    clone = pickle.loads(pickle.dumps(snap))
    assert clone.manifest_digest == snap.manifest_digest
    assert clone.artifact_digests == snap.artifact_digests
    verify_snapshot(clone)


# ---------- worker lifecycle ----------------------------------------------

def test_worker_load_infer_unload(tmp_path):
    wb = _worker({"latency_seconds": 0.01})
    handle = wb.load(_snapshot(tmp_path))
    wb.health_probe(handle)
    out = wb.infer(handle, {"prompt": "hello", "request_id": "r1"})
    assert out["completion"] == "echo:hello"
    # the work ran in the child, not this process
    pid = out["metrics"]["worker_pid"]
    assert pid == handle.proc.pid != os.getpid()
    assert wb.alive(handle)
    wb.unload(handle)
    assert not wb.alive(handle)
    assert handle.proc.poll() == 0


def test_unresolvable_backend_spec_fails_fast():
    with pytest.raises(WorkerBackendError, match="resolve"):
        WorkerBackend(BackendSpec(module="minagi.runtime.simulated_backend",
                                  qualname="DoesNotExist", kwargs={}))
    with pytest.raises(WorkerBackendError):
        WorkerBackend(BackendSpec(module="minagi.no.such.module",
                                  qualname="X", kwargs={}))


def test_one_proxy_one_worker(tmp_path):
    wb = _worker()
    h = wb.load(_snapshot(tmp_path))
    with pytest.raises(WorkerBackendError, match="already loaded"):
        wb.load(_snapshot(tmp_path, "second"))
    wb.unload(h)


def test_foreign_handle_refused(tmp_path):
    wb = _worker()
    with pytest.raises(WorkerBackendError, match="foreign"):
        wb.infer({"not": "ours"}, {"prompt": "x"})


def test_remote_exception_type_survives_the_wire(tmp_path):
    """A backend-side refusal must not be flattened: PermissionError in
    the worker arrives as PermissionError in the parent."""
    wb = _worker()
    with pytest.raises(PermissionError, match="MeasuredSnapshot"):
        wb.load("not-a-snapshot")
    # the failed load reaped the worker — nothing resident is left
    assert not wb._handles


def test_broken_command_pipe_fails_without_deadlocking(tmp_path):
    wb = _worker()
    handle = wb.load(_snapshot(tmp_path))
    handle.proc.stdin.close()
    with pytest.raises(WorkerDied, match="command channel broken"):
        wb._rpc(handle, {"op": "probe"}, timeout=1, what="probe")
    wb.terminate(handle)


# ---------- preemptive termination (the RUN-401 remedy) -------------------

def test_sigkill_mid_inference_fails_lease_and_router_survives(tmp_path):
    """Worker killed mid-request: the in-flight infer raises WorkerDied,
    the lease releases, and the router is intact enough to retire."""
    router = ServingRouter(drain_timeout=0.2)
    wb = _worker({"latency_seconds": 60.0})
    handle = wb.load(_snapshot(tmp_path))
    aid = "aa" * 16
    router.activate(aid, wb, handle)
    results, errors = [], []
    t = _route_bg(router, {"prompt": "x"}, results, errors)
    assert _wait_inflight(router, aid)
    os.kill(handle.proc.pid, signal.SIGKILL)
    t.join(timeout=5)
    assert errors and isinstance(errors[0], WorkerDied)
    assert router.inflight(aid) == 0
    # a dead worker refuses new work fast — no hang on a ghost route
    with pytest.raises(WorkerDied):
        wb.infer(handle, {"prompt": "again"})
    out = router.retire(aid)
    assert out["unloaded"] is True


def test_wedged_worker_terminated_on_drain_timeout(tmp_path):
    """RUN-401: a generation that ignores cooperative cancellation is
    SIGKILLed after drain+cancel grace — the request fails, the lease
    releases, resources are reclaimed, and the router survives."""
    router = ServingRouter(drain_timeout=0.1, cancel_grace=0.1,
                           terminate_grace=5.0)
    wb = _worker({"wedge": True})
    handle = wb.load(_snapshot(tmp_path))
    aid = "aa" * 16
    router.activate(aid, wb, handle)
    results, errors = [], []
    t = _route_bg(router, {"prompt": "x"}, results, errors)
    assert _wait_inflight(router, aid)
    out = router.retire(aid)
    assert out["terminated"] is True
    assert out["drained"] is True and out["unloaded"] is True
    assert handle.proc.poll() is not None     # process really gone
    t.join(timeout=5)
    assert errors and isinstance(errors[0], RoutingRefused)


def test_terminate_frees_process_even_when_lease_lingers(tmp_path):
    """A lease held but never dispatched still blocks unload — but the
    WORKER is already dead: resources are reclaimed, only the honest
    lease accounting remains."""
    router = ServingRouter(drain_timeout=0.05, cancel_grace=0.05,
                           terminate_grace=0.5)
    wb = _worker()
    handle = wb.load(_snapshot(tmp_path))
    aid = "aa" * 16
    router.activate(aid, wb, handle)
    lease = router.acquire_lease()
    out = router.retire(aid)
    assert out["terminated"] is True
    assert out["unloaded"] is False and out["inflight"] == 1
    assert handle.proc.poll() is not None     # resources already freed
    router.release_lease(lease)
    out = router.retire(aid)
    assert out["unloaded"] is True


def test_watchdog_marks_stalled_and_unload_recovers(tmp_path):
    """A worker that stops answering is marked stalled — subsequent
    requests fail fast instead of queueing behind the wedge."""
    wb = _worker({"latency_seconds": 60.0, "respect_cancel": False},
                 request_watchdog=0.2)
    handle = wb.load(_snapshot(tmp_path))
    with pytest.raises(WorkerUnresponsive, match="no response"):
        wb.infer(handle, {"prompt": "x"})
    with pytest.raises(WorkerUnresponsive):
        wb.infer(handle, {"prompt": "y"})
    wb.unload(handle)   # graceful op first, kill as escalation
    assert handle.proc.poll() is not None


# ---------- cooperative path still works across the wire ------------------

def test_cooperative_cancel_crosses_process_boundary(tmp_path):
    """Cancel must still be honored when it is cheap: retire() asks
    politely first and terminates only what ignores it."""
    router = ServingRouter(drain_timeout=0.1, cancel_grace=3.0)
    wb = _worker({"latency_seconds": 60.0})
    handle = wb.load(_snapshot(tmp_path))
    aid = "aa" * 16
    router.activate(aid, wb, handle)
    results, errors = [], []
    t = _route_bg(router, {"prompt": "x"}, results, errors)
    assert _wait_inflight(router, aid)
    out = router.retire(aid)
    assert out["cancelled"] == 1
    assert out["terminated"] is False
    assert out["unloaded"] is True
    t.join(timeout=5)
    # the request completed early through the worker's cancel event
    assert results and results[0]["result"]["metrics"]["cancelled"]
    assert handle.proc.poll() == 0


# ---------- supervisor-level integration -----------------------------------

def _chain(tmp_path):
    storage = tmp_path / "storage"
    write_trust_root(storage / ".keys", storage / "trust_root.json")
    registry = AuthorityRegistry.load(storage / "trust_root.json")
    signers = {r: Ed25519Signer.from_private_bytes(
        (storage / ".keys" / f"{r}.pem").read_bytes())
        for r in AUTHORITY_ROLES}
    return registry, signers


def _grant(signers, artifact_root, backend="simulated"):
    return issue_grant(
        signers["admission"], decision_digest=digest({"d": 1}),
        qualification_digest=digest({"q": 1}),
        runtime_manifest_digest=digest({"m": 1}),
        artifact_root_digest=artifact_root, backend_id=backend,
        audience_runtime_identity="local-supervisor", now=NOW,
        revocation_epoch=0)


def _drive_active(sup, signers, tmp_path, backend, snapshot, root,
                  backend_id="simulated"):
    grant = _grant(signers, root, backend=backend_id)
    aid = sup.request().activation_id
    sup.authorize(aid, grant)
    sup.stage(aid, snapshot)
    sup.prepare(aid, backend)
    sup.health_check(aid)
    sup.commit_activation(aid)
    return aid


def test_quarantine_terminates_wedged_worker_and_reconciles(tmp_path):
    """SEC-302/303 + RUN-401 end to end: quarantine a wedged active
    model -> its worker is SIGKILLed -> the durable deployment is
    reconciled to UNAVAILABLE (no phantom serving model) and the
    termination is itself journal evidence."""
    registry, signers = _chain(tmp_path)
    store = AuthorityStore(tmp_path / "journal" / "authority.sqlite")
    sup = ServingSupervisor(store, runtime_signer=signers["runtime"],
                            registry=registry, now=NOW)
    sup.router = ServingRouter(drain_timeout=0.1, cancel_grace=0.1,
                               terminate_grace=5.0)
    wb = _worker({"wedge": True})
    snap = _snapshot(tmp_path)
    root = digest({n: d for n, d in snap.artifact_digests})
    aid = _drive_active(sup, signers, tmp_path, wb, snap, root)
    assert sup.serving_state is ServingState.SERVING
    handle = wb._handles[0]

    results, errors = [], []
    t = _route_bg(sup.router, {"prompt": "x"}, results, errors)
    assert _wait_inflight(sup.router, aid)

    sup.quarantine_active(reason="wedged model under inference")

    t.join(timeout=10)
    assert errors                      # the wedged request did not hang
    assert handle.proc.poll() is not None   # the worker was terminated
    assert sup.router.route_entry(aid) is None
    assert sup.router.inflight(aid) == 0
    assert sup.serving_state is ServingState.UNAVAILABLE
    pointer = sup.active_pointer()
    assert not pointer or not pointer.get("activation_id")
    kinds = [e["event_type"] for e in store.events(aid)]
    assert "backend_terminated" in kinds
    assert sup._deferred_unloads == set()   # nothing left unreleased


def test_abort_before_publish_terminates_worker(tmp_path):
    """A candidate retired before ever routing still releases its
    worker — the prepare-time unload path escalates too."""
    registry, signers = _chain(tmp_path)
    store = AuthorityStore(tmp_path / "journal" / "authority.sqlite")
    sup = ServingSupervisor(store, runtime_signer=signers["runtime"],
                            registry=registry, now=NOW)
    wb = _worker()
    snap = _snapshot(tmp_path)
    root = digest({n: d for n, d in snap.artifact_digests})
    grant = _grant(signers, root)
    aid = sup.request().activation_id
    sup.authorize(aid, grant)
    sup.stage(aid, snap)
    sup.prepare(aid, wb)
    handle = wb._handles[0]
    sup.abort(aid, reason="superseded before publish")
    assert handle.proc.poll() is not None
    assert not wb._handles


# ---------- real-model qualification ---------------------------------------

_VOCAB = ["[PAD]", "[EOS]", "[UNK]", "user:", "assistant:", "ask", "me",
          "ans", "ok", "a1", "b2", "c3", "d4", "x", "y", "hello", "world"]


def _save_real_model(model_dir: Path, adapter_dir: Path, *, seed: int):
    """A locally constructed single-layer GPT-2 + LoRA adapter, saved in
    HF format — the same artifact shape PeftServingBackend serves."""
    import torch
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import (GPT2Config, GPT2LMHeadModel,
                              PreTrainedTokenizerFast)
    import peft
    vocab = {t: i for i, t in enumerate(_VOCAB)}
    tok_obj = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
    tok_obj.pre_tokenizer = pre_tokenizers.Whitespace()
    tok = PreTrainedTokenizerFast(tokenizer_object=tok_obj,
                                  eos_token="[EOS]", pad_token="[PAD]",
                                  unk_token="[UNK]")
    cfg = GPT2Config(n_layer=1, n_head=2, n_embd=32,
                     vocab_size=len(_VOCAB), n_positions=64,
                     bos_token_id=1, eos_token_id=1, pad_token_id=0)
    torch.manual_seed(seed)
    model = GPT2LMHeadModel(cfg)
    model_dir.mkdir(parents=True)
    model.save_pretrained(model_dir)
    tok.save_pretrained(model_dir)
    peft.get_peft_model(
        model, peft.LoraConfig(r=2, lora_alpha=4,
                               target_modules=["c_attn"])
    ).save_pretrained(adapter_dir)


def test_real_model_in_worker_process_then_kill_and_quarantine(tmp_path):
    """RUN-401 with real weights: a real GPT-2+LoRA is loaded, health-
    probed and queried inside a separate process through the REAL
    PeftServingBackend, then the worker is SIGKILLed — in-flight and
    subsequent requests fail fast, the supervisor survives, and
    quarantine reconciles durable state to UNAVAILABLE."""
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    pytest.importorskip("peft")
    from minagi.runtime.inference_policy import InferenceBudgetPolicyV1

    registry, signers = _chain(tmp_path)
    store = AuthorityStore(tmp_path / "journal" / "authority.sqlite")
    sup = ServingSupervisor(store, runtime_signer=signers["runtime"],
                            registry=registry, now=NOW)
    sup.router = ServingRouter(drain_timeout=0.2, cancel_grace=0.2,
                               terminate_grace=5.0)
    budget = InferenceBudgetPolicyV1(
        max_prompt_tokens=64, max_new_tokens=16,
        execution_deadline_seconds=30.0)
    wb = WorkerBackend(
        BackendSpec(module="minagi.v161.peft_serving",
                    qualname="PeftServingBackend",
                    kwargs={"budget": budget}),
        start_timeout=120.0, probe_timeout=60.0, shutdown_grace=10.0)

    model_dir, adapter_dir = tmp_path / "rm", tmp_path / "ra"
    _save_real_model(model_dir, adapter_dir, seed=0)
    digests = {"model": close_tree(model_dir).digest,
               "adapter": close_tree(adapter_dir).digest}
    snap = stage_snapshot(
        tmp_path / "rsnap" / "a",
        {"model": str(model_dir), "adapter": str(adapter_dir)},
        expected_digests=digests, manifest_digest=digest({"m": "a"}))
    _drive_active(sup, signers, tmp_path, wb, snap,
                  digest(digests), backend_id="hf-peft")
    assert sup.serving_state is ServingState.SERVING

    # the real model answers through the process boundary
    handle = wb._handles[0]
    assert handle.proc.pid != os.getpid()
    out = sup.router.route({"prompt": "ask hello", "max_new_tokens": 4,
                            "request_id": "q1"})
    assert isinstance(out["result"]["completion"], str)
    assert out["result"]["metrics"]["prompt_tokens"] > 0

    # the model is unreachable the moment its worker dies — no hang,
    # no phantom service — and quarantine reconciles honest state
    os.kill(handle.proc.pid, signal.SIGKILL)
    t0 = time.monotonic()
    with pytest.raises(WorkerDied):
        sup.router.route({"prompt": "anyone?"})
    assert time.monotonic() - t0 < 10.0
    sup.quarantine_active(reason="worker process killed")
    assert sup.serving_state is ServingState.UNAVAILABLE
    pointer = sup.active_pointer()
    assert not pointer or not pointer.get("activation_id")
    assert sup.router.inflight() == 0
    assert sup._deferred_unloads == set()
