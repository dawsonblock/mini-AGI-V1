"""v16.4.9 — G10 real-model lifecycle INSIDE the NsJail boundary.

The audit's remaining qualification gap: an actual HF/PEFT model
load → health probe → inference → cancellation → unload → verified
teardown, with the worker confined by systemd cgroup v2 + NsJail.

Linux-only: skips honestly anywhere else. Requires the real stack
(torch, transformers, peft, tokenizers) on the host — the jail sees
the same site-packages through the broker's read-only runtime
closure, so the worker runs the REAL model code, not a fixture.
"""
from __future__ import annotations

import hashlib
import os
import shutil
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    sys.platform != "linux",
    reason="NsJail/systemd boundary exists only on Linux")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src-python"))
sys.path.insert(0, str(ROOT / "tests-python" / "helpers"))

PROFILE_PATH = ROOT / "configs" / "isolation" / \
    "linux-production-offline-v1.json"


def _nsjail():
    for cand in (os.environ.get("MINAGI_NSJAIL"), "/opt/nsjail/nsjail",
                 shutil.which("nsjail")):
        if cand and Path(cand).is_file():
            return cand, "sha256:" + hashlib.sha256(
                Path(cand).read_bytes()).hexdigest()
    return None




def _traversable(path):
    """The userns-mapped worker uid must traverse every ancestor of
    the bind-mounted scratch — chmod o+x up the chain (traverse-only,
    listing still denied)."""
    p = Path(path).resolve()
    for d in (p, *p.parents):
        try:
            os.chmod(d, os.stat(d).st_mode | 0o001)
        except OSError:
            break
        if d.parent == d or d.name == "tmp":
            break

def _executor(tmp_path):
    _traversable(tmp_path)
    from minagi.runtime.isolation_profile import load_profile
    from minagi.runtime.nsjail_executor import NsJailWorkerExecutor
    found = _nsjail()
    if found is None:
        pytest.skip("nsjail binary not found — set MINAGI_NSJAIL")
    if not (shutil.which("systemd-run") and shutil.which("systemctl")):
        pytest.skip("systemd-run/systemctl not available")
    if not Path("/sys/fs/cgroup/cgroup.controllers").is_file():
        pytest.skip("host is not cgroup v2 (unified hierarchy)")
    binary, digest = found
    euid = os.geteuid()
    wuid = int(os.environ.get("MINAGI_WORKER_UID")
               or (65534 if euid == 0 else euid))
    if wuid == euid and euid != 0:
        pytest.skip("set MINAGI_WORKER_UID to a uid distinct from "
                    "the supervisor")
    ex = NsJailWorkerExecutor.from_profile(
        load_profile(PROFILE_PATH),
        worker_state_root=tmp_path / "workers",
        nsjail_path=binary, nsjail_digest=digest,
        worker_uid=wuid,
        worker_gid=int(os.environ.get("MINAGI_WORKER_GID") or wuid),
        user_manager=euid != 0,
        python_path_entries=tuple(str(p) for p in sys.path if p))
    report = ex.validate_host()
    if not report.get("ok"):
        pytest.skip(f"host cannot establish the nsjail boundary: "
                    f"{report}")
    return ex


def _real_model(model_dir: Path, adapter_dir: Path, *, seed: int):
    """A REAL GPT2 + LoRA — tiny (1 layer / 32 embd / 16 tokens) but
    exercised through the genuine transformers/peft load path."""
    import torch
    import peft
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import (GPT2Config, GPT2LMHeadModel,
                              PreTrainedTokenizerFast)
    vocab = ["[PAD]", "[EOS]", "[UNK]", "user:", "assistant:", "ask",
             "me", "ans", "ok", "a1", "b2", "c3", "d4", "x", "y",
             "hello", "world"]
    tok_obj = Tokenizer(models.WordLevel(
        vocab={t: i for i, t in enumerate(vocab)}, unk_token="[UNK]"))
    tok_obj.pre_tokenizer = pre_tokenizers.Whitespace()
    tok = PreTrainedTokenizerFast(tokenizer_object=tok_obj,
                                  eos_token="[EOS]", pad_token="[PAD]",
                                  unk_token="[UNK]")
    cfg = GPT2Config(n_layer=1, n_head=2, n_embd=32,
                     vocab_size=len(vocab), n_positions=64,
                     bos_token_id=1, eos_token_id=1, pad_token_id=0)
    torch.manual_seed(seed)
    model = GPT2LMHeadModel(cfg)
    model_dir.mkdir(parents=True)
    model.save_pretrained(model_dir)
    tok.save_pretrained(model_dir)
    served = peft.get_peft_model(
        model, peft.LoraConfig(r=2, lora_alpha=4,
                               target_modules=["c_attn"]))
    served.save_pretrained(adapter_dir)


def _snapshot(tmp_path, tag, seed):
    from egai.common.canonical import digest
    from minagi.v161.artifact_closure import close_tree
    from minagi.v161.immutable_snapshot import stage_snapshot
    model = tmp_path / f"model-{tag}"
    adir = tmp_path / f"adapter-{tag}"
    if not model.exists():
        _real_model(model, adir, seed=seed)
    digests = {"model": close_tree(model).digest,
               "adapter": close_tree(adir).digest}
    return stage_snapshot(
        tmp_path / "snap" / tag,
        {"model": str(model), "adapter": str(adir)},
        expected_digests=digests,
        manifest_digest=digest({"m": tag}))


def _backend(ex):
    from minagi.runtime.worker_backend import (BackendSpec,
                                               WorkerBackend)
    return WorkerBackend(
        BackendSpec(module="minagi.v161.peft_serving",
                    qualname="PeftServingBackend", kwargs={}),
        backend_id="hf-peft", executor=ex,
        backend_manifest_digest="sha256:" + "d" * 64,
        start_timeout=180.0, probe_timeout=120.0,
        shutdown_grace=15.0)


def _stack_or_skip():
    for mod in ("torch", "transformers", "peft", "tokenizers"):
        pytest.importorskip(mod)


def test_g10_real_model_load_infer_terminate_inside_jail(tmp_path):
    """Full lifecycle under real containment: the worker loads a real
    HF model + PEFT adapter inside NsJail, answers inference, and the
    unit's cgroup verifies EMPTY after teardown."""
    _stack_or_skip()
    ex = _executor(tmp_path)
    b = _backend(ex)
    h = b.load(_snapshot(tmp_path, "a", seed=0))
    st = ex.containment_status(h.launched)
    assert st["cgroup_populated"] is True
    out = b.infer(h, {"prompt": "ask hello",
                      "max_new_tokens": 4,
                      "request_id": "g10-a"})
    assert isinstance(out["completion"], str)
    assert out["metrics"]["prompt_tokens"] > 0
    rep = ex.terminate(h.launched, deadline_seconds=15.0)
    assert rep.verified is True, rep.detail
    ex.release(h.launched)


def test_g10_real_model_cancel_and_switch_inside_jail(tmp_path):
    """Cooperative cancellation + a live model switch (rollback path):
    model A serves, a cancelled request reports cancelled, then model
    B loads in a fresh jail and serves — all under cgroup ownership."""
    _stack_or_skip()
    ex = _executor(tmp_path)
    b = _backend(ex)
    ha = b.load(_snapshot(tmp_path, "a", seed=0))
    b.cancel(ha)
    out = b.infer(ha, {"prompt": "ask world",
                       "max_new_tokens": 16,
                       "request_id": "g10-cancel"})
    assert out["metrics"]["cancelled"] is True
    b.unload(ha)
    # Rollback direction: the supervisor launches a NEW jailed worker
    # (one proxy supervises one worker) serving model B.
    b2 = _backend(ex)
    hb = b2.load(_snapshot(tmp_path, "b", seed=1))
    out_b = b2.infer(hb, {"prompt": "ask hello",
                          "max_new_tokens": 4,
                          "request_id": "g10-b"})
    assert isinstance(out_b["completion"], str)
    rep = ex.terminate(hb.launched, deadline_seconds=15.0)
    assert rep.verified is True
    ex.release(hb.launched)
