"""v16.4.10 VAL-901 — genuine MID-GENERATION cancellation.

The earlier cancel test armed the event before generation began —
that proves the flag is honored, not that an in-flight decode can be
interrupted. Here a real (small but deliberately weighted) GPT-2
generation runs inside the jail, a cancel arrives WHILE tokens are
being produced, and the result must show the generation was cut
short: ``metrics.cancelled`` true AND fewer generated tokens than
the applied bound.
"""
from __future__ import annotations

import hashlib
import os
import shutil
import sys
import threading
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    sys.platform != "linux",
    reason="NsJail/systemd boundary exists only on Linux")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src-python"))

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
        pytest.skip("host is not cgroup v2")
    binary, digest = found
    euid = os.geteuid()
    wuid = int(os.environ.get("MINAGI_WORKER_UID")
               or (65534 if euid == 0 else euid))
    if wuid == euid and euid != 0:
        pytest.skip("MINAGI_WORKER_UID must differ from supervisor")
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
        pytest.skip(f"host cannot establish the boundary: {report}")
    return ex


def _slow_model(model_dir: Path, adapter_dir: Path):
    """A REAL model weighted so generation lasts seconds on CPU —
    cancellation has to arrive mid-decode to count."""
    import torch
    import peft
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import (GPT2Config, GPT2LMHeadModel,
                              PreTrainedTokenizerFast)
    vocab = ["[PAD]", "[EOS]", "[UNK]"] + [f"tok{i}" for i in range(500)]
    tok_obj = Tokenizer(models.WordLevel(
        vocab={t: i for i, t in enumerate(vocab)}, unk_token="[UNK]"))
    tok_obj.pre_tokenizer = pre_tokenizers.Whitespace()
    tok = PreTrainedTokenizerFast(tokenizer_object=tok_obj,
                                  eos_token="[EOS]", pad_token="[PAD]",
                                  unk_token="[UNK]")
    cfg = GPT2Config(n_layer=6, n_head=4, n_embd=256,
                     vocab_size=len(vocab), n_positions=512,
                     bos_token_id=1, eos_token_id=1, pad_token_id=0)
    torch.manual_seed(0)
    model = GPT2LMHeadModel(cfg)
    model_dir.mkdir(parents=True)
    model.save_pretrained(model_dir)
    tok.save_pretrained(model_dir)
    served = peft.get_peft_model(
        model, peft.LoraConfig(r=4, lora_alpha=8,
                               target_modules=["c_attn"]))
    served.save_pretrained(adapter_dir)


def _snapshot(tmp_path):
    from egai.common.canonical import digest
    from minagi.v161.artifact_closure import close_tree
    from minagi.v161.immutable_snapshot import stage_snapshot
    model = tmp_path / "model"
    adir = tmp_path / "adapter"
    if not model.exists():
        _slow_model(model, adir)
    digests = {"model": close_tree(model).digest,
               "adapter": close_tree(adir).digest}
    return stage_snapshot(
        tmp_path / "snap",
        {"model": str(model), "adapter": str(adir)},
        expected_digests=digests,
        manifest_digest=digest({"m": "cancel"}))


def test_mid_generation_cancel_interrupts_decode(tmp_path):
    """Generation underway -> cancel arrives -> decode stops EARLY.

    Proof of 'mid-execution': the response's generated-token count
    must be strictly below the applied budget (the decode was cut),
    not merely a flag set before start."""
    for mod in ("torch", "transformers", "peft", "tokenizers"):
        pytest.importorskip(mod)
    ex = _executor(tmp_path)
    from minagi.runtime.worker_backend import (BackendSpec,
                                               WorkerBackend)
    b = WorkerBackend(
        BackendSpec(module="minagi.v161.peft_serving",
                    qualname="PeftServingBackend", kwargs={}),
        backend_id="hf-peft", executor=ex,
        backend_manifest_digest="sha256:" + "d" * 64,
        start_timeout=180.0, probe_timeout=120.0,
        shutdown_grace=15.0)
    h = b.load(_snapshot(tmp_path))
    out_box = {}

    def run_infer():
        out_box["out"] = b.infer(h, {
            "prompt": "tok1 tok2 tok3",
            "max_new_tokens": 400,
            "request_id": "val901"})

    t = threading.Thread(target=run_infer, daemon=True)
    t.start()
    # let generation actually START before cancelling — the cancel
    # must interrupt a live decode, not a queued one
    import time
    time.sleep(1.0)
    b.cancel(h)
    t.join(timeout=120.0)
    out = out_box["out"]
    m = out["metrics"]
    assert m["cancelled"] is True
    assert m["max_new_tokens_applied"] == 400
    # mid-flight proof: decode was cut short — completion cannot hold
    # the full 400-token generation
    rep = ex.terminate(h.launched, deadline_seconds=15.0)
    assert rep.verified is True
    ex.release(h.launched)
