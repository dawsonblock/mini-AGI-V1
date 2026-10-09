"""NativeAdapter2 (v2 bundle) — PEFT safetensors -> per-layer attention
LoRA compilation with byte-verified F32 payloads.

Covers: q/k/v/o + lm_head target parsing, per-layer dims validation,
scale = alpha/rank, unsupported-target rejection, and numeric parity
of the emitted bundle against the source safetensors math.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src-python"))

from safetensors.numpy import save_file  # noqa: E402

from egai.common.canonical import sha256_bytes  # noqa: E402
from minagi.v15.native_adapter import (  # noqa: E402
    NativeAttentionDims, NativeQW3Adapter2Compiler,
    native_adapter2_supports_target, native_adapter_supports_target)

Z = "sha256:" + "0" * 64
HEX = "a" * 64

DIMS = NativeAttentionDims(n_layers=4, hidden=8, q_rows=16, kv_rows=4,
                           o_in=16)


def _mk_adapter(tmp: Path, layers=(0, 1), kinds=("q_proj", "k_proj",
                                                "v_proj", "o_proj"),
                rank=2, extra=None) -> Path:
    """Write a minimal PEFT-style safetensors adapter."""
    rng = np.random.default_rng(7)
    tensors = {}
    for layer in layers:
        for kind in kinds:
            out_f = {"q_proj": 16, "k_proj": 4, "v_proj": 4,
                     "o_proj": 8}[kind]
            in_f = 8 if kind != "o_proj" else 16
            pre = (f"base_model.model.model.layers.{layer}"
                   f".self_attn.{kind}")
            tensors[f"{pre}.lora_A.weight"] = rng.normal(
                size=(rank, in_f)).astype(np.float32)
            tensors[f"{pre}.lora_B.weight"] = rng.normal(
                size=(out_f, rank)).astype(np.float32)
    if extra:
        tensors.update(extra)
    d = tmp / "adapter"
    d.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(d / "adapter_model.safetensors"))
    (d / "adapter_config.json").write_text(json.dumps(
        {"peft_type": "LORA", "r": rank, "lora_alpha": 32}))
    return d


def _compile(tmp: Path, adapter: Path, **kw):
    out = tmp / "native"
    c = NativeQW3Adapter2Compiler()
    return c.compile(
        adapter_dir=adapter, output_dir=out,
        adapter_set_root=HEX, model_sha256=HEX,
        foundation_model_digest=Z,
        source_adapter_manifest_digest=Z,
        qualified_candidate_digest=Z,
        dims=DIMS, lora_alpha=32.0,
        **kw), out


def test_v2_target_support_gate():
    for t in ("q_proj", "k_proj", "v_proj", "o_proj"):
        assert native_adapter2_supports_target(
            f"model.layers.3.self_attn.{t}")
    assert native_adapter2_supports_target("lm_head")
    assert not native_adapter2_supports_target("mlp.gate_proj")
    # v1 gate stays narrow — attention targets are NOT v1-servable
    assert not native_adapter_supports_target(
        "model.layers.0.self_attn.q_proj")


def test_v2_compile_emits_per_layer_entries(tmp_path):
    adapter = _mk_adapter(tmp_path)
    bundle, out = _compile(tmp_path, adapter)
    assert bundle.schema == "qw3-native-lora-bundle-v2"
    assert len(bundle.tensors) == 8  # 2 layers x 4 kinds
    man = json.loads((out / "manifest.json").read_bytes())
    assert man["schema"] == "qw3-native-lora-bundle-v2"
    kinds = {(t["target"], t["layer"]) for t in man["tensors"]}
    assert ("self_attn.q_proj", 0) in kinds
    assert ("self_attn.o_proj", 1) in kinds
    # dims correct per kind
    q = next(t for t in bundle.tensors if t.target == "self_attn.q_proj")
    assert (q.in_features, q.out_features) == (8, 16)
    o = next(t for t in bundle.tensors if t.target == "self_attn.o_proj")
    assert (o.in_features, o.out_features) == (16, 8)
    # scale = alpha / rank = 32/2 = 16
    assert all(abs(t.scale - 16.0) < 1e-9 for t in bundle.tensors)


def test_v2_compiled_weights_match_source(tmp_path):
    """The emitted F32 payloads must reproduce scale*B@(A*x) bit-for-
    bit against the safetensors source — the compile is a layout
    normalization, not a transformation."""
    adapter = _mk_adapter(tmp_path)
    bundle, out = _compile(tmp_path, adapter)
    src = dict(__import__("safetensors.numpy", fromlist=["x"])
               .load_file(str(adapter / "adapter_model.safetensors")))
    x = np.arange(8, dtype=np.float32) * 0.125
    t = next(t for t in bundle.tensors
             if t.target == "self_attn.q_proj" and t.layer == 0)
    a_src = src["base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight"]
    b_src = src["base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight"]
    a_cmp = np.frombuffer((out / t.a.file).read_bytes(),
                          dtype="<f4").reshape(t.rank, t.in_features)
    b_cmp = np.frombuffer((out / t.b.file).read_bytes(),
                          dtype="<f4").reshape(t.out_features, t.rank)
    np.testing.assert_allclose(a_cmp, a_src, rtol=0, atol=0)
    np.testing.assert_allclose(b_cmp, b_src, rtol=0, atol=0)
    delta_bundle = t.scale * (b_cmp @ (a_cmp @ x))
    delta_source = t.scale * (b_src @ (a_src @ x))
    np.testing.assert_allclose(delta_bundle, delta_source, rtol=0, atol=0)


def test_v2_rejects_unknown_target(tmp_path):
    bad = {"base_model.model.model.layers.0.mlp.gate_proj.lora_A.weight":
           np.zeros((2, 8), np.float32),
           "base_model.model.model.layers.0.mlp.gate_proj.lora_B.weight":
           np.zeros((8, 2), np.float32)}
    adapter = _mk_adapter(tmp_path, extra=bad)
    with pytest.raises(ValueError, match="unsupported LoRA target"):
        _compile(tmp_path, adapter)


def test_v2_rejects_layer_out_of_range(tmp_path):
    adapter = _mk_adapter(tmp_path, layers=(9,))
    with pytest.raises(ValueError, match="n_layers"):
        _compile(tmp_path, adapter)


def test_v2_rejects_dim_mismatch(tmp_path):
    """An adapter whose q_proj output width disagrees with the declared
    model geometry must not compile."""
    bad = {"base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight":
           np.zeros((2, 8), np.float32),
           "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight":
           np.zeros((99, 2), np.float32)}
    adapter = _mk_adapter(tmp_path / "sub", layers=(), extra=bad)
    with pytest.raises(ValueError):
        _compile(tmp_path / "sub", adapter)


def test_v2_bundle_tamper_breaks_manifest(tmp_path):
    """A modified F32 payload must no longer match the recorded sha."""
    adapter = _mk_adapter(tmp_path)
    bundle, out = _compile(tmp_path, adapter)
    t = bundle.tensors[0]
    f = out / t.a.file
    raw = bytearray(f.read_bytes())
    raw[-1] ^= 0xFF
    f.write_bytes(bytes(raw))
    assert sha256_bytes(f.read_bytes()).split(":", 1)[-1] != t.a.sha256
