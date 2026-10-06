from types import SimpleNamespace
import torch
import torch.nn as nn
import torch.nn.functional as F

from minagi.v4 import (
    HybridLayerKind, discover_hybrid_layout, capture_gdn_from_hidden,
    HFHybridTraceCollector, extract_full_attention_pic, gdn_scan,
    zero_state_for, TransitionOrientation,
)


class FakeGatedDeltaNet(nn.Module):
    def __init__(self, hidden=8, heads=2, dk=2, dv=2, kernel=3, layer_idx=0):
        super().__init__()
        self.layer_idx = layer_idx
        self.num_v_heads = heads
        self.num_k_heads = heads
        self.head_k_dim = dk
        self.head_v_dim = dv
        self.key_dim = heads * dk
        self.value_dim = heads * dv
        total = self.key_dim * 2 + self.value_dim
        self.in_proj_qkv = nn.Linear(hidden, total, bias=False)
        self.in_proj_b = nn.Linear(hidden, heads, bias=False)
        self.in_proj_a = nn.Linear(hidden, heads, bias=False)
        self.A_log = nn.Parameter(torch.zeros(heads))
        self.dt_bias = nn.Parameter(torch.zeros(heads))
        self.conv1d = nn.Conv1d(total, total, kernel, groups=total, bias=True)
        self.activation = "silu"

    def forward(self, hidden_states, **kwargs):
        # The collector captures pre-forward inputs; the fake module does not need
        # to duplicate the vendor kernel to exercise hook lifetime semantics.
        return hidden_states


class FakeAttention(nn.Module):
    def __init__(self, hidden=8, layer_idx=1):
        super().__init__()
        self.layer_idx = layer_idx
        self.q_proj = nn.Linear(hidden, hidden, bias=False)
        self.k_proj = nn.Linear(hidden, hidden, bias=False)
        self.v_proj = nn.Linear(hidden, hidden, bias=False)

    def forward(self, hidden_states, **kwargs):
        return hidden_states


class FakeHybridModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(64, 8)
        self.gdn = FakeGatedDeltaNet(layer_idx=0)
        self.attention = FakeAttention(layer_idx=1)
        self.head = nn.Linear(8, 64, bias=False)
        self.config = SimpleNamespace(model_type="fake_hybrid", to_dict=lambda: {"model_type": "fake_hybrid", "hidden": 8})

    def forward(self, input_ids, use_cache=False, output_hidden_states=False, **kwargs):
        x = self.embed(input_ids)
        h0 = x
        x = self.gdn(x)
        h1 = x
        x = self.attention(x)
        logits = self.head(x)
        hs = (h0, h1, x) if output_hidden_states else None
        return SimpleNamespace(logits=logits, hidden_states=hs, past_key_values=None)


def test_layout_discovery_is_capability_based():
    model = FakeHybridModel()
    layout = discover_hybrid_layout(model)
    assert [(x.layer_idx, x.kind) for x in layout.layers] == [
        (0, HybridLayerKind.GATED_DELTA),
        (1, HybridLayerKind.FULL_ATTENTION),
    ]
    layout.assert_hybrid()


def test_capture_matches_direct_transformers_style_projection_path():
    torch.manual_seed(2)
    mod = FakeGatedDeltaNet()
    hs = torch.randn(1, 7, 8)
    cap = capture_gdn_from_hidden(mod, hs)

    raw = mod.in_proj_qkv(hs)
    w = mod.conv1d.weight.squeeze(1)
    # Independent full-sequence causal depthwise convolution.
    x = raw.transpose(1, 2)
    y = F.conv1d(F.pad(x, (w.shape[-1] - 1, 0)), w.unsqueeze(1), mod.conv1d.bias, groups=w.shape[0])
    mixed = F.silu(y.transpose(1, 2))
    _, k, v = torch.split(mixed, [mod.key_dim, mod.key_dim, mod.value_dim], dim=-1)
    k = k.reshape(1, 7, mod.num_k_heads, mod.head_k_dim)
    k = k * torch.rsqrt((k.float() * k.float()).sum(-1, keepdim=True) + 1e-6).to(k.dtype)
    v = v.reshape(1, 7, mod.num_v_heads, mod.head_v_dim)
    beta = mod.in_proj_b(hs).sigmoid()
    g = -mod.A_log.float().exp() * F.softplus(mod.in_proj_a(hs).float() + mod.dt_bias)

    torch.testing.assert_close(cap.keys, k)
    torch.testing.assert_close(cap.values, v)
    torch.testing.assert_close(cap.beta, beta)
    torch.testing.assert_close(cap.decay, torch.exp(g).to(v.dtype))
    torch.testing.assert_close(cap.conv_inputs, raw)

    steps = []
    from minagi.v4.qwen35_rc10 import captured_steps
    steps = captured_steps(cap)
    state = zero_state_for(steps[0], orientation=TransitionOrientation.LEFT)
    final = gdn_scan(steps, state, orientation=TransitionOrientation.LEFT)
    assert final.shape == (1, mod.num_v_heads, mod.head_k_dim, mod.head_v_dim)
    assert torch.isfinite(final).all()


def test_trace_collector_hooks_real_module_calls_and_removes_hooks():
    torch.manual_seed(3)
    model = FakeHybridModel()
    collector = HFHybridTraceCollector(model)
    out = collector.run(torch.tensor([[1, 2, 3, 4]]), output_hidden_states=True)
    assert set(out.recurrent) == {0}
    assert set(out.full_attention_inputs) == {1}
    assert out.logits.shape == (1, 4, 64)
    assert len(out.hidden_states) == 3
    # Running normally after capture must not append/modify capture state through stale hooks.
    before = out.recurrent[0].keys.clone()
    model(torch.tensor([[5, 6]]))
    torch.testing.assert_close(before, out.recurrent[0].keys)


def test_extract_full_attention_pic_from_hybrid_cache_layer():
    class L:
        pass
    layer = L()
    layer.keys = torch.randn(1, 2, 5, 4)
    layer.values = torch.randn(1, 2, 5, 4)
    cache = SimpleNamespace(layers=[SimpleNamespace(), layer])
    pic = extract_full_attention_pic(cache, 1, positions=[10, 11, 12, 13, 14], token_digest="abc")
    pic.validate()
    assert pic.source_positions == (10, 11, 12, 13, 14)
    assert pic.token_digest == "abc"
