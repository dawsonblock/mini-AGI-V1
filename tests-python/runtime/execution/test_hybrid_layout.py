from kvcontinual.execution.hybrid.layout import HybridLayerKind, HybridModelLayout
from kvcontinual.execution.hybrid.instrumentation import build_capture_manifest


def test_qwen_style_interval_layout_and_capture_manifest():
    cfg = {
        "model_type": "qwen3_5",
        "num_hidden_layers": 8,
        "full_attention_interval": 4,
        "num_attention_heads": 8,
        "num_key_value_heads": 2,
        "hidden_size": 1024,
        "linear_num_key_heads": 4,
        "linear_num_value_heads": 8,
        "conv_kernel_size": 4,
    }
    layout = HybridModelLayout.from_config(cfg)
    assert layout.gdn_layers == (0,1,2,4,5,6)
    assert layout.full_attention_layers == (3,7)
    assert layout.layers[0].kind == HybridLayerKind.GATED_DELTA
    assert layout.layers[0].q_heads == 8
    assert layout.layers[0].kv_heads == 4
    manifest = build_capture_manifest(layout, seam_width=8)
    assert manifest.required_gdn_layers() == layout.gdn_layers
    assert manifest.required_attention_layers() == layout.full_attention_layers
    assert manifest.digest.startswith("sha256:")


def test_explicit_layer_types_override_interval():
    cfg = {
        "model_type":"hybrid",
        "num_hidden_layers":4,
        "layer_types":["gated_delta","full_attention","gated_delta","full_attention"],
        "num_attention_heads":4,
        "num_key_value_heads":2,
        "head_dim":16,
    }
    layout = HybridModelLayout.from_config(cfg)
    assert layout.gdn_layers == (0,2)
    assert layout.full_attention_layers == (1,3)
