from kvcontinual.execution.hybrid import (
    AttentionLayerCapture, GdnLayerCapture, HybridCaptureBuffer,
    HybridModelLayout, build_capture_manifest,
)


def test_capture_buffer_requires_every_hybrid_layer_payload():
    cfg={"num_hidden_layers":4,"full_attention_interval":4,"num_attention_heads":4,
         "num_key_value_heads":2,"head_dim":8,"conv_kernel_size":4,"model_type":"qwen"}
    manifest=build_capture_manifest(HybridModelLayout.from_config(cfg),8)
    b=HybridCaptureBuffer(manifest)
    for layer in (0,1,2):
        b.put_gdn(layer,GdnLayerCapture("k","v","g","b","proj","hist","h"))
    assert not b.complete
    assert b.missing_requirements()==["layer 3:attention"]
    b.put_attention(3,AttentionLayerCapture("pre_k","v","pos","h"))
    assert b.complete
