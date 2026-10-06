from types import SimpleNamespace
import torch
import pytest

from minagi.v4 import (
    CapturedGDNLayer, CapturedFullAttentionLayer, ForwardCapture,
    EffectiveModelGeneration, ModelProbeReport,
    save_trace_bundle, load_trace_bundle,
)


def _generation(adapter=False):
    return EffectiveModelGeneration.build(
        base_weights='base-sha', tokenizer='tok-sha', cache_abi_version=3,
        adapters=[('skill','adapter-sha',1.0)] if adapter else (),
    )


def test_trace_bundle_roundtrip_and_generation_guard(tmp_path):
    cap=ForwardCapture(
        recurrent={2:CapturedGDNLayer(2,torch.randn(1,4,2,3),torch.randn(1,4,2,5),torch.rand(1,4,2),torch.rand(1,4,2),torch.randn(1,4,16),3)},
        full_attention_inputs={3:CapturedFullAttentionLayer(3,torch.randn(1,4,8),torch.arange(4).unsqueeze(0),None)},
    )
    probe=ModelProbeReport('M','qwen3_5','x',(2,),(3,),(('g','GDN'),('a','Attention')),'cfg')
    g=_generation()
    manifest=save_trace_bundle(tmp_path/'trace',capture=cap,generation=g,probe=probe,tokens=[1,2,3,4])
    loaded,recurrent,full=load_trace_bundle(tmp_path/'trace',generation=g)
    assert loaded.generation_digest==g.digest
    assert set(recurrent)=={2}; assert set(full)=={3}
    torch.testing.assert_close(recurrent[2].keys,cap.recurrent[2].keys)
    with pytest.raises(ValueError,match='different effective model generation'):
        load_trace_bundle(tmp_path/'trace',generation=_generation(adapter=True))


def test_trace_bundle_detects_payload_tampering(tmp_path):
    cap=ForwardCapture(recurrent={},full_attention_inputs={})
    probe=ModelProbeReport('M','x','x',(),(),(),'cfg')
    save_trace_bundle(tmp_path/'trace',capture=cap,generation=_generation(),probe=probe,tokens=[])
    path=tmp_path/'trace'/'capture_tensors.pt'
    path.write_bytes(path.read_bytes()+b'corrupt')
    with pytest.raises(ValueError,match='hash mismatch'):
        load_trace_bundle(tmp_path/'trace',generation=_generation())
