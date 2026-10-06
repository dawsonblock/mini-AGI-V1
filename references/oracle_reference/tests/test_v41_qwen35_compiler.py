import torch
import pytest
from minagi.v4 import (
    CapturedGDNLayer, EffectiveModelGeneration, CompilePolicy,
    compile_qwen35_layer, compile_qwen35_block, FAPICSegment,
    TransitionOrientation,
)


def _capture(layer=0,t=48,h=2,dk=4,dv=5,channels=6):
    torch.manual_seed(100+layer)
    k=torch.randn(t,h,dk)*.15
    k=k/(k.norm(dim=-1,keepdim=True)+1e-6)
    v=torch.randn(t,h,dv)*.1
    beta=torch.sigmoid(torch.randn(t,h))*.75
    decay=torch.sigmoid(torch.randn(t,h))*.3+.65
    conv=torch.randn(t,channels)
    return CapturedGDNLayer(layer,k,v,beta,decay,conv,4)


def test_qwen35_compile_per_layer_per_head_and_conv_boundary():
    cap=_capture()
    compiled=compile_qwen35_layer(cap,seams=(8,32),segment_id='seg')
    assert [x.seam_width for x in compiled.variants]==[8,32]
    assert compiled.variants[0].transition.shape==(2,4,4)
    assert compiled.variants[0].zero_state.shape==(2,4,5)
    assert compiled.conv_boundary is not None
    assert compiled.conv_boundary.history.shape==(6,3)


def test_tiered_block_compiler_limits_transition_variants():
    caps=[_capture(0),_capture(3)]
    g=EffectiveModelGeneration.build(base_weights='base',tokenizer='tok',cache_abi_version=2)
    fa=FAPICSegment(torch.randn(2,48,8),torch.randn(2,48,8),tuple(range(48)))
    warm=compile_qwen35_block(block_id='b',tokens=list(range(48)),generation=g,captures=caps,
                              full_attention=fa,policy=CompilePolicy(tier='warm'))
    hot=compile_qwen35_block(block_id='h',tokens=list(range(48)),generation=g,captures=caps,
                             full_attention=fa,policy=CompilePolicy(tier='hot'))
    assert warm.supported_seams()==(8,)
    assert hot.supported_seams()==(8,32)
    assert len(hot.variant(8).summaries)==2
    assert set(hot.boundary.trailing_conv_payload)=={0,3}
    assert hot.compatibility.cache_abi_version==2


def test_adapter_change_makes_compiled_block_incompatible():
    cap=_capture()
    g1=EffectiveModelGeneration.build(base_weights='base',tokenizer='tok',cache_abi_version=2)
    block=compile_qwen35_block(block_id='b',tokens=list(range(48)),generation=g1,captures=[cap])
    g2=EffectiveModelGeneration.build(base_weights='base',tokenizer='tok',cache_abi_version=2,
                                      adapters=[('skill','sha-new',1.0)])
    with pytest.raises(ValueError,match='different effective model generation'):
        block.compatibility.assert_compatible(g2)
