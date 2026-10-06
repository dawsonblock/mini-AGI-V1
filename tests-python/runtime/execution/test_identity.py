from kvcontinual.execution.cache.identity import compare_execution_identity
from kvcontinual.execution.types import Compatibility, ExecutionIdentity, ModelIdentity


def ident(adapter="a", kernel="k1", runtime="frozen"):
    m=ModelIdentity("base",adapter,"tok","qwen-hybrid","rope","bf16")
    return ExecutionIdentity(m,kernel,"gdn-v1","column-major","bf16","fp16",runtime)


def test_adapter_change_requires_replay():
    assert compare_execution_identity(ident("a"),ident("b")) == Compatibility.REPLAY_REQUIRED


def test_kernel_change_is_numeric_compat_only():
    assert compare_execution_identity(ident(kernel="k1"),ident(kernel="k2")) == Compatibility.NUMERICALLY_COMPATIBLE


def test_fast_weight_state_change_requires_replay():
    assert compare_execution_identity(ident(runtime="fw1"), ident(runtime="fw2")) == Compatibility.REPLAY_REQUIRED
