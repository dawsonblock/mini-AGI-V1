from kvcontinual.continual.cache.identity import compare_cache_identity
from kvcontinual.continual.types import CacheIdentity, Compatibility


def ident(adapter):
    return CacheIdentity("base", adapter, "tok", "layout", "rope", "rec")


def test_adapter_change_requires_replay():
    assert compare_cache_identity(ident("a"), ident("b")) == Compatibility.REPLAY_REQUIRED


def test_exact_identity():
    assert compare_cache_identity(ident("a"), ident("a")) == Compatibility.EXACT_COMPATIBLE
