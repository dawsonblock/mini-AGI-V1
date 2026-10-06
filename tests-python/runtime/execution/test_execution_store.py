import numpy as np
from kvcontinual.execution.cache.block import ExecutionArtifact, RecurrentTailArtifact
from kvcontinual.execution.cache.store import ExecutionArtifactStore
from kvcontinual.execution.recurrent.affine import AffineSummary
from kvcontinual.execution.types import CacheTier, ExecutionIdentity, ModelIdentity


def ident(adapter="a"):
    return ExecutionIdentity(ModelIdentity("base",adapter,"tok","arch","rope"),"abi","rec","layout","bf16","fp16")


def art(sid, identity, seam=8, size=100):
    r=RecurrentTailArtifact(0,0,seam,AffineSummary(np.eye(2),np.zeros((2,2))))
    return ExecutionArtifact(sid,identity,seam,{(0,0):r},byte_size=size)


def test_store_namespaces_by_execution_identity():
    st=ExecutionArtifactStore(); st.put(art("s",ident("a")))
    assert st.get("s",ident("a")) is not None
    assert st.get("s",ident("b")) is None


def test_tier_accounting_and_namespace_evict():
    st=ExecutionArtifactStore(); i=ident(); st.put(art("a",i,size=100)); st.put(art("b",i,size=50))
    assert st.bytes_by_tier()[CacheTier.DRAM.value] == 150
    st.move_tier("a",i,CacheTier.HBM)
    assert st.bytes_by_tier()[CacheTier.HBM.value] == 100
    assert st.evict_namespace(i) == 2
