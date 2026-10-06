import numpy as np
from kvcontinual.execution.cache.block import ExecutionArtifact, RecurrentTailArtifact
from kvcontinual.execution.cache.persistent import PersistentExecutionArtifactStore
from kvcontinual.execution.recurrent.affine import AffineSummary
from kvcontinual.execution.types import CacheTier, ExecutionIdentity, ModelIdentity

def ident(tag='a'):
    return ExecutionIdentity(ModelIdentity('w'+tag,'ad'+tag,'tok','arch','rope'),'abi','gdn','layout','f32','f16')
def artifact(i):
    r=RecurrentTailArtifact(0,0,8,AffineSummary(np.eye(2),np.ones((2,2))))
    return ExecutionArtifact('seg',i,8,{(0,0):r},source_content_digest='sha256:'+'1'*64,byte_size=17,tier=CacheTier.NVME)
def test_persistent_roundtrip_restart(tmp_path):
    i=ident(); s=PersistentExecutionArtifactStore(tmp_path); s.put(artifact(i)); s.close()
    s=PersistentExecutionArtifactStore(tmp_path); a=s.get('seg',i,'sha256:'+'1'*64); assert a is not None
    assert np.array_equal(a.recurrent[(0,0)].tail_summary.Z,np.ones((2,2))); assert s.bytes_by_tier()['NVME']>17; s.close()
def test_wrong_source_digest_fails_closed(tmp_path):
    i=ident(); s=PersistentExecutionArtifactStore(tmp_path); s.put(artifact(i)); assert s.get('seg',i,'sha256:'+'2'*64) is None; s.close()
def test_namespace_evict_persists(tmp_path):
    i=ident(); s=PersistentExecutionArtifactStore(tmp_path); s.put(artifact(i)); assert s.evict_namespace(i)==1; s.close()
    s=PersistentExecutionArtifactStore(tmp_path); assert s.get('seg',i) is None; s.close()
def test_corrupt_cas_fails_closed_and_removes_index(tmp_path):
    i=ident(); s=PersistentExecutionArtifactStore(tmp_path); s.put(artifact(i)); d=s.db.execute('select object_digest from artifacts').fetchone()[0]
    p=s.cas.path_for(d); p.write_bytes(b'corrupt'); assert s.get('seg',i) is None
    assert s.db.execute('select count(*) from artifacts').fetchone()[0]==0; s.close()
