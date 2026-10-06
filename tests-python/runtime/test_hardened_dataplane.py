import pytest
from kvcontinual.execution.cache.persistent import PersistentExecutionArtifactStore
from kvcontinual.execution.durability.resource_budget import ResourceBudget,BudgetExceeded
from kvcontinual.execution.authority import Ed25519ReceiptSigner,Ed25519ReceiptVerifier


def test_fencing_stale_owner_survives_restart(tmp_path):
    s=PersistentExecutionArtifactStore(tmp_path);obj=s.cas.put(b'leased');gen=s.fence_lease_owner('worker')
    s.acquire_lease(obj.digest,'worker',gen,ttl_seconds=30,now=100)
    assert s.collect_orphans(now=110)==0
    new_gen=s.fence_lease_owner('worker');assert new_gen==gen+1;s.close()
    s=PersistentExecutionArtifactStore(tmp_path)
    with pytest.raises(RuntimeError,match='stale'):s.acquire_lease(obj.digest,'worker',gen,now=111)
    with pytest.raises(RuntimeError,match='stale'):s.release_lease(obj.digest,'worker',gen)
    assert s.collect_orphans(now=111)==1
    s.close()


def test_expired_lease_cannot_be_renewed(tmp_path):
    s=PersistentExecutionArtifactStore(tmp_path);obj=s.cas.put(b'payload');g=s.fence_lease_owner('worker')
    s.acquire_lease(obj.digest,'worker',g,ttl_seconds=1,now=100)
    with pytest.raises(RuntimeError,match='expired'):s.renew_lease(obj.digest,'worker',g,now=102)
    assert s.collect_orphans(now=102)==1;s.close()


def test_checkpoint_tampering_rejected_on_restart(tmp_path):
    s=PersistentExecutionArtifactStore(tmp_path);s.fence_lease_owner('w')
    with s.db:s.db.execute("UPDATE checkpoint SET sequence=99")
    s.close()
    with pytest.raises(RuntimeError,match='checkpoint'):PersistentExecutionArtifactStore(tmp_path)


def test_mutation_hash_tampering_rejected(tmp_path):
    s=PersistentExecutionArtifactStore(tmp_path);s.fence_lease_owner('w')
    with s.db:s.db.execute("UPDATE mutations SET payload='{}'")
    s.close()
    with pytest.raises(RuntimeError,match='hash chain'):PersistentExecutionArtifactStore(tmp_path)


def test_actual_cas_bytes_count_toward_admission(tmp_path):
    s=PersistentExecutionArtifactStore(tmp_path,budget=ResourceBudget(max_payload_bytes=4));s.cas.put(b'12345')
    with pytest.raises(BudgetExceeded):s.admission()
    s.close()


def test_revoked_execution_key_fails_verification():
    signer=Ed25519ReceiptSigner.generate();revoked=set()
    v=Ed25519ReceiptVerifier(signer.private_key.public_key(),key_id=signer.key_id,revocation_check=lambda k:k in revoked)
    r=signer.issue({'identity':'model'});assert v.verify(r)
    revoked.add(signer.key_id);assert not v.verify(r)


def test_unknown_token_accounting_cannot_admit_acceleration(tmp_path):
    from kvcontinual.execution.cache.block import ExecutionArtifact
    from kvcontinual.execution.types import ExecutionIdentity,ModelIdentity
    i=ExecutionIdentity(ModelIdentity('w','a','t','arch','rope'),'abi','gdn','layout','f32','f16')
    a=ExecutionArtifact('source',i,0,{},source_content_digest='sha256:'+'1'*64)
    s=PersistentExecutionArtifactStore(tmp_path);s.put(a)
    with pytest.raises(RuntimeError,match='token accounting'):s.admission()
    s.put(a,source_token_count=3);assert s.admission()=={'ok':1,'bad':0};s.close()


def test_budget_rejects_source_token_overflow_without_index_publication(tmp_path):
    from kvcontinual.execution.cache.block import ExecutionArtifact
    from kvcontinual.execution.types import ExecutionIdentity,ModelIdentity
    i=ExecutionIdentity(ModelIdentity('w','a','t','arch','rope'),'abi','gdn','layout','f32','f16')
    a=ExecutionArtifact('source',i,0,{},source_content_digest='sha256:'+'1'*64)
    s=PersistentExecutionArtifactStore(tmp_path,budget=ResourceBudget(max_tokens=2))
    with pytest.raises(BudgetExceeded):s.put(a,source_token_count=3)
    assert s.get('source',i) is None
    assert s.verify_recovery_checkpoint()['sequence']==0;s.close()
