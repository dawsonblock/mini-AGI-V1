from kvcontinual.execution.registry import AdapterRegistry
from kvcontinual.execution.qualification import QualificationPolicy
from kvcontinual.execution.types import PromotionDecision


def make_candidate(reg,tmp_path,payload):
    f=tmp_path/(str(len(payload))+".safetensors"); f.write_bytes(payload)
    return reg.register_candidate(str(f),"sha256:base","sha256:data",{"lr":1e-5})


def qualify_and_stage(reg,c):
    q=QualificationPolicy().evaluate(c.candidate_id,.04,.005,True,True)
    assert q.decision==PromotionDecision.PROMOTE
    reg.write_qualification(q); reg.stage_cache_namespace(c.candidate_id,"sha256:ns",.8)


def test_promotion_requires_staged_cache_namespace(tmp_path):
    reg=AdapterRegistry(str(tmp_path/"registry")); c=make_candidate(reg,tmp_path,b"a")
    q=QualificationPolicy().evaluate(c.candidate_id,.04,.005,True,True); reg.write_qualification(q)
    try: reg.promote(c.candidate_id)
    except RuntimeError as e: assert "namespace" in str(e)
    else: raise AssertionError("promotion should fail without staged namespace")


def test_atomic_promote_and_rollback_namespace(tmp_path):
    reg=AdapterRegistry(str(tmp_path/"registry"))
    c1=make_candidate(reg,tmp_path,b"a"); qualify_and_stage(reg,c1); reg.promote(c1.candidate_id)
    c2=make_candidate(reg,tmp_path,b"bb"); qualify_and_stage(reg,c2); reg.promote(c2.candidate_id)
    assert reg.current_id()==c2.candidate_id
    assert reg.rollback()==c1.candidate_id
