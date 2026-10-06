from pathlib import Path
from minagi.candidate_pipeline import (
    CandidateLedger, QualificationPolicy, Ed25519PromotionSigner,
    artifact_digest,
)


def _qualified(tmp_path):
    led=CandidateLedger(tmp_path/'authority')
    cid=led.propose(base_generation='g7',source='shadow')
    led.append(cid,'trained'); led.append(cid,'evaluated')
    led.qualify(cid,{'functional_accuracy':1.0,'heldout_regression':0.0,
                     'old_domain_regression':0.0},QualificationPolicy())
    return led,cid


def test_ed25519_public_verifier_cannot_sign_and_binds_tree(tmp_path):
    led,cid=_qualified(tmp_path)
    art=tmp_path/'candidate'; art.mkdir()
    (art/'a.bin').write_bytes(b'a'); (art/'sub').mkdir(); (art/'sub'/'b.bin').write_bytes(b'b')
    signer=Ed25519PromotionSigner.generate('k1')
    verifier=signer.verifier()
    receipt=signer.issue(led,cid,artifact_path=art,expected_base_generation='g7')
    assert verifier.verify_receipt(receipt,art)
    assert not hasattr(verifier,'issue')
    (art/'sub'/'b.bin').write_bytes(b'changed')
    assert not verifier.verify_receipt(receipt,art)


def test_artifact_tree_digest_is_path_order_stable(tmp_path):
    a=tmp_path/'a'; b=tmp_path/'b'; a.mkdir(); b.mkdir()
    for root in (a,b):
        (root/'z').write_text('z'); (root/'a').write_text('a')
    assert artifact_digest(a)[0] == artifact_digest(b)[0]
