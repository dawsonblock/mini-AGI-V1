from egai.common.canonical import digest
from egai.common.crypto import Ed25519Signer, Ed25519Verifier
from minagi.v161.executed_run import ExecutedRunReceiptV161

def test_signed_execution_receipt():
    d=digest({'x':1}); z='sha256:'+'0'*64
    s=Ed25519Signer.generate('executor'); v=Ed25519Verifier(); v.register(s.key_id,s.public_bytes())
    r=ExecutedRunReceiptV161.sign(signer=s,campaign_digest=d,arm='A0',seed=0,environment_digest=d,model_digest=d,tokenizer_digest=d,adapter_digest=z,dataset_digest=d,evaluator_digest=d,metrics={'score':1.0})
    assert r.verify(v)
