from dataclasses import dataclass, asdict, replace
from egai.common.canonical import digest, validate_digest
from egai.common.crypto import SignedEnvelope

@dataclass(frozen=True)
class VerificationReceipt:
    receipt_id:str
    case_id:str
    task_kind:str
    input_digest:str
    attempted_output_digest:str
    repaired_output_digest:str
    expected_digest:str
    score:float
    passed:bool
    verifier_id:str
    verifier_key_id:str=''
    signature_b64:str=''
    def __post_init__(self):
        for d in (self.input_digest,self.attempted_output_digest,self.repaired_output_digest,self.expected_digest): validate_digest(d)
        if not 0.0 <= float(self.score) <= 1.0: raise ValueError('verification score must be within [0,1]')
    @property
    def digest(self): return digest(self)
    def unsigned(self): return replace(self,verifier_key_id='',signature_b64='')

class VerificationAuthority:
    """Produces signed repair-verification receipts for experience episodes."""
    def __init__(self, verifier_id, signer, scorer):
        self.verifier_id=verifier_id; self.signer=signer; self.scorer=scorer
    def verify_repair(self, case, attempted_output, repaired_output):
        from egai.common.canonical import digest
        score=float(self.scorer(repaired_output, case.expected))
        u=VerificationReceipt(
            'vr-'+case.case_id, case.case_id, case.task_kind,
            digest({'input':case.input}), digest({'output':attempted_output}),
            digest({'output':repaired_output}), digest({'expected':case.expected}),
            score, score >= 1.0, self.verifier_id)
        env=self.signer.sign(asdict(u))
        return replace(u,verifier_key_id=env.key_id,signature_b64=env.signature_b64)

class VerificationReceiptValidator:
    def __init__(self, verifier, trusted_key_ids):
        self.verifier=verifier; self.trusted=set(trusted_key_ids)
    def validate(self, receipt, case, attempted_output, repaired_output):
        from egai.common.canonical import digest
        if receipt.verifier_key_id not in self.trusted: raise PermissionError('untrusted feedback verifier')
        if not self.verifier.verify(asdict(receipt.unsigned()), SignedEnvelope(receipt.verifier_key_id, receipt.signature_b64)):
            raise PermissionError('invalid feedback verification signature')
        expected={
            'case_id':case.case_id, 'task_kind':case.task_kind,
            'input_digest':digest({'input':case.input}),
            'attempted_output_digest':digest({'output':attempted_output}),
            'repaired_output_digest':digest({'output':repaired_output}),
            'expected_digest':digest({'expected':case.expected})}
        for k,v in expected.items():
            if getattr(receipt,k)!=v: raise ValueError('verification receipt binding mismatch: '+k)
        if not receipt.passed or receipt.score < 1.0: raise ValueError('repair did not pass verifier')
        return True
