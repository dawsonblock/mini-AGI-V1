from dataclasses import dataclass,asdict,replace
from egai.common.canonical import digest,validate_digest
from egai.common.crypto import SignedEnvelope
@dataclass(frozen=True)
class HTTPRuntimeAttestation:
    served_model:str;model_digest:str;runtime_digest:str;backend_version:str;issuer_id:str;issuer_key_id:str='';signature_b64:str=''
    def __post_init__(self):validate_digest(self.model_digest);validate_digest(self.runtime_digest)
    @property
    def digest(self):return digest(self)
    def unsigned(self):return replace(self,issuer_key_id='',signature_b64='')

def issue_attestation(served_model,model_digest,runtime_digest,backend_version,issuer_id,signer):
    u=HTTPRuntimeAttestation(served_model,model_digest,runtime_digest,backend_version,issuer_id);e=signer.sign(asdict(u));return replace(u,issuer_key_id=e.key_id,signature_b64=e.signature_b64)

def verify_attestation(a,verifier,trusted_keys,expected_model_digest,served_model):
    if a.issuer_key_id not in set(trusted_keys):raise PermissionError('untrusted HTTP runtime attestation')
    if a.model_digest!=expected_model_digest or a.served_model!=served_model:raise ValueError('HTTP runtime attestation binding mismatch')
    if not verifier.verify(asdict(a.unsigned()),SignedEnvelope(a.issuer_key_id,a.signature_b64)):raise PermissionError('invalid HTTP runtime attestation signature')
    return True
