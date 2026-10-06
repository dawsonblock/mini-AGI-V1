import base64
from dataclasses import dataclass
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey
from cryptography.hazmat.primitives import serialization
from .canonical import canonical_bytes, sha256_bytes

@dataclass(frozen=True)
class SignedEnvelope:
    key_id:str
    signature_b64:str

class Ed25519Signer:
    def __init__(self, private_key:Ed25519PrivateKey, key_id:str|None=None):
        self._key=private_key
        pub=self.public_bytes()
        self.key_id=key_id or sha256_bytes(pub)[:24]
    @classmethod
    def generate(cls,key_id=None): return cls(Ed25519PrivateKey.generate(),key_id)
    @classmethod
    def from_private_bytes(cls,b,key_id=None): return cls(Ed25519PrivateKey.from_private_bytes(b),key_id)
    def private_bytes(self):
        return self._key.private_bytes(serialization.Encoding.Raw,serialization.PrivateFormat.Raw,serialization.NoEncryption())
    def public_bytes(self):
        return self._key.public_key().public_bytes(serialization.Encoding.Raw,serialization.PublicFormat.Raw)
    def sign(self,obj)->SignedEnvelope:
        sig=self._key.sign(canonical_bytes(obj))
        return SignedEnvelope(self.key_id,base64.b64encode(sig).decode())

class Ed25519Verifier:
    def __init__(self): self._keys={}
    def register(self,key_id:str,public_bytes:bytes):
        self._keys[key_id]=Ed25519PublicKey.from_public_bytes(public_bytes)
    def verify(self,obj,envelope:SignedEnvelope)->bool:
        k=self._keys.get(envelope.key_id)
        if not k: return False
        try:
            k.verify(base64.b64decode(envelope.signature_b64),canonical_bytes(obj)); return True
        except Exception: return False
