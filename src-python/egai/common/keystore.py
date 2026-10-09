from pathlib import Path
import os
import json
import base64
from .crypto import Ed25519Signer,Ed25519Verifier

class LocalKeyStore:
    """Local reference keystore. Production deployments should replace this with OS/HSM-backed keys."""
    def __init__(self,root):
        self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True)
    def load_or_create(self,name):
        priv=self.root/(name+'.key'); meta=self.root/(name+'.json')
        if priv.exists() and meta.exists():
            m=json.loads(meta.read_text());return Ed25519Signer.from_private_bytes(base64.b64decode(priv.read_text()),m['key_id'])
        s=Ed25519Signer.generate(name)
        priv.write_text(base64.b64encode(s.private_bytes()).decode()); os.chmod(priv,0o600)
        meta.write_text(json.dumps({'key_id':s.key_id,'public_b64':base64.b64encode(s.public_bytes()).decode()},indent=2))
        return s
    def build_verifier(self):
        v=Ed25519Verifier()
        for p in self.root.glob('*.json'):
            m=json.loads(p.read_text());v.register(m['key_id'],base64.b64decode(m['public_b64']))
        return v
