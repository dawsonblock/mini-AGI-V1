from egai.common.crypto import Ed25519Signer,Ed25519Verifier
def trust():
    s=Ed25519Signer.generate('test');v=Ed25519Verifier();v.register(s.key_id,s.public_bytes());return s,v
