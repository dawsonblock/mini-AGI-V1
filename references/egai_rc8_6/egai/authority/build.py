from dataclasses import asdict, replace
from egai.common.crypto import SignedEnvelope
from egai.common.canonical import validate_digest
from .model import BuildManifest

class BuildAuthority:
    """Build-plane signer. It binds a candidate to an environment and produced artifact."""
    def __init__(self,builder_id,signer,artifact_store):
        self.builder_id=builder_id; self.signer=signer; self.artifact_store=artifact_store
    def attest(self,build_id,candidate,environment_digest,output_digest):
        for d in (candidate.artifact_digest,environment_digest,output_digest):
            validate_digest(d)
            if not self.artifact_store.exists(d): raise FileNotFoundError(d)
        u=BuildManifest(build_id,candidate.digest,environment_digest,output_digest,self.builder_id)
        env=self.signer.sign(asdict(u)); return replace(u,builder_key_id=env.key_id,signature_b64=env.signature_b64)
