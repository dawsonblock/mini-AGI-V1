from pathlib import Path
from egai.common.keystore import LocalKeyStore
from egai.evidence.ledger import EvidenceLedger
from egai.epistemic.beliefs import BeliefCompiler
from egai.skills.registry import SkillRegistry
from egai.authority.ledger import ImprovementLedger
from egai.authority.runtime_registry import RuntimeRegistry
from egai.common.artifacts import ArtifactStore

class Runtime:
    def __init__(self,root,signer=None,verifier=None):
        self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True);ks=LocalKeyStore(self.root/'keys')
        self.signer=signer or ks.load_or_create('evidence');self.verifier=verifier or ks.build_verifier();self.verifier.register(self.signer.key_id,self.signer.public_bytes())
        self.evidence=EvidenceLedger(self.root/'evidence.sqlite',self.signer,self.verifier);self.skills=SkillRegistry(self.root/'skills.sqlite');self.improvements=ImprovementLedger(self.root/'improvements.sqlite',self.signer,self.verifier);self.runtimes=RuntimeRegistry(self.root/'runtimes.sqlite');self.artifacts=ArtifactStore(self.root/'artifacts');self.beliefs=BeliefCompiler()
    def close(self):
        for x in (self.evidence,self.skills,self.improvements,self.runtimes): x.close()
    def status(self):
        ev=self.evidence.all();return {'evidence_records':len(ev),'evidence_chain_valid':self.evidence.verify_chain(),'beliefs':len(self.beliefs.compile(self.evidence.eligible('belief'))),'promoted_skills':len(self.skills.promoted()),'improvement_chain_valid':self.improvements.verify_chain(),'active_runtime':self.runtimes.current_digest()}
