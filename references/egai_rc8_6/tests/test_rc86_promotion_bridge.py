import tempfile,unittest
from dataclasses import asdict,replace
from pathlib import Path
from egai.common.crypto import Ed25519Signer,Ed25519Verifier
from egai.common.trust import AuthorityTrust
from egai.common.artifacts import ArtifactStore
from egai.authority.runtime_registry import RuntimeRegistry
from egai.experiment.qualify import ResearchQualification
from egai.integration.promotion import GovernedPromotionPipeline
from egai.integration.learning_artifacts import LearningArtifactPackage
from egai.skills.model import SkillManifest
from egai.bench.engine import TaskCase
class T(unittest.TestCase):
 def test_untrusted_research_qualification_rejected_and_valid_one_requalified(self):
  with tempfile.TemporaryDirectory() as d:
   store=ArtifactStore(Path(d)/'cas');rr=RuntimeRegistry(Path(d)/'r.db');roles={x:Ed25519Signer.generate(x) for x in ('benchmark','builder','runner','evaluator','qualifier','promotion')};research=Ed25519Signer.generate('research');bad=Ed25519Signer.generate('bad');v=Ed25519Verifier()
   for s in [*roles.values(),research,bad]:v.register(s.key_id,s.public_bytes())
   trust=AuthorityTrust.from_signers(**roles);pipe=GovernedPromotionPipeline(store,trust,v,roles,rr,(research.key_id,))
   skill=SkillManifest('s','1','reverse',('reverse',),(),(),('task',),('answer',),{'handler':'model_procedure'},supporting_evidence=('sha256:'+'1'*64,'sha256:'+'2'*64))
   pkg=LearningArtifactPackage(skill,'sha256:'+'3'*64,'sha256:'+'4'*64,'sha256:'+'5'*64,'sha256:'+'6'*64,'sha256:'+'7'*64)
   def rq(signer):
    u=ResearchQualification('rq','sha256:'+'8'*64,True,(),'rq');e=signer.sign(asdict(u));return replace(u,qualifier_key_id=e.key_id,signature_b64=e.signature_b64)
   cases=[TaskCase('f','future','a','a'),TaskCase('r','retention','b','b'),TaskCase('s','security','c','c')]
   with self.assertRaises(PermissionError):pipe.promote(pkg,rq(bad),'sha256:'+'9'*64,cases,lambda x:'X',lambda x:x)
   out=pipe.promote(pkg,rq(research),'sha256:'+'9'*64,cases,lambda x:'X' if x=='a' else x,lambda x:x)
   self.assertEqual(rr.current_digest(),out.runtime_digest)
