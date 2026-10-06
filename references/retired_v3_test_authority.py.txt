import tempfile,unittest
from dataclasses import replace
from egai.common.artifacts import ArtifactStore
from egai.common.crypto import Ed25519Signer,Ed25519Verifier
from egai.common.trust import AuthorityTrust
from egai.common.canonical import sha256_bytes
from egai.authority.model import CandidateManifest,BuildManifest
from egai.authority.build import BuildAuthority
from egai.authority.benchmark import BenchmarkRegistrar
from egai.authority.runner import BenchmarkRunner
from egai.authority.evaluate import IndependentEvaluator
from egai.authority.qualify import Qualifier,PromotionAuthority
from egai.authority.apply import PromotionApplier
from egai.authority.runtime_registry import RuntimeRegistry
from egai.bench.engine import TaskCase

class T(unittest.TestCase):
 def chain(self,security_bad=False):
  td=tempfile.TemporaryDirectory();self.addCleanup(td.cleanup);store=ArtifactStore(td.name+'/art');rr=RuntimeRegistry(td.name+'/runtime.db')
  roles={x:Ed25519Signer.generate(x) for x in ('benchmark','builder','runner','evaluator','qualifier','promotion')};v=Ed25519Verifier()
  for x in roles.values():v.register(x.key_id,x.public_bytes())
  trust=AuthorityTrust.from_signers(**roles)
  art=store.put_bytes(b'a');env=store.put_bytes(b'e');c=CandidateManifest('c','sha256:'+'0'*64,art);b=BuildAuthority('builder',roles['builder'],store).attest('b',c,env,art)
  cases=[TaskCase('f','future',1,1),TaskCase('r','retention',2,2),TaskCase('s','security',3,3)]
  bench=BenchmarkRegistrar('br',roles['benchmark'],store).register_cases('bench','1',cases)
  base=lambda x:0 if x==1 else x; cand=(lambda x:0 if security_bad and x==3 else x)
  rb=BenchmarkRunner('runner',roles['runner'],v,trust).run('run',b,bench,cases,base,cand,resource_fn=lambda *a:(1.,1.))
  e=IndependentEvaluator('evaluator',roles['evaluator'],v,trust).evaluate('ev',b,bench,rb,1.)
  q=Qualifier('qualifier',roles['qualifier'],v,trust).evaluate(e);d=PromotionAuthority('pa',roles['promotion'],v,trust).decide(q)
  return v,trust,store,rr,c,b,bench,rb,e,q,d
 def test_hard_security_gate(self):
  vals=self.chain(True);q,d=vals[-2],vals[-1];self.assertFalse(q.passed);self.assertFalse(d.approved)
 def test_valid_chain_activates_and_replay_fails(self):
  v,t,s,rr,c,b,bench,rb,e,q,d=self.chain(False);ap=PromotionApplier(v,t,rr,s);rt=ap.activate('rt','sha256:'+'1'*64,(b.output_digest,),c,b,bench,rb,e,q,d);self.assertEqual(rr.current_digest(),rt.digest)
  with self.assertRaises(PermissionError):ap.activate('rt2','sha256:'+'1'*64,(b.output_digest,),c,b,bench,rb,e,q,d)
 def test_binding_tamper_rejected(self):
  v,t,s,rr,c,b,bench,rb,e,q,d=self.chain(False);bad=replace(b,candidate_digest=sha256_bytes(b'wrong'))
  with self.assertRaises((ValueError,PermissionError)):PromotionApplier(v,t,rr,s).activate('rt','sha256:'+'1'*64,(b.output_digest,),c,bad,bench,rb,e,q,d)
 def test_unsigned_runner_results_rejected(self):
  v,t,s,rr,c,b,bench,rb,e,q,d=self.chain(False);bad=replace(rb,signature_b64='')
  with self.assertRaises(PermissionError):IndependentEvaluator('x',Ed25519Signer.generate('x'),v,t).evaluate('z',b,bench,bad,1.)
