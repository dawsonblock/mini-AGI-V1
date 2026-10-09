import tempfile
import unittest
from egai.common.artifacts import ArtifactStore
from egai.common.crypto import Ed25519Signer,Ed25519Verifier
from egai.common.trust import AuthorityTrust
from egai.authority.model import CandidateManifest
from egai.authority.build import BuildAuthority
from egai.authority.benchmark import BenchmarkRegistrar
from egai.authority.runner import BenchmarkRunner
from egai.bench.engine import TaskCase
class T(unittest.TestCase):
 def test_substituted_future_set_rejected(self):
  with tempfile.TemporaryDirectory() as d:
   store=ArtifactStore(d);bs=Ed25519Signer.generate('benchmark');bu=Ed25519Signer.generate('builder');rs=Ed25519Signer.generate('runner');v=Ed25519Verifier()
   for s in (bs,bu,rs):v.register(s.key_id,s.public_bytes())
   trust=AuthorityTrust({'benchmark':(bs.key_id,),'builder':(bu.key_id,),'runner':(rs.key_id,)})
   art=store.put_bytes(b'a');env=store.put_bytes(b'e');c=CandidateManifest('c','sha256:'+'0'*64,art);b=BuildAuthority('builder',bu,store).attest('b',c,env,art)
   original=[TaskCase('f','future','a','a'),TaskCase('r','retention','b','b')]
   bench=BenchmarkRegistrar('reg',bs,store).register_cases('bench','1',original)
   substituted=[TaskCase('f','future','easy','easy'),TaskCase('r','retention','b','b')]
   with self.assertRaises(ValueError):BenchmarkRunner('runner',rs,v,trust).run('run',b,bench,substituted,lambda x:x,lambda x:x)
