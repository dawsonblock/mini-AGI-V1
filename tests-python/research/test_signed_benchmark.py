import tempfile,unittest
from egai.common.artifacts import ArtifactStore
from egai.common.crypto import Ed25519Signer,Ed25519Verifier
from egai.common.trust import AuthorityTrust
from egai.authority.benchmark import BenchmarkRegistrar
from egai.authority.build import BuildAuthority
from egai.authority.model import CandidateManifest,BenchmarkSpec
from egai.authority.runner import BenchmarkRunner
from egai.bench.engine import TaskCase
class T(unittest.TestCase):
 def test_unsigned_benchmark_rejected(self):
  with tempfile.TemporaryDirectory() as d:
   s=ArtifactStore(d);ba=Ed25519Signer.generate('bench');bu=Ed25519Signer.generate('builder');ru=Ed25519Signer.generate('runner')
   v=Ed25519Verifier()
   for x in (ba,bu,ru):v.register(x.key_id,x.public_bytes())
   trust=AuthorityTrust({'benchmark':(ba.key_id,),'builder':(bu.key_id,),'runner':(ru.key_id,)})
   art=s.put_bytes(b'a');env=s.put_bytes(b'e');c=CandidateManifest('c','sha256:'+'0'*64,art);b=BuildAuthority('b',bu,s).attest('b',c,env,art)
   cases=[TaskCase('f','future',1,1),TaskCase('r','retention',2,2)]
   signed=BenchmarkRegistrar('reg',ba,s).register_cases('x','1',cases)
   unsigned=BenchmarkSpec(signed.benchmark_id,signed.version,signed.task_set_digest,signed.retention_set_digest,signed.security_set_digest,signed.scorer_id,signed.harness_version,True)
   with self.assertRaises(PermissionError):BenchmarkRunner('r',ru,v,trust).run('x',b,unsigned,cases,lambda x:x,lambda x:x)
