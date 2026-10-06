import tempfile,unittest
from egai.common.artifacts import ArtifactStore
from egai.bench.engine import TaskCase
from egai.bench.dataset import persist_benchmark_sets,decode_cases
class T(unittest.TestCase):
 def test_signed_material_can_be_exactly_reloaded(self):
  with tempfile.TemporaryDirectory() as d:
   s=ArtifactStore(d);cases=[TaskCase('f','future','x','y'),TaskCase('r','retention','a','b')]
   fd,rd,sd=persist_benchmark_sets(s,cases)
   self.assertEqual(decode_cases(s.get_bytes(fd))[0].case_id,'f')
   self.assertEqual(decode_cases(s.get_bytes(rd))[0].case_id,'r')
   self.assertEqual(decode_cases(s.get_bytes(sd)),[])
