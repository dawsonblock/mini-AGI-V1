import tempfile
import unittest
from pathlib import Path
from egai.cognition.backends import LlamaCppCLIModel
class T(unittest.TestCase):
 def test_llama_identity_hashes_model_file(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'m.gguf';p.write_bytes(b'weights')
   m=LlamaCppCLIModel('/not/run',p)
   before=m.model_digest;p.write_bytes(b'changed');m2=LlamaCppCLIModel('/not/run',p)
   self.assertNotEqual(before,m2.model_digest)
