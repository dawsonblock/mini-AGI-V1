import tempfile, unittest
from pathlib import Path
from egai.cognition.backends import LlamaCppCLIModel

class T(unittest.TestCase):
 def test_generation_config_changes_identity(self):
  with tempfile.TemporaryDirectory() as d:
   m=Path(d)/'m.gguf'; exe=Path(d)/'llama'; m.write_bytes(b'w'); exe.write_bytes(b'bin')
   a=LlamaCppCLIModel(exe,m,seed=0); b=LlamaCppCLIModel(exe,m,seed=1)
   self.assertNotEqual(a.model_digest,b.model_digest)
   self.assertNotEqual(a.backend_digest,b.backend_digest)

 def test_runtime_binary_changes_identity(self):
  with tempfile.TemporaryDirectory() as d:
   m=Path(d)/'m.gguf'; exe=Path(d)/'llama'; m.write_bytes(b'w'); exe.write_bytes(b'bin1')
   a=LlamaCppCLIModel(exe,m); exe.write_bytes(b'bin2'); b=LlamaCppCLIModel(exe,m)
   self.assertNotEqual(a.model_digest,b.model_digest)
