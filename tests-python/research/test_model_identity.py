import tempfile
import unittest
from pathlib import Path
from egai.cognition.model import FrozenModelIdentity
class T(unittest.TestCase):
 def test_directory_identity_changes_with_file(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'model.safetensors';p.write_bytes(b'a');a=FrozenModelIdentity.from_directory('m',d).digest
   p.write_bytes(b'b');b=FrozenModelIdentity.from_directory('m',d).digest
   self.assertNotEqual(a,b)
