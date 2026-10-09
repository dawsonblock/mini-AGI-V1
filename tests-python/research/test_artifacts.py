import tempfile
import unittest
from egai.common.artifacts import ArtifactStore
class T(unittest.TestCase):
 def test_roundtrip_and_parser(self):
  with tempfile.TemporaryDirectory() as d:
   s=ArtifactStore(d);h=s.put_bytes(b'abc');self.assertEqual(s.get_bytes(h),b'abc')
   for bad in ('../etc/passwd','sha256:abc','SHA256:'+'0'*64):
    with self.assertRaises(ValueError):s.get_bytes(bad)
