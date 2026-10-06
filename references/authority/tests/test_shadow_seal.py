from pathlib import Path
from minagi.shadow_live import ShadowLearningBuffer

class Tok:
    class E:
        def __init__(self, ids): self.ids=ids
    def encode(self,s): return self.E(list(s.encode()))

def test_shadow_batch_is_immutable_snapshot(tmp_path):
    b=ShadowLearningBuffer(tmp_path/'shadow')
    b.feed('one',Tok()); s1=b.seal()
    first=Path(s1['path']).read_bytes()
    b.feed('two',Tok()); s2=b.seal()
    assert s1['sha256'] != s2['sha256']
    assert Path(s1['path']).read_bytes() == first
