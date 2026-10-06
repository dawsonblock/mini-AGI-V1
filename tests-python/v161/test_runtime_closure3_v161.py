import pytest
from egai.common.canonical import digest
from minagi.v161.runtime_closure3 import RuntimeArtifactSpec, RuntimeClosureV161, sha256_path

def test_physical_closure_detects_substitution(tmp_path):
    p=tmp_path/'adapter.bin'; p.write_bytes(b'abc')
    spec=RuntimeArtifactSpec('adapter',str(p),sha256_path(p))
    d=digest({'x':1})
    c=RuntimeClosureV161(d,d,d,(spec,),d,d)
    assert c.verify_physical()['adapter']==sha256_path(p)
    p.write_bytes(b'abd')
    with pytest.raises(PermissionError): c.verify_physical()
