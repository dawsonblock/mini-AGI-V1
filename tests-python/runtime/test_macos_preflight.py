from dataclasses import replace
import pytest
import kvcontinual.execution.macos_qualification_preflight as p
from kvcontinual.execution.platforms.macos import MacOSCapabilities
from kvcontinual.execution.hardware_qualification import HardwareRuntimeFingerprint

def caps(ok=True):
 return MacOSCapabilities(ok,"arm64" if ok else "x86_64",ok,"15.0","3.12",ok,True,True,ok,True,ok,True,True,"mps" if ok else "cpu")
def fp():
 return HardwareRuntimeFingerprint("Darwin","arm64","15","3.12","2","4","1","metal",True,True,"backend","Mac16,1","Apple M4",34359738368)
def test_preflight_pass_and_tamper(monkeypatch):
 monkeypatch.setattr(p,"detect_macos_capabilities",lambda:caps(True)); monkeypatch.setattr(p,"detect_hardware_runtime_fingerprint",lambda **k:fp())
 r=p.run_macos_qualification_preflight(backend_fingerprint="backend",require_mlx=True); assert r.passed; r.validate()
 with pytest.raises(ValueError): replace(r,backend_fingerprint="evil").validate()
def test_preflight_fails_off_mac(monkeypatch):
 monkeypatch.setattr(p,"detect_macos_capabilities",lambda:caps(False)); monkeypatch.setattr(p,"detect_hardware_runtime_fingerprint",lambda **k:fp())
 r=p.run_macos_qualification_preflight(backend_fingerprint="backend"); assert not r.passed
 with pytest.raises(ValueError): r.validate()
