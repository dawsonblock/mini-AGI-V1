from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

from kvcontinual.execution.hardware_qualification import detect_hardware_runtime_fingerprint
from kvcontinual.execution.platforms.macos import detect_macos_capabilities


def _digest(payload: Any) -> str:
    raw=json.dumps(payload,sort_keys=True,separators=(",",":"),allow_nan=False).encode()
    return "sha256:"+hashlib.sha256(raw).hexdigest()

@dataclass(frozen=True)
class PreflightCheck:
    name: str
    passed: bool
    detail: str

@dataclass(frozen=True)
class MacQualificationPreflight:
    version: int
    created_at: str
    backend_fingerprint: str
    hardware_runtime_digest: str
    checks: tuple[PreflightCheck,...]
    report_digest: str=""

    def payload(self) -> dict[str,Any]:
        return {"version":self.version,"created_at":self.created_at,"backend_fingerprint":self.backend_fingerprint,
                "hardware_runtime_digest":self.hardware_runtime_digest,"checks":[asdict(x) for x in self.checks]}
    def expected_digest(self)->str: return _digest(self.payload())
    @property
    def passed(self)->bool: return bool(self.checks) and all(x.passed for x in self.checks)
    def validate(self)->None:
        if self.version!=1: raise ValueError("unsupported preflight version")
        if self.report_digest!=self.expected_digest(): raise ValueError("preflight digest mismatch")
        if not self.passed: raise ValueError("Mac qualification preflight did not pass")
    def to_dict(self)->dict[str,Any]: return {**self.payload(),"passed":self.passed,"report_digest":self.report_digest}


def run_macos_qualification_preflight(*,backend_fingerprint:str, require_mlx:bool=False)->MacQualificationPreflight:
    if not backend_fingerprint: raise ValueError("backend_fingerprint is required")
    c=detect_macos_capabilities()
    fp=detect_hardware_runtime_fingerprint(backend_fingerprint=backend_fingerprint)
    checks=[
      PreflightCheck("darwin",c.is_macos,"Darwin required for hardware qualification"),
      PreflightCheck("apple_silicon",c.apple_silicon,f"machine={c.machine}"),
      PreflightCheck("native_arm64_python",c.native_arm64_python,f"python={c.python_version}"),
      PreflightCheck("torch",c.torch_installed,"PyTorch required"),
      PreflightCheck("mps_built",c.torch_mps_built,"PyTorch must be built with MPS"),
      PreflightCheck("mps_available",c.torch_mps_available,"MPS must be available on target host"),
      PreflightCheck("clang",c.clang_available,"clang++ required for native references"),
      PreflightCheck("cmake",c.cmake_available,"CMake required for native build"),
      PreflightCheck("metal_toolchain",c.metal_compiler_available and fp.metal_toolchain!="unavailable",fp.metal_toolchain),
    ]
    if require_mlx: checks.append(PreflightCheck("mlx",c.mlx_installed,"MLX required by requested backend"))
    r=MacQualificationPreflight(1,datetime.now(timezone.utc).isoformat(),backend_fingerprint,fp.digest,tuple(checks))
    return MacQualificationPreflight(**{**asdict(r),"checks":r.checks,"report_digest":r.expected_digest()})


def write_preflight(report:MacQualificationPreflight,path:str|Path)->None:
    p=Path(path); p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+".tmp")
    tmp.write_text(json.dumps(report.to_dict(),indent=2,sort_keys=True)+"\n",encoding="utf-8")
    tmp.replace(p)
