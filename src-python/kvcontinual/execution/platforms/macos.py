from __future__ import annotations

from dataclasses import asdict, dataclass
import importlib.util
import platform
import shutil
import subprocess
import sys
from typing import Any


@dataclass(frozen=True)
class MacOSCapabilities:
    is_macos: bool
    machine: str
    apple_silicon: bool
    macos_version: str
    python_version: str
    native_arm64_python: bool
    clang_available: bool
    cmake_available: bool
    metal_compiler_available: bool
    torch_installed: bool
    torch_mps_built: bool
    torch_mps_available: bool
    mlx_installed: bool
    recommended_device: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _macos_version() -> str:
    if platform.system() != "Darwin":
        return ""
    try:
        return subprocess.check_output(["sw_vers", "-productVersion"], text=True).strip()
    except Exception:
        return platform.mac_ver()[0]


def detect_macos_capabilities() -> MacOSCapabilities:
    is_macos = platform.system() == "Darwin"
    machine = platform.machine().lower()
    apple_silicon = is_macos and machine in {"arm64", "aarch64"}
    torch_installed = importlib.util.find_spec("torch") is not None
    mps_built = False
    mps_available = False
    if torch_installed:
        try:
            import torch
            mps_built = bool(torch.backends.mps.is_built())
            mps_available = bool(torch.backends.mps.is_available())
        except Exception:
            pass
    mlx_installed = importlib.util.find_spec("mlx") is not None
    if mps_available:
        device = "mps"
    else:
        device = "cpu"
    return MacOSCapabilities(
        is_macos=is_macos,
        machine=machine,
        apple_silicon=apple_silicon,
        macos_version=_macos_version(),
        python_version=platform.python_version(),
        native_arm64_python=apple_silicon and sys.maxsize > 2**32,
        clang_available=shutil.which("clang++") is not None,
        cmake_available=shutil.which("cmake") is not None,
        metal_compiler_available=shutil.which("xcrun") is not None if is_macos else False,
        torch_installed=torch_installed,
        torch_mps_built=mps_built,
        torch_mps_available=mps_available,
        mlx_installed=mlx_installed,
        recommended_device=device,
    )
