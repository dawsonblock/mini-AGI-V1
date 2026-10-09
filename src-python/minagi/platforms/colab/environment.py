from __future__ import annotations
from dataclasses import dataclass
import importlib.metadata
import os
import platform
import shutil
import sys
from egai.common.canonical import digest

@dataclass(frozen=True)
class ColabEnvironment:
    python: str
    platform: str
    machine: str
    is_colab: bool
    torch_version: str
    cuda_available: bool
    cuda_version: str
    gpu_name: str
    gpu_vram_bytes: int
    ram_bytes: int
    disk_free_bytes: int
    package_digest: str
    schema: str = "mini-agi-v16.1-colab-environment-v1"
    @property
    def digest(self): return digest(self)


def _packages_digest() -> str:
    rows=[]
    for d in importlib.metadata.distributions():
        name=(d.metadata.get("Name") or "").lower()
        if name: rows.append((name,d.version))
    return digest(sorted(set(rows)))


def _ram_bytes() -> int:
    try:
        pages=os.sysconf("SC_PHYS_PAGES"); size=os.sysconf("SC_PAGE_SIZE"); return int(pages*size)
    except Exception: return 0


def probe_environment(workdir: str = "/content") -> ColabEnvironment:
    torch_version="unavailable"; cuda=False; cuda_version=""; gpu=""; vram=0
    try:
        import torch
        torch_version=str(torch.__version__); cuda=bool(torch.cuda.is_available()); cuda_version=str(torch.version.cuda or "")
        if cuda:
            p=torch.cuda.get_device_properties(0); gpu=str(p.name); vram=int(p.total_memory)
    except Exception: pass
    disk_root=workdir if os.path.exists(workdir) else "/"
    return ColabEnvironment(
        python=sys.version.split()[0], platform=platform.platform(), machine=platform.machine(),
        is_colab="COLAB_RELEASE_TAG" in os.environ or os.path.exists("/content"),
        torch_version=torch_version, cuda_available=cuda, cuda_version=cuda_version,
        gpu_name=gpu, gpu_vram_bytes=vram, ram_bytes=_ram_bytes(),
        disk_free_bytes=shutil.disk_usage(disk_root).free, package_digest=_packages_digest(),
    )
