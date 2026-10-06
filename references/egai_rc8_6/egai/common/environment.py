from dataclasses import dataclass
import platform,sys,importlib.metadata,os,subprocess
from egai.common.canonical import digest
@dataclass(frozen=True)
class EnvironmentFingerprint:
    python_version:str;implementation:str;platform:str;machine:str;processor:str
    packages:tuple[tuple[str,str],...];runtime_notes:tuple[tuple[str,str],...]=();selected_env:tuple[tuple[str,str],...]=();tool_versions:tuple[tuple[str,str],...]=()
    @property
    def digest(self):return digest(self)
    @classmethod
    def capture(cls,package_names=('cryptography',),runtime_notes=None,env_names=('PYTHONHASHSEED','CUDA_VISIBLE_DEVICES','METAL_DEVICE_WRAPPER_TYPE'),tools=()):
        rows=[]
        for name in sorted(set(package_names)):
            try:rows.append((name,importlib.metadata.version(name)))
            except importlib.metadata.PackageNotFoundError:rows.append((name,'missing'))
        env=tuple((n,os.environ.get(n,'')) for n in sorted(set(env_names)));tv=[]
        for tool in sorted(set(tools)):
            try:r=subprocess.run([tool,'--version'],capture_output=True,text=True,timeout=5);tv.append((tool,(r.stdout or r.stderr).strip()[:300]))
            except Exception:tv.append((tool,'unavailable'))
        return cls(sys.version.split()[0],platform.python_implementation(),platform.platform(),platform.machine(),platform.processor(),tuple(rows),tuple(sorted((runtime_notes or {}).items())),env,tuple(tv))
