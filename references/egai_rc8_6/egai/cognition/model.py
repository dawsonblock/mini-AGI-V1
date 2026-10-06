from abc import ABC,abstractmethod
from dataclasses import dataclass
from pathlib import Path
import hashlib
from egai.common.canonical import digest,validate_digest

class FrozenModel(ABC):
    @property
    @abstractmethod
    def model_digest(self):...
    @property
    def backend_digest(self): return digest({'backend':type(self).__name__})
    @abstractmethod
    def generate(self,prompt:str)->str:...

@dataclass(frozen=True)
class FrozenModelIdentity:
    model_name:str
    files:tuple[tuple[str,str],...]
    config_digest:str=''
    tokenizer_digest:str=''
    runtime_family:str='local'
    runtime_digest:str=''
    generation_config_digest:str=''
    def __post_init__(self):
        for _,d in self.files: validate_digest(d)
        for d in (self.config_digest,self.tokenizer_digest,self.runtime_digest,self.generation_config_digest):
            if d: validate_digest(d)
    @property
    def digest(self):return digest(self)
    @staticmethod
    def _file_digest(path,chunk_size=1024*1024):
        h=hashlib.sha256()
        with Path(path).open('rb') as f:
            while True:
                b=f.read(chunk_size)
                if not b: break
                h.update(b)
        return 'sha256:'+h.hexdigest()
    @classmethod
    def from_files(cls,model_name,paths,config=None,tokenizer=None,runtime_family='local',runtime_path=None,generation_config=None):
        rows=[]
        for p in sorted(map(Path,paths),key=lambda x:str(x)):
            rows.append((p.name,cls._file_digest(p)))
        runtime_digest=cls._file_digest(runtime_path) if runtime_path and Path(runtime_path).is_file() else digest({'runtime_family':runtime_family})
        return cls(model_name,tuple(rows),digest(config or {}),digest(tokenizer or {}),runtime_family,runtime_digest,digest(generation_config or {}))
    @classmethod
    def from_directory(cls,model_name,directory,include_suffixes=('.safetensors','.json','.model','.txt','.gguf'),runtime_family='local',runtime_path=None,generation_config=None):
        d=Path(directory); paths=[p for p in d.rglob('*') if p.is_file() and p.suffix in include_suffixes]
        if not paths: raise ValueError('no model identity files found')
        rows=tuple((str(p.relative_to(d)),cls._file_digest(p)) for p in sorted(paths))
        runtime_digest=cls._file_digest(runtime_path) if runtime_path and Path(runtime_path).is_file() else digest({'runtime_family':runtime_family})
        return cls(model_name,rows,'','',runtime_family,runtime_digest,digest(generation_config or {}))

class FrozenEchoModel(FrozenModel):
    def __init__(self,identity='frozen-echo-v1'):
        self.identity=identity;self._identity=FrozenModelIdentity(identity,(),digest({'identity':identity}),'','test',digest({'runtime':'echo'}),digest({'deterministic':True}))
    @property
    def model_digest(self):return self._identity.digest
    @property
    def backend_digest(self): return self._identity.runtime_digest
    def generate(self,prompt):return 'FROZEN:'+prompt[:160]

class FrozenModelGuard:
    def __init__(self,model:FrozenModel): self.expected=model.model_digest; self.backend_expected=model.backend_digest; validate_digest(self.expected); validate_digest(self.backend_expected)
    def assert_unchanged(self,model:FrozenModel):
        if model.model_digest!=self.expected: raise RuntimeError('frozen model identity changed during experiment')
        if model.backend_digest!=self.backend_expected: raise RuntimeError('frozen backend identity changed during experiment')
        return True
