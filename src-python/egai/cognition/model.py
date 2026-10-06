from abc import ABC,abstractmethod
from dataclasses import dataclass
from pathlib import Path
import hashlib, json
from egai.common.canonical import digest,validate_digest

class FrozenModel(ABC):
    @property
    @abstractmethod
    def model_digest(self):...
    @abstractmethod
    def generate(self,prompt:str)->str:...

@dataclass(frozen=True)
class FrozenModelIdentity:
    model_name:str
    files:tuple[tuple[str,str],...]
    config_digest:str=''
    tokenizer_digest:str=''
    runtime_family:str='local'
    def __post_init__(self):
        for _,d in self.files: validate_digest(d)
        if self.config_digest: validate_digest(self.config_digest)
        if self.tokenizer_digest: validate_digest(self.tokenizer_digest)
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
    def from_files(cls,model_name,paths,config=None,tokenizer=None,runtime_family='local'):
        rows=[]
        for p in sorted(map(Path,paths),key=lambda x:str(x)):
            rows.append((p.name,cls._file_digest(p)))
        return cls(model_name,tuple(rows),digest(config or {}),digest(tokenizer or {}),runtime_family)
    @classmethod
    def from_directory(cls,model_name,directory,include_suffixes=('.safetensors','.json','.model','.txt'),runtime_family='local'):
        d=Path(directory); paths=[p for p in d.rglob('*') if p.is_file() and p.suffix in include_suffixes]
        if not paths: raise ValueError('no model identity files found')
        rows=tuple((str(p.relative_to(d)),cls._file_digest(p)) for p in sorted(paths))
        return cls(model_name,rows,'','',runtime_family)

class FrozenEchoModel(FrozenModel):
    def __init__(self,identity='frozen-echo-v1'):
        self.identity=identity;self._identity=FrozenModelIdentity(identity,(),digest({'identity':identity}),'','test')
    @property
    def model_digest(self):return self._identity.digest
    def generate(self,prompt):return 'FROZEN:'+prompt[:160]

class FrozenModelGuard:
    def __init__(self,model:FrozenModel): self.expected=model.model_digest; validate_digest(self.expected)
    def assert_unchanged(self,model:FrozenModel):
        if model.model_digest!=self.expected: raise RuntimeError('frozen model identity changed during experiment')
        return True
