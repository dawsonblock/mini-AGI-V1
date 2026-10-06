import json, os, subprocess, urllib.request
from pathlib import Path
from dataclasses import dataclass
from .model import FrozenModel, FrozenModelIdentity

class FrozenHTTPChatModel(FrozenModel):
    """OpenAI-compatible local HTTP backend. Model identity is supplied independently of the endpoint."""
    def __init__(self,identity:FrozenModelIdentity,endpoint,served_model,api_key_env='',timeout=120.,temperature=0.0,max_tokens=2048):
        self.identity=identity;self.endpoint=endpoint.rstrip('/');self.served_model=served_model
        self.api_key_env=api_key_env;self.timeout=timeout;self.temperature=temperature;self.max_tokens=max_tokens
    @property
    def model_digest(self):return self.identity.digest
    def generate(self,prompt):
        body=json.dumps({'model':self.served_model,'messages':[{'role':'user','content':prompt}],
                         'temperature':self.temperature,'max_tokens':self.max_tokens}).encode()
        headers={'Content-Type':'application/json'}
        if self.api_key_env and os.getenv(self.api_key_env): headers['Authorization']='Bearer '+os.environ[self.api_key_env]
        req=urllib.request.Request(self.endpoint+'/v1/chat/completions',body,headers=headers,method='POST')
        with urllib.request.urlopen(req,timeout=self.timeout) as r: data=json.loads(r.read())
        return data['choices'][0]['message']['content']

class LlamaCppCLIModel(FrozenModel):
    """Local llama.cpp CLI backend for macOS/Linux. The GGUF file digest is part of the immutable model identity."""
    def __init__(self,executable,model_path,args=(),timeout=180.):
        self.executable=str(executable);self.model_path=str(model_path);self.args=tuple(args);self.timeout=timeout
        p=Path(model_path)
        if not p.is_file(): raise FileNotFoundError(p)
        self.identity=FrozenModelIdentity.from_files(p.name,[p],config={'backend':'llama.cpp','args':self.args},runtime_family='llama.cpp')
    @property
    def model_digest(self):
        # Rehash the actual checkpoint rather than trusting a cached identity label.
        p=Path(self.model_path)
        return FrozenModelIdentity.from_files(p.name,[p],config={'backend':'llama.cpp','args':self.args},runtime_family='llama.cpp').digest
    def generate(self,prompt):
        cmd=[self.executable,'-m',self.model_path,'-p',prompt,'--no-display-prompt',*self.args]
        cp=subprocess.run(cmd,capture_output=True,text=True,timeout=self.timeout,check=True)
        return cp.stdout.strip()

def identity_from_manifest(path):
    d=json.loads(Path(path).read_text())
    return FrozenModelIdentity(d['model_name'],tuple(tuple(x) for x in d['files']),d.get('config_digest',''),d.get('tokenizer_digest',''),d.get('runtime_family','local'))

def load_backend(config_path):
    c=json.loads(Path(config_path).read_text());kind=c['backend']
    if kind=='http':
        return FrozenHTTPChatModel(identity_from_manifest(c['model_manifest']),c['endpoint'],c['served_model'],c.get('api_key_env',''),float(c.get('timeout',120)),float(c.get('temperature',0)),int(c.get('max_tokens',2048)))
    if kind=='llama_cpp_cli':
        return LlamaCppCLIModel(c['executable'],c['model_path'],tuple(c.get('args',[])),float(c.get('timeout',180)))
    raise ValueError('unsupported backend: '+kind)
