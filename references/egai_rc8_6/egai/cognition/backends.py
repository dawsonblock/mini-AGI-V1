import json,os,subprocess,urllib.request,urllib.error,base64
from pathlib import Path
from egai.common.canonical import digest
from egai.common.crypto import Ed25519Verifier
from .model import FrozenModel,FrozenModelIdentity
from .attestation import HTTPRuntimeAttestation,verify_attestation

class FrozenHTTPChatModel(FrozenModel):
    """OpenAI-compatible local backend. Research mode can require a signed runtime attestation bound to exact model identity."""
    def __init__(self,identity,endpoint,served_model,api_key_env='',timeout=120.,temperature=0.0,max_tokens=2048,research_mode=True,seed=0,top_p=1.0,runtime_fingerprint='',attestation=None,attestation_verifier=None,trusted_attestation_keys=()):
        if research_mode and float(temperature)!=0.0:raise ValueError('research mode requires temperature=0')
        if research_mode and float(top_p)!=1.0:raise ValueError('research mode requires top_p=1')
        self.identity=identity;self.endpoint=endpoint.rstrip('/');self.served_model=served_model;self.api_key_env=api_key_env;self.timeout=float(timeout);self.temperature=float(temperature);self.max_tokens=int(max_tokens);self.research_mode=research_mode;self.seed=int(seed);self.top_p=float(top_p);self.runtime_fingerprint=str(runtime_fingerprint);self.attestation=attestation
        if attestation is not None:
            if attestation_verifier is None:raise ValueError('attestation verifier required')
            verify_attestation(attestation,attestation_verifier,trusted_attestation_keys,identity.digest,served_model)
            runtime_binding=attestation.digest
        else:
            if research_mode and not self.runtime_fingerprint:raise ValueError('research HTTP backend requires signed attestation or explicit legacy runtime fingerprint')
            runtime_binding=digest({'legacy_runtime_fingerprint':self.runtime_fingerprint})
        self._backend_digest=digest({'kind':'openai-compatible-local','served_model':served_model,'endpoint':self.endpoint,'seed':self.seed,'temperature':self.temperature,'top_p':self.top_p,'max_tokens':self.max_tokens,'runtime_binding':runtime_binding})
    @property
    def model_digest(self):return self.identity.digest
    @property
    def backend_digest(self):return self._backend_digest
    def _headers(self):
        h={'Content-Type':'application/json'}
        if self.api_key_env and os.getenv(self.api_key_env):h['Authorization']='Bearer '+os.environ[self.api_key_env]
        return h
    def generate(self,prompt):
        body={'model':self.served_model,'messages':[{'role':'user','content':prompt}],'temperature':self.temperature,'top_p':self.top_p,'max_tokens':self.max_tokens,'seed':self.seed}
        req=urllib.request.Request(self.endpoint+'/v1/chat/completions',json.dumps(body).encode(),headers=self._headers(),method='POST')
        try:
            with urllib.request.urlopen(req,timeout=self.timeout) as r:data=json.loads(r.read())
        except urllib.error.HTTPError as e:raise RuntimeError(f'local model backend HTTP {e.code}') from e
        if not data.get('choices') or 'message' not in data['choices'][0]:raise RuntimeError('malformed local model response')
        return str(data['choices'][0]['message'].get('content',''))

class LlamaCppCLIModel(FrozenModel):
    def __init__(self,executable,model_path,args=(),timeout=180.,seed=0,temperature=0.0,max_tokens=2048):
        self.executable=str(Path(executable).resolve());self.model_path=str(Path(model_path).resolve());self.args=tuple(args);self.timeout=float(timeout);self.seed=int(seed);self.temperature=float(temperature);self.max_tokens=int(max_tokens);ep=Path(self.executable);mp=Path(self.model_path)
        if not mp.is_file():raise FileNotFoundError(mp)
        g={'seed':self.seed,'temperature':self.temperature,'max_tokens':self.max_tokens,'args':self.args};self.identity=FrozenModelIdentity.from_files(mp.name,[mp],config={'backend':'llama.cpp','executable':str(ep)},runtime_family='llama.cpp',runtime_path=ep if ep.is_file() else None,generation_config=g);self._backend_digest=digest({'runtime_digest':self.identity.runtime_digest,'generation_config_digest':self.identity.generation_config_digest})
    @property
    def model_digest(self):return self.identity.digest
    @property
    def backend_digest(self):return self._backend_digest
    def generate(self,prompt):
        cmd=[self.executable,'-m',self.model_path,'-p',prompt,'--no-display-prompt','--seed',str(self.seed),'--temp',str(self.temperature),'-n',str(self.max_tokens),*self.args];cp=subprocess.run(cmd,capture_output=True,text=True,timeout=self.timeout,check=True);return cp.stdout.strip()

def identity_from_manifest(path):
    d=json.loads(Path(path).read_text());return FrozenModelIdentity(d['model_name'],tuple(tuple(x) for x in d['files']),d.get('config_digest',''),d.get('tokenizer_digest',''),d.get('runtime_family','local'),d.get('runtime_digest',''),d.get('generation_config_digest',''))

def _attestation_from_config(c):
    path=c.get('attestation_file')
    if not path:return None,None,()
    d=json.loads(Path(path).read_text());a=HTTPRuntimeAttestation(d['served_model'],d['model_digest'],d['runtime_digest'],d['backend_version'],d['issuer_id'],d['issuer_key_id'],d['signature_b64']);v=Ed25519Verifier();pub=base64.b64decode(d['issuer_public_key_b64']);v.register(a.issuer_key_id,pub);return a,v,(a.issuer_key_id,)

def load_backend(config_path):
    c=json.loads(Path(config_path).read_text());kind=c['backend']
    if kind=='http':
        a,v,keys=_attestation_from_config(c)
        return FrozenHTTPChatModel(identity_from_manifest(c['model_manifest']),c['endpoint'],c['served_model'],c.get('api_key_env',''),float(c.get('timeout',120)),float(c.get('temperature',0)),int(c.get('max_tokens',2048)),True,int(c.get('seed',0)),float(c.get('top_p',1.0)),c.get('runtime_fingerprint',''),a,v,keys)
    if kind=='llama_cpp_cli':return LlamaCppCLIModel(c['executable'],c['model_path'],tuple(c.get('args',[])),float(c.get('timeout',180)),int(c.get('seed',0)),float(c.get('temperature',0)),int(c.get('max_tokens',2048)))
    raise ValueError('unsupported backend: '+kind)
