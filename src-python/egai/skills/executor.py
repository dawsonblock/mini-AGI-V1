from dataclasses import dataclass
import time

@dataclass(frozen=True)
class SkillExecutionResult:
    ok:bool; output:object; verification:bool; elapsed_ms:float; error:str=''

class SkillExecutor:
    """Executes only pre-registered handlers; manifests cannot inject arbitrary Python/shell."""
    def __init__(self): self.handlers={}; self.verifiers={}
    def register_handler(self,name,fn): self.handlers[name]=fn
    def register_verifier(self,name,fn): self.verifiers[name]=fn
    def execute(self,manifest,inputs,granted_permissions=()):
        required=set(manifest.permissions)
        if not required.issubset(set(granted_permissions)):
            return SkillExecutionResult(False,None,False,0,'permission denied')
        kind=manifest.implementation.get('handler')
        fn=self.handlers.get(kind)
        if not fn:return SkillExecutionResult(False,None,False,0,'unknown handler')
        start=time.perf_counter()
        try: out=fn(inputs)
        except Exception as e:return SkillExecutionResult(False,None,False,(time.perf_counter()-start)*1000,str(e))
        vname=manifest.verifier.get('handler'); vf=self.verifiers.get(vname)
        verified=bool(vf(inputs,out)) if vf else False
        return SkillExecutionResult(verified,out,verified,(time.perf_counter()-start)*1000,'' if verified else 'verification failed')
