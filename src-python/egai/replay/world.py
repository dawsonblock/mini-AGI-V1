from dataclasses import dataclass,asdict,replace
from egai.common.canonical import digest
from egai.common.crypto import SignedEnvelope

@dataclass(frozen=True)
class ReplayNode:
    node_id:str;t:int;observation:dict;action:dict;outcome:dict;parent_id:str|None=None
@dataclass(frozen=True)
class ReplayCapability:
    world_id:str;max_t:int;visible_prefix_digest:str;issuer_key_id:str='';signature_b64:str=''
    def unsigned(self):return replace(self,issuer_key_id='',signature_b64='')
@dataclass(frozen=True)
class GroundedReplayResult:
    grounded:bool; node:ReplayNode|None
@dataclass(frozen=True)
class ImaginedResult:
    grounded:bool; prediction:dict; predictor_digest:str; uncertainty:float

class ReplayWorld:
    def __init__(self,world_id,nodes,signer,verifier):self.world_id=world_id;self.nodes=sorted(nodes,key=lambda n:n.t);self.signer=signer;self.verifier=verifier
    def issue_capability(self,max_t):
        visible=[asdict(n) for n in self.nodes if n.t<=max_t];pd=digest(visible)
        u=ReplayCapability(self.world_id,max_t,pd);env=self.signer.sign(asdict(u));return replace(u,issuer_key_id=env.key_id,signature_b64=env.signature_b64)
    def view(self,cap):
        if cap.world_id!=self.world_id:raise PermissionError('wrong world')
        if not self.verifier.verify(asdict(cap.unsigned()),SignedEnvelope(cap.issuer_key_id,cap.signature_b64)):raise PermissionError('invalid replay capability')
        visible=[n for n in self.nodes if n.t<=cap.max_t]
        if digest([asdict(n) for n in visible])!=cap.visible_prefix_digest:raise PermissionError('prefix changed')
        return visible
    def grounded_transition(self,node_id,action,cap):
        visible=self.view(cap);matches=[n for n in visible if n.parent_id==node_id and n.action==action]
        return GroundedReplayResult(True,matches[0]) if matches else GroundedReplayResult(False,None)
