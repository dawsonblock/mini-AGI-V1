from __future__ import annotations

class EvidenceAuthorityClientV143:
    __slots__=("__issue",)
    def __init__(self,authority): self.__issue=authority.issue
    def issue(self,**kwargs): return self.__issue(**kwargs)

class EvaluationAuthorityClientV143:
    __slots__=("__issue",)
    def __init__(self,authority): self.__issue=authority.issue
    def issue(self,**kwargs): return self.__issue(**kwargs)

class QualificationAuthorityClientV143:
    __slots__=("__qualify",)
    def __init__(self,authority): self.__qualify=authority.qualify
    def qualify(self,*args,**kwargs): return self.__qualify(*args,**kwargs)

class PromotionAuthorityClientV143:
    __slots__=("__authorize",)
    def __init__(self,authority): self.__authorize=authority.authorize
    def authorize(self,**kwargs): return self.__authorize(**kwargs)

class AuditAuthorityClientV143:
    """Client facade; private audit signer remains in the authority object, not runtime."""
    __slots__=("__checkpoint",)
    def __init__(self,*,db,signer,anchor,authority_generation:int):
        def checkpoint():
            cp=db.create_audit_checkpoint(signer=signer,authority_generation=int(authority_generation))
            return cp,anchor.write(event_seq=cp.event_seq,event_digest=cp.event_digest,checkpoint_digest=cp.digest,
                                   authority_generation=int(authority_generation))
        self.__checkpoint=checkpoint
    def checkpoint(self): return self.__checkpoint()

import json,socket

for _cls in (EvidenceAuthorityClientV143,EvaluationAuthorityClientV143,QualificationAuthorityClientV143,PromotionAuthorityClientV143,AuditAuthorityClientV143):
    _cls.is_local = True

class UnixJsonAuthorityClientV143:
    """Minimal transport client for running an authority under a separate UID/process.

    The authority protocol is intentionally generic: each server receives an op
    and canonical JSON-compatible payload and returns a JSON-compatible result.
    Production runtimes can therefore hold only socket paths and public keys.
    """
    is_local=False
    def __init__(self,socket_path:str,operation:str): self.socket_path=str(socket_path);self.operation=str(operation)
    def call(self,payload):
        req=json.dumps({"op":self.operation,"payload":payload},sort_keys=True,separators=(",",":")).encode()+b"\n"
        with socket.socket(socket.AF_UNIX,socket.SOCK_STREAM) as s:
            s.connect(self.socket_path);s.sendall(req);buf=b""
            while not buf.endswith(b"\n"):
                part=s.recv(65536)
                if not part:break
                buf+=part
        body=json.loads(buf.decode())
        if not body.get("ok"): raise PermissionError(body.get("error","authority RPC rejected"))
        return body.get("result")

from dataclasses import asdict
from .evidence_v143 import EvidenceStrengthProofV143
from .evaluation_v143 import EvaluationCaseResultV143,SignedEvaluationBundleV143
from .qualification_v143 import QualificationRecordV143
from .authority_v143 import MutationCommitmentV143,PromotionAuthorizationV143

class UnixEvidenceAuthorityClientV143(UnixJsonAuthorityClientV143):
    def __init__(self,socket_path): super().__init__(socket_path,"evidence.issue")
    def issue(self,**kwargs): return EvidenceStrengthProofV143(**self.call(kwargs))
class UnixEvaluationAuthorityClientV143(UnixJsonAuthorityClientV143):
    def __init__(self,socket_path): super().__init__(socket_path,"evaluation.issue")
    def issue(self,**kwargs):
        payload=dict(kwargs);payload["case_results"]=[asdict(x) for x in payload["case_results"]]
        body=self.call(payload);body["case_results"]=tuple(EvaluationCaseResultV143(**x) for x in body["case_results"])
        return SignedEvaluationBundleV143(**body)
class UnixQualificationAuthorityClientV143(UnixJsonAuthorityClientV143):
    def __init__(self,socket_path): super().__init__(socket_path,"qualification.qualify")
    def qualify(self,bundle,**kwargs):
        body=self.call({"bundle":asdict(bundle),**kwargs});body["reasons"]=tuple(body.get("reasons",()))
        return QualificationRecordV143(**body)
class UnixPromotionAuthorityClientV143(UnixJsonAuthorityClientV143):
    def __init__(self,socket_path): super().__init__(socket_path,"promotion.authorize")
    def authorize(self,**kwargs):
        payload=dict(kwargs);payload["qualification"]=asdict(payload["qualification"]);payload["mutation_commitments"]=[asdict(x) for x in payload["mutation_commitments"]]
        body=self.call(payload);body["mutation_commitments"]=tuple(MutationCommitmentV143(**x) for x in body["mutation_commitments"])
        return PromotionAuthorizationV143(**body)

def authority_handlers_v143(*,evidence_authority,evaluation_authority,qualification_authority,promotion_authority):
    def evidence(payload): return asdict(evidence_authority.issue(**payload))
    def evaluation(payload):
        payload=dict(payload);payload["case_results"]=tuple(EvaluationCaseResultV143(**x) for x in payload["case_results"])
        return asdict(evaluation_authority.issue(**payload))
    def qualification(payload):
        payload=dict(payload);b=payload.pop("bundle");b["case_results"]=tuple(EvaluationCaseResultV143(**x) for x in b["case_results"])
        return asdict(qualification_authority.qualify(SignedEvaluationBundleV143(**b),**payload))
    def promotion(payload):
        payload=dict(payload);q=payload.pop("qualification");q["reasons"]=tuple(q.get("reasons",()))
        payload["mutation_commitments"]=tuple(MutationCommitmentV143(**x) for x in payload["mutation_commitments"])
        return asdict(promotion_authority.authorize(qualification=QualificationRecordV143(**q),**payload))
    return {"evidence.issue":evidence,"evaluation.issue":evaluation,"qualification.qualify":qualification,"promotion.authorize":promotion}

class UnixAuthorityServerV143:
    """Small Unix-domain authority server. Run one authority set under a separate UID.

    For stronger separation deploy different operations on separate sockets/UIDs.
    Socket permissions default to owner-only (0600).
    """
    def __init__(self,socket_path,handlers): self.socket_path=str(socket_path);self.handlers=dict(handlers);self._server=None
    def serve_forever(self):
        import os,socketserver
        try: os.unlink(self.socket_path)
        except FileNotFoundError: pass
        handlers=self.handlers
        class H(socketserver.StreamRequestHandler):
            def handle(self):
                try:
                    req=json.loads(self.rfile.readline().decode());op=req.get("op");
                    if op not in handlers: raise PermissionError("unsupported authority operation")
                    result=handlers[op](req.get("payload",{}));resp={"ok":True,"result":result}
                except Exception as exc: resp={"ok":False,"error":str(exc)}
                self.wfile.write(json.dumps(resp,sort_keys=True,separators=(",",":")).encode()+b"\n")
        with socketserver.UnixStreamServer(self.socket_path,H) as srv:
            os.chmod(self.socket_path,0o600);self._server=srv;srv.serve_forever()
