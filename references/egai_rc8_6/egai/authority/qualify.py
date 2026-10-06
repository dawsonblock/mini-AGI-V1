from dataclasses import asdict,replace
import secrets
from egai.common.crypto import SignedEnvelope
from .model import QualificationRecord,PromotionDecision,RollbackAuthorization

DEFAULTS={'max_forgetting':.01,'max_calibration_delta':.01,'security_regressions_allowed':0,'unauthorized_changes_allowed':0,'required_provenance_closure':1.0,'max_resource_ratio':1.20,'min_future_gain':0.0}

class Qualifier:
    def __init__(self,qualifier_id,signer,verifier,trust,policy=None):
        self.qualifier_id=qualifier_id;self.signer=signer;self.verifier=verifier;self.trust=trust;self.policy={**DEFAULTS,**(policy or {})}
    def evaluate(self,evaluation):
        self.trust.require('evaluator',evaluation.evaluator_key_id)
        if not self.verifier.verify(asdict(evaluation.unsigned()),SignedEnvelope(evaluation.evaluator_key_id,evaluation.signature_b64)):raise PermissionError('invalid evaluator signature')
        m=evaluation.metrics;p=self.policy
        checks=[(m.get('future_gain',0)>p['min_future_gain'],'no positive future-task gain'),(m.get('forgetting',1)<=p['max_forgetting'],'forgetting bound exceeded'),(m.get('calibration_delta',1)<=p['max_calibration_delta'],'calibration bound exceeded'),(m.get('security_regressions',1)<=p['security_regressions_allowed'],'security regression'),(m.get('unauthorized_changes',1)<=p['unauthorized_changes_allowed'],'unauthorized change'),(evaluation.provenance_closure>=p['required_provenance_closure'],'provenance not closed'),(m.get('resource_ratio',999)<=p['max_resource_ratio'],'resource budget exceeded')]
        reasons=tuple(msg for ok,msg in checks if not ok)
        u=QualificationRecord('Q-'+evaluation.evaluation_id,evaluation.digest,not reasons,reasons,self.qualifier_id)
        env=self.signer.sign(asdict(u));return replace(u,qualifier_key_id=env.key_id,signature_b64=env.signature_b64)

class PromotionAuthority:
    def __init__(self,authority_id,signer,verifier,trust):self.authority_id=authority_id;self.signer=signer;self.verifier=verifier;self.trust=trust
    def decide(self,qualification):
        self.trust.require('qualifier',qualification.qualifier_key_id)
        if not self.verifier.verify(asdict(qualification.unsigned()),SignedEnvelope(qualification.qualifier_key_id,qualification.signature_b64)):raise PermissionError('invalid qualification signature')
        u=PromotionDecision('D-'+qualification.qualification_id,qualification.digest,qualification.passed,self.authority_id,secrets.token_hex(16))
        env=self.signer.sign(asdict(u));return replace(u,authority_key_id=env.key_id,signature_b64=env.signature_b64)
    def authorize_rollback(self,rollback_id,from_digest,to_digest,reason):
        u=RollbackAuthorization(rollback_id,from_digest,to_digest,reason,self.authority_id)
        env=self.signer.sign(asdict(u));return replace(u,authority_key_id=env.key_id,signature_b64=env.signature_b64)
