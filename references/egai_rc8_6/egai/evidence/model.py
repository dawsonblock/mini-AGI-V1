from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timezone
from typing import Any

class Origin(str,Enum):
    ENVIRONMENT="environment"; DETERMINISTIC_TOOL="deterministic_tool"; EXTERNAL_SOURCE="external_source"
    HUMAN="human"; MODEL_INFERENCE="model_inference"; SIMULATION="simulation"; REPLAY="replay"
class Verification(str,Enum): VERIFIED="verified"; UNVERIFIED="unverified"; REJECTED="rejected"

@dataclass(frozen=True)
class EvidenceRecord:
    record_id:str; payload:dict[str,Any]; origin:Origin; observed_at:str
    valid_from:str|None=None; valid_to:str|None=None; source_identity:str=""
    environment_digest:str=""; model_digest:str=""; tool_digest:str=""
    parent_evidence:tuple[str,...]=(); trust_class:str="normal"
    verification_state:Verification=Verification.UNVERIFIED; quarantined:bool=False
    sequence:int=0; previous_record_hash:str=""; record_hash:str=""
    signer_key_id:str=""; signature_b64:str=""
    @staticmethod
    def now(record_id,payload,origin,**kw):
        return EvidenceRecord(record_id,payload,origin,datetime.now(timezone.utc).isoformat(),**kw)

@dataclass(frozen=True)
class EvidenceEligibility:
    observation_eligible:bool
    belief_eligible:bool
    hypothesis_eligible:bool
    training_eligible:bool
    promotion_evidence_eligible:bool
    reason:str

class EvidencePolicy:
    def classify(self,r:EvidenceRecord)->EvidenceEligibility:
        if r.quarantined or r.verification_state==Verification.REJECTED:
            return EvidenceEligibility(False,False,False,False,False,"quarantined/rejected")
        verified=r.verification_state==Verification.VERIFIED
        if r.origin in (Origin.SIMULATION,Origin.REPLAY):
            return EvidenceEligibility(False,False,True,False,False,"derived world output")
        if r.origin==Origin.MODEL_INFERENCE:
            return EvidenceEligibility(False,False,True,verified,False,"model inference is not observation")
        if not verified:
            return EvidenceEligibility(False,False,True,False,False,"unverified source")
        return EvidenceEligibility(True,True,True,True,True,"verified observation")
