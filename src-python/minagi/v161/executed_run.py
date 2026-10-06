from __future__ import annotations
from dataclasses import asdict, dataclass
from typing import Mapping, Any
from egai.common.canonical import digest, validate_digest
from egai.common.crypto import Ed25519Signer, Ed25519Verifier, SignedEnvelope

@dataclass(frozen=True)
class ExecutedRunReceiptV161:
    campaign_digest: str
    arm: str
    seed: int
    environment_digest: str
    model_digest: str
    tokenizer_digest: str
    adapter_digest: str
    dataset_digest: str
    evaluator_digest: str
    metrics: Mapping[str, float | int]
    signer_key_id: str
    signature_b64: str
    schema: str = "mini-agi-v16.1-executed-run-receipt-v1"
    def __post_init__(self):
        for x in (self.campaign_digest,self.environment_digest,self.model_digest,self.tokenizer_digest,self.adapter_digest,self.dataset_digest,self.evaluator_digest): validate_digest(x)
        if self.arm not in {"A0","A1"}: raise ValueError("arm must be A0 or A1")
        if int(self.seed)<0 or not self.signer_key_id or not self.signature_b64: raise ValueError("invalid signed run receipt")
        object.__setattr__(self,"metrics",dict(self.metrics))
    @property
    def body(self):
        return {"schema":self.schema,"campaign_digest":self.campaign_digest,"arm":self.arm,"seed":int(self.seed),"environment_digest":self.environment_digest,"model_digest":self.model_digest,"tokenizer_digest":self.tokenizer_digest,"adapter_digest":self.adapter_digest,"dataset_digest":self.dataset_digest,"evaluator_digest":self.evaluator_digest,"metrics":dict(self.metrics),"signer_key_id":self.signer_key_id}
    @property
    def digest(self): return digest(self)
    @classmethod
    def sign(cls, *, signer: Ed25519Signer, campaign_digest: str, arm: str, seed: int, environment_digest: str, model_digest: str, tokenizer_digest: str, adapter_digest: str, dataset_digest: str, evaluator_digest: str, metrics: Mapping[str,float|int]):
        body={"schema":"mini-agi-v16.1-executed-run-receipt-v1","campaign_digest":campaign_digest,"arm":arm,"seed":int(seed),"environment_digest":environment_digest,"model_digest":model_digest,"tokenizer_digest":tokenizer_digest,"adapter_digest":adapter_digest,"dataset_digest":dataset_digest,"evaluator_digest":evaluator_digest,"metrics":dict(metrics),"signer_key_id":signer.key_id}
        env=signer.sign(body)
        return cls(signature_b64=env.signature_b64,**body)
    def verify(self, verifier: Ed25519Verifier) -> bool:
        return verifier.verify(self.body,SignedEnvelope(self.signer_key_id,self.signature_b64))


ARMS_V162 = frozenset({"L1", "L2", "L3", "L4", "L5", "L6", "NC"})


@dataclass(frozen=True)
class ExecutedRunReceiptV162:
    """Six-arm execution receipt.

    Extends V161 with a broader arm vocabulary and a `state_digest`
    binding the arm's non-parametric state (retrieval index, memory
    store, skill library, replay buffer). `adapter_digest` binds only
    parametric artifacts (L6/NC LoRA outputs). Metrics carry resource
    accounting (eval_seconds, generated_tokens, arm_state_bytes) in
    addition to scores.
    """
    campaign_digest: str
    arm: str
    seed: int
    environment_digest: str
    model_digest: str
    tokenizer_digest: str
    adapter_digest: str
    state_digest: str
    dataset_digest: str
    evaluator_digest: str
    metrics: Mapping[str, float | int]
    signer_key_id: str
    signature_b64: str
    schema: str = "mini-agi-v16.2-executed-run-receipt-v1"
    def __post_init__(self):
        for x in (self.campaign_digest,self.environment_digest,self.model_digest,self.tokenizer_digest,self.adapter_digest,self.state_digest,self.dataset_digest,self.evaluator_digest): validate_digest(x)
        if self.arm not in ARMS_V162: raise ValueError(f"arm must be one of {sorted(ARMS_V162)}")
        if int(self.seed)<0 or not self.signer_key_id or not self.signature_b64: raise ValueError("invalid signed run receipt")
        object.__setattr__(self,"metrics",dict(self.metrics))
    @property
    def body(self):
        return {"schema":self.schema,"campaign_digest":self.campaign_digest,"arm":self.arm,"seed":int(self.seed),"environment_digest":self.environment_digest,"model_digest":self.model_digest,"tokenizer_digest":self.tokenizer_digest,"adapter_digest":self.adapter_digest,"state_digest":self.state_digest,"dataset_digest":self.dataset_digest,"evaluator_digest":self.evaluator_digest,"metrics":dict(self.metrics),"signer_key_id":self.signer_key_id}
    @property
    def digest(self): return digest(self)
    @classmethod
    def sign(cls, *, signer: Ed25519Signer, campaign_digest: str, arm: str, seed: int, environment_digest: str, model_digest: str, tokenizer_digest: str, adapter_digest: str, state_digest: str, dataset_digest: str, evaluator_digest: str, metrics: Mapping[str,float|int]):
        body={"schema":"mini-agi-v16.2-executed-run-receipt-v1","campaign_digest":campaign_digest,"arm":arm,"seed":int(seed),"environment_digest":environment_digest,"model_digest":model_digest,"tokenizer_digest":tokenizer_digest,"adapter_digest":adapter_digest,"state_digest":state_digest,"dataset_digest":dataset_digest,"evaluator_digest":evaluator_digest,"metrics":dict(metrics),"signer_key_id":signer.key_id}
        env=signer.sign(body)
        return cls(signature_b64=env.signature_b64,**body)
    def verify(self, verifier: Ed25519Verifier) -> bool:
        return verifier.verify(self.body,SignedEnvelope(self.signer_key_id,self.signature_b64))
