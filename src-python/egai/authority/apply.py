from dataclasses import asdict
from egai.common.crypto import SignedEnvelope
from egai.common.canonical import validate_digest
from .model import RuntimeManifest

class PromotionApplier:
    """Only component that mutates promoted runtime state."""
    def __init__(self,verifier,trust,runtime_registry,artifact_store=None):
        self.verifier=verifier;self.trust=trust;self.runtime_registry=runtime_registry;self.artifact_store=artifact_store
    def _sig(self,role,obj,key_id,sig):
        self.trust.require(role,key_id)
        if not self.verifier.verify(asdict(obj.unsigned()),SignedEnvelope(key_id,sig)):
            raise PermissionError(f'invalid {role} signature')
    def _verify_chain(self,candidate,build,benchmark,result_bundle,evaluation,qualification,decision):
        if build.candidate_digest!=candidate.digest:raise ValueError('candidate/build binding mismatch')
        if result_bundle.build_digest!=build.digest:raise ValueError('build/result binding mismatch')
        if result_bundle.benchmark_digest!=benchmark.digest:raise ValueError('benchmark/result binding mismatch')
        if evaluation.result_bundle_digest!=result_bundle.digest:raise ValueError('result/evaluation binding mismatch')
        if evaluation.build_digest!=build.digest or evaluation.benchmark_digest!=benchmark.digest:raise ValueError('evaluation binding mismatch')
        if qualification.evaluation_digest!=evaluation.digest:raise ValueError('evaluation/qualification binding mismatch')
        if decision.qualification_digest!=qualification.digest:raise ValueError('qualification/decision binding mismatch')
        if not decision.approved:raise PermissionError('decision rejected')
        self._sig('builder',build,build.builder_key_id,build.signature_b64)
        self._sig('benchmark',benchmark,benchmark.benchmark_key_id,benchmark.signature_b64)
        self._sig('runner',result_bundle,result_bundle.runner_key_id,result_bundle.signature_b64)
        self._sig('evaluator',evaluation,evaluation.evaluator_key_id,evaluation.signature_b64)
        self._sig('qualifier',qualification,qualification.qualifier_key_id,qualification.signature_b64)
        self._sig('promotion',decision,decision.authority_key_id,decision.signature_b64)
        if self.runtime_registry.nonce_used(decision.decision_nonce):raise PermissionError('replayed promotion decision')
        if self.artifact_store:
            for d in (candidate.artifact_digest,build.environment_digest,build.output_digest,benchmark.task_set_digest,benchmark.retention_set_digest,benchmark.security_set_digest):
                if not self.artifact_store.exists(d):raise FileNotFoundError('required artifact missing: '+d)
    def activate(self,runtime_id,base_model_digest,components,candidate,build,benchmark,result_bundle,evaluation,qualification,decision):
        self._verify_chain(candidate,build,benchmark,result_bundle,evaluation,qualification,decision);validate_digest(base_model_digest)
        for d in components:
            validate_digest(d)
            if self.artifact_store and not self.artifact_store.exists(d):raise FileNotFoundError('runtime component missing: '+d)
        prev=self.runtime_registry.current_digest() or '';runtime=RuntimeManifest(runtime_id,decision.digest,base_model_digest,tuple(components),prev);self.runtime_registry.activate(runtime,decision.decision_nonce);return runtime
