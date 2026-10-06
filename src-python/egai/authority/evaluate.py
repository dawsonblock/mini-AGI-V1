from dataclasses import asdict,replace
from statistics import mean
from egai.common.crypto import SignedEnvelope
from .model import EvaluationBundle

class IndependentEvaluator:
    def __init__(self,evaluator_id,signer,verifier,trust):
        self.evaluator_id=evaluator_id;self.signer=signer;self.verifier=verifier;self.trust=trust
    def evaluate(self,evaluation_id,build,benchmark,result_bundle,provenance_closure):
        self.trust.require('builder',build.builder_key_id)
        self.trust.require('benchmark',benchmark.benchmark_key_id)
        self.trust.require('runner',result_bundle.runner_key_id)
        if not self.verifier.verify(asdict(build.unsigned()),SignedEnvelope(build.builder_key_id,build.signature_b64)):
            raise PermissionError('invalid builder signature')
        if not self.verifier.verify(asdict(benchmark.unsigned()),SignedEnvelope(benchmark.benchmark_key_id,benchmark.signature_b64)):
            raise PermissionError('invalid benchmark signature')
        if result_bundle.build_digest!=build.digest or result_bundle.benchmark_digest!=benchmark.digest:raise ValueError('result bundle binding mismatch')
        if not self.verifier.verify(asdict(result_bundle.unsigned()),SignedEnvelope(result_bundle.runner_key_id,result_bundle.signature_b64)):raise PermissionError('invalid benchmark runner signature')
        raw=list(result_bundle.results);future=[r for r in raw if r.split=='future'];old=[r for r in raw if r.split=='retention']
        if not future or not old:raise ValueError('future and retention splits required')
        metrics={
          'future_gain':mean(r.candidate_score-r.baseline_score for r in future),
          'future_baseline':mean(r.baseline_score for r in future),
          'future_candidate':mean(r.candidate_score for r in future),
          'forgetting':max(0.,mean(r.baseline_score-r.candidate_score for r in old)),
          'calibration_delta':max(0.,mean(r.candidate_calibration-r.baseline_calibration for r in raw)),
          'security_regressions':sum(1 for r in raw if r.security_regression),
          'unauthorized_changes':sum(1 for r in raw if not r.authorized_change),
          'resource_ratio':sum(r.candidate_resource for r in raw)/max(sum(r.baseline_resource for r in raw),1e-12),
          'n_results':len(raw)}
        u=EvaluationBundle(evaluation_id,build.digest,benchmark.digest,result_bundle.digest,metrics,provenance_closure,self.evaluator_id)
        env=self.signer.sign(asdict(u));return replace(u,evaluator_key_id=env.key_id,signature_b64=env.signature_b64)
