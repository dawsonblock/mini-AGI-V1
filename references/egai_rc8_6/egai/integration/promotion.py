from dataclasses import dataclass,asdict
import json
from egai.authority.model import CandidateManifest
from egai.authority.build import BuildAuthority
from egai.authority.benchmark import BenchmarkRegistrar
from egai.authority.runner import BenchmarkRunner
from egai.authority.evaluate import IndependentEvaluator
from egai.authority.qualify import Qualifier,PromotionAuthority
from egai.authority.apply import PromotionApplier
from egai.common.canonical import digest
from egai.common.crypto import SignedEnvelope

@dataclass(frozen=True)
class ResearchPromotionEnvelope:
    learning_package_digest:str;research_qualification_digest:str
    @property
    def digest(self):return digest(self)

@dataclass(frozen=True)
class PromotionBundle:
    candidate_digest:str;build_digest:str;benchmark_digest:str;result_digest:str;evaluation_digest:str;qualification_digest:str;decision_digest:str;runtime_digest:str

class GovernedPromotionPipeline:
    """Explicit authority-side bridge. A valid research qualification only creates eligibility; authority re-evaluates the exact built artifact."""
    def __init__(self,store,trust,verifier,signers,runtime_registry,trusted_research_qualifier_keys=()):
        self.store=store;self.trust=trust;self.verifier=verifier;self.signers=signers;self.runtime_registry=runtime_registry;self.trusted_research_qualifier_keys=set(trusted_research_qualifier_keys)
    def _verify_research_qualification(self,q):
        if not q.passed:raise PermissionError('research qualification failed')
        if q.qualifier_key_id not in self.trusted_research_qualifier_keys:raise PermissionError('untrusted research qualifier')
        if not self.verifier.verify(asdict(q.unsigned()),SignedEnvelope(q.qualifier_key_id,q.signature_b64)):raise PermissionError('invalid research qualification signature')
    def promote(self,package,research_qualification,base_model_digest,cases,baseline_solver,candidate_solver,environment_bytes=b'{}',policy=None):
        self._verify_research_qualification(research_qualification)
        skill_artifact=self.store.put_bytes(json.dumps(asdict(package.skill_manifest),sort_keys=True,separators=(',',':')).encode());envd=self.store.put_bytes(environment_bytes)
        package_digest=self.store.put_bytes(json.dumps(asdict(package),sort_keys=True,separators=(',',':')).encode())
        envelope=ResearchPromotionEnvelope(package_digest,research_qualification.digest);proposal_digest=self.store.put_bytes(json.dumps(asdict(envelope),sort_keys=True,separators=(',',':')).encode())
        candidate=CandidateManifest('C-'+package.skill_manifest.skill_id,proposal_digest,skill_artifact)
        build=BuildAuthority('builder',self.signers['builder'],self.store).attest('B-'+candidate.candidate_id,candidate,envd,skill_artifact)
        benchmark=BenchmarkRegistrar('benchmark',self.signers['benchmark'],self.store).register_cases('authority-hidden','1',cases)
        result=BenchmarkRunner('runner',self.signers['runner'],self.verifier,self.trust).run('RUN-'+candidate.candidate_id,build,benchmark,cases,baseline_solver,candidate_solver,evidence_digest=package.evidence_root)
        evaluation=IndependentEvaluator('evaluator',self.signers['evaluator'],self.verifier,self.trust).evaluate('E-'+candidate.candidate_id,build,benchmark,result,1.0)
        qualification=Qualifier('qualifier',self.signers['qualifier'],self.verifier,self.trust,policy).evaluate(evaluation)
        decision=PromotionAuthority('promotion',self.signers['promotion'],self.verifier,self.trust).decide(qualification)
        runtime=PromotionApplier(self.verifier,self.trust,self.runtime_registry,self.store).activate('RT-'+candidate.candidate_id,base_model_digest,(skill_artifact,),candidate,build,benchmark,result,evaluation,qualification,decision)
        return PromotionBundle(candidate.digest,build.digest,benchmark.digest,result.digest,evaluation.digest,qualification.digest,decision.digest,runtime.digest)
