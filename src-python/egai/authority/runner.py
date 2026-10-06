from dataclasses import asdict,replace
import time
from egai.common.crypto import SignedEnvelope
from egai.bench.datasets import cases_digest
from .model import RawResult,ResultBundle
from .scoring import ScorerRegistry

class BenchmarkRunner:
    """Independent runner creates signed raw results from a preregistered case set and runner-owned scorer."""
    def __init__(self,runner_id,signer,verifier,trust,scorers=None):
        self.runner_id=runner_id; self.signer=signer; self.verifier=verifier; self.trust=trust
        self.scorers=scorers or ScorerRegistry.defaults()
    def _verify_inputs(self,build,benchmark):
        self.trust.require('builder',build.builder_key_id)
        if not self.verifier.verify(asdict(build.unsigned()),SignedEnvelope(build.builder_key_id,build.signature_b64)):
            raise PermissionError('invalid builder signature')
        self.trust.require('benchmark',benchmark.benchmark_key_id)
        if not benchmark.preregistered or not self.verifier.verify(asdict(benchmark.unsigned()),SignedEnvelope(benchmark.benchmark_key_id,benchmark.signature_b64)):
            raise PermissionError('benchmark is not validly preregistered')
    def _verify_cases(self,benchmark,cases):
        if cases_digest(cases,'future')!=benchmark.task_set_digest: raise ValueError('future case set does not match preregistration')
        if cases_digest(cases,'retention')!=benchmark.retention_set_digest: raise ValueError('retention case set does not match preregistration')
        if cases_digest(cases,'security')!=benchmark.security_set_digest: raise ValueError('security case set does not match preregistration')
    def run(self,run_id,build,benchmark,cases,baseline_solver,candidate_solver,evidence_digest='',resource_fn=None):
        self._verify_inputs(build,benchmark);self._verify_cases(benchmark,cases);scorer=self.scorers.get(benchmark.scorer_id);results=[]
        seen=set()
        for c in cases:
            if c.split not in ('future','retention','security'): continue
            if c.case_id in seen: raise ValueError('duplicate benchmark case id')
            seen.add(c.case_id)
            t=time.perf_counter();bp=baseline_solver(c.input);bt=max(time.perf_counter()-t,1e-9)
            t=time.perf_counter();cp=candidate_solver(c.input);ct=max(time.perf_counter()-t,1e-9)
            if resource_fn: br,cr=resource_fn(c,bp,cp,bt,ct)
            else: br,cr=bt,ct
            bs=float(scorer(bp,c.expected));cs=float(scorer(cp,c.expected))
            results.append(RawResult(c.case_id,c.split,bs,cs,security_regression=(c.split=='security' and cs<bs),baseline_resource=float(br),candidate_resource=float(cr),evidence_digest=evidence_digest))
        if not results: raise ValueError('no qualification cases')
        u=ResultBundle(run_id,benchmark.digest,build.digest,tuple(results),self.runner_id)
        env=self.signer.sign(asdict(u));return replace(u,runner_key_id=env.key_id,signature_b64=env.signature_b64)
