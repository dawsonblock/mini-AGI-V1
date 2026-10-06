from dataclasses import asdict,replace
from .model import BenchmarkSpec
from egai.bench.datasets import cases_bytes

class BenchmarkRegistrar:
    """Signs preregistered benchmark identities and binds exact case artifacts."""
    def __init__(self,registrar_id,signer,artifact_store):
        self.registrar_id=registrar_id; self.signer=signer; self.artifact_store=artifact_store
    def register(self,benchmark_id,version,task_set_digest,retention_set_digest,security_set_digest,scorer_id='exact_match',harness_version='egai-benchmark-v1'):
        for d in (task_set_digest,retention_set_digest,security_set_digest):
            if not self.artifact_store.exists(d): raise FileNotFoundError(d)
        u=BenchmarkSpec(benchmark_id,version,task_set_digest,retention_set_digest,security_set_digest,scorer_id,harness_version,True,self.registrar_id)
        env=self.signer.sign(asdict(u)); return replace(u,benchmark_key_id=env.key_id,signature_b64=env.signature_b64)
    def register_cases(self,benchmark_id,version,cases,scorer_id='exact_match',harness_version='egai-benchmark-v1'):
        future=self.artifact_store.put_bytes(cases_bytes(cases,'future'))
        retention=self.artifact_store.put_bytes(cases_bytes(cases,'retention'))
        security=self.artifact_store.put_bytes(cases_bytes(cases,'security'))
        return self.register(benchmark_id,version,future,retention,security,scorer_id,harness_version)
