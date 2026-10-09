from dataclasses import asdict,replace
from egai.common.artifacts import ArtifactStore
from egai.common.crypto import Ed25519Signer,Ed25519Verifier,SignedEnvelope
from egai.common.trust import AuthorityTrust
from egai.authority.model import CandidateManifest
from egai.authority.build import BuildAuthority
from egai.authority.benchmark import BenchmarkRegistrar
from egai.authority.runner import BenchmarkRunner
from egai.authority.scoring import ScorerRegistry
from egai.authority.evaluate import IndependentEvaluator
from egai.authority.qualify import Qualifier
from egai.bench.engine import TaskCase
from minagi.egai.canonical import canonical_bytes
from minagi.egai.trust_chain import ResearchChain,ResearchChainVerifier
from minagi.egai.qualification import IndependentQualificationGate


def prepare_chain(root,proposal,bundle,candidate_doc=None):
    store=ArtifactStore(root/'artifacts')
    if candidate_doc is None:
        candidate_digest=store.put_bytes(b'candidate')
    else:
        candidate_digest=store.put_bytes(canonical_bytes(asdict(candidate_doc)))
        assert candidate_digest==candidate_doc.digest
    signers={r:Ed25519Signer.generate() for r in ('benchmark','builder','runner','evaluator','qualifier')}
    verifier=Ed25519Verifier()
    for signer in signers.values():verifier.register(signer.key_id,signer.public_bytes())
    trust=AuthorityTrust.from_signers(**signers)
    candidate=CandidateManifest('candidate',proposal.digest,candidate_digest)
    build=BuildAuthority('builder',signers['builder'],store).attest('build',candidate,store.put_bytes(b'env'),candidate_digest)
    cases=[TaskCase('future-'+str(i),'future',i,i) for i in range(bundle.qualification_worlds)]
    cases.extend([TaskCase('retention','retention',-1,-1),TaskCase('security','security',-2,-2)])
    scorers=ScorerRegistry();scorers.register('numeric',lambda p,e:float(p))
    bm=BenchmarkRegistrar('benchmark',signers['benchmark'],store).register_cases('benchmark','1',cases,scorer_id='numeric')
    def baseline(x):
        return 0. if x>=0 else 1.
    def candidate_solver(x):
        return bundle.metrics.forward_transfer_delta if x>=0 else 1.
    results=BenchmarkRunner('runner',signers['runner'],verifier,trust,scorers).run('run',build,bm,cases,baseline,candidate_solver,resource_fn=lambda *a:(1.,1.))
    evaluation=IndependentEvaluator('independent-q',signers['evaluator'],verifier,trust).evaluate('evaluation',build,bm,results,1.)
    qualification=Qualifier('qualifier',signers['qualifier'],verifier,trust).evaluate(evaluation)
    bundle=replace(bundle,candidate_digest=candidate_digest,suite_digest=bm.digest)
    chain=ResearchChain(candidate,build,bm,results,evaluation,qualification,SignedEnvelope('',''),SignedEnvelope('',''))
    chain=replace(chain,evaluator_binding=signers['evaluator'].sign(chain.binding(bundle)),qualifier_binding=signers['qualifier'].sign(chain.binding(bundle)))
    cv=ResearchChainVerifier(verifier,trust,store)
    return IndependentQualificationGate(chain_verifier=cv),bundle,chain,cv,signers
