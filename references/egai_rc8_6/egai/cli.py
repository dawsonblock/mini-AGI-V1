import argparse,json
from dataclasses import asdict
from pathlib import Path
from egai.runtime import Runtime
from egai.common.keystore import LocalKeyStore
from egai.common.trust import AuthorityTrust
from egai.evidence.model import EvidenceRecord,Origin,Verification
from egai.learning.proposals import PlasticityProposer
from egai.authority.model import CandidateManifest
from egai.authority.build import BuildAuthority
from egai.authority.benchmark import BenchmarkRegistrar
from egai.authority.evaluate import IndependentEvaluator
from egai.authority.runner import BenchmarkRunner
from egai.bench.engine import TaskCase
from egai.bench.datasets import read_jsonl
from egai.bench.sequential import SequentialExperiment
from egai.bench.repair import ReferenceRepairProvider,JSONLRepairProvider
from egai.authority.qualify import Qualifier,PromotionAuthority
from egai.authority.apply import PromotionApplier
from egai.cognition.model import FrozenEchoModel,FrozenModelIdentity,FrozenModel
from egai.cognition.backends import load_backend
from egai.cognition.agent import SandboxAdaptiveAgent
from egai.experiment.manifest import ExperimentRegistrar
from egai.experiment.journal import ExperimentJournal
from egai.experiment.orchestrator import ExperimentOrchestrator


def _authority(rt):
    ks=LocalKeyStore(rt.root/'keys')
    signers={name:ks.load_or_create(name) for name in ('benchmark','builder','runner','evaluator','qualifier','promotion')}
    verifier=ks.build_verifier();trust=AuthorityTrust.from_signers(**signers)
    rt.verifier=verifier;rt.evidence.verifier=verifier;rt.improvements.verifier=verifier
    return signers,verifier,trust

def demo(root):
    rt=Runtime(root);s,v,t=_authority(rt)
    if not rt.evidence.all():
        e=rt.evidence.append(EvidenceRecord.now('E-0001',{'claim':'verified repair works','stance':'support'},Origin.DETERMINISTIC_TOOL,source_identity='test-runner',verification_state=Verification.VERIFIED))
    else:e=rt.evidence.all()[-1]
    prop=PlasticityProposer().propose('P-0001',e.record_hash,'R2',skill_possible=True)
    artifact=rt.artifacts.put_bytes(b'{"skill":"repair"}');cand=CandidateManifest('C-0001',prop.digest,artifact)
    env=rt.artifacts.put_bytes(b'python=3.11');build=BuildAuthority('builder',s['builder'],rt.artifacts).attest('B-0001',cand,env,artifact)
    cases=[TaskCase('f1','future',1,1),TaskCase('f2','future',2,2),TaskCase('r1','retention',3,3),TaskCase('r2','retention',4,4),TaskCase('s1','security',5,5)]
    bench=BenchmarkRegistrar('benchmark-registry',s['benchmark'],rt.artifacts).register_cases('bench-1','1',cases)
    baseline=lambda x:x if x>=3 else 0;candidate_solver=lambda x:x
    rb=BenchmarkRunner('runner',s['runner'],v,t).run('RUN-1',build,bench,cases,baseline,candidate_solver,e.record_hash,resource_fn=lambda *args:(1.,1.))
    evaluation=IndependentEvaluator('evaluator',s['evaluator'],v,t).evaluate('EV-1',build,bench,rb,1.0)
    q=Qualifier('qualifier',s['qualifier'],v,t).evaluate(evaluation);d=PromotionAuthority('promotion-authority',s['promotion'],v,t).decide(q)
    model=FrozenEchoModel();runtime=PromotionApplier(v,t,rt.runtimes,rt.artifacts).activate('runtime-1',model.model_digest,(build.output_digest,),cand,build,bench,rb,evaluation,q,d)
    rt.improvements.record('I-1',prop.digest,{'gain':prop.expected_gain},evaluation.metrics,q.digest,d.digest,{'runtime':runtime.digest})
    print(json.dumps({'status':rt.status(),'proposal':prop.mechanism,'qualification':{'passed':q.passed,'reasons':q.reasons},'promotion':d.approved,'runtime_digest':runtime.digest,'base_model_digest':model.model_digest},indent=2))

def model_id(manifest_path):
    d=json.loads(Path(manifest_path).read_text());ident=FrozenModelIdentity(d['model_name'],tuple((x[0],x[1]) for x in d['files']),d.get('config_digest',''),d.get('tokenizer_digest',''),d.get('runtime_family','local'),d.get('runtime_digest',''),d.get('generation_config_digest',''))
    print(json.dumps({'model_digest':ident.digest,'model_name':ident.model_name,'files':len(ident.files)},indent=2))

def make_model_manifest(name,files,out,runtime_family='local'):
    ident=FrozenModelIdentity.from_files(name,files,runtime_family=runtime_family)
    body={'model_name':ident.model_name,'files':list(ident.files),'config_digest':ident.config_digest,'tokenizer_digest':ident.tokenizer_digest,'runtime_family':ident.runtime_family,'runtime_digest':ident.runtime_digest,'generation_config_digest':ident.generation_config_digest,'model_digest':ident.digest}
    Path(out).write_text(json.dumps(body,indent=2));print(json.dumps(body,indent=2))

def sequential(experience_path,evaluation_path,backend_config,checkpoints):
    model=load_backend(backend_config);agent=SandboxAdaptiveAgent(model)
    experience=read_jsonl(experience_path);evaluation=read_jsonl(evaluation_path)
    scorer=lambda p,e:1.0 if str(p).strip()==str(e).strip() else 0.0
    cps=tuple(int(x) for x in checkpoints.split(','))
    report=SequentialExperiment(experience,evaluation,scorer,checkpoints=cps).run(model,agent)
    print(json.dumps({'model_digest':model.model_digest,'model_digest_constant':report.model_digest_constant,'leakage_clean':report.leakage_clean,'learned_procedures':report.learned_procedures,'points':[asdict(x) for x in report.points]},indent=2))


def research_run(experience_path,evaluation_path,backend_config,root,checkpoints,repair_jsonl='',synthetic_reference_repairs=False):
    rt=Runtime(root);ks=LocalKeyStore(rt.root/'keys');reg=ks.load_or_create('experiment-registrar');runner=ks.load_or_create('experiment-runner');v=ks.build_verifier();v.register(reg.key_id,reg.public_bytes());v.register(runner.key_id,runner.public_bytes())
    model=load_backend(backend_config);experience=read_jsonl(experience_path);evaluation=read_jsonl(evaluation_path);cps=tuple(int(x) for x in checkpoints.split(','))
    cfg={'bootstrap_samples':1000,'synthetic_reference_repairs':bool(synthetic_reference_repairs)}
    mf=ExperimentRegistrar('experiment-registrar',reg).register('EXP-1','1',model.model_digest,experience,evaluation,cps,learner_config=cfg)
    repair=JSONLRepairProvider(repair_jsonl) if repair_jsonl else None
    journal=ExperimentJournal(rt.root/'experiment.jsonl',runner,v);orch=ExperimentOrchestrator('experiment-runner',runner,v,(reg.key_id,),journal,rt.artifacts,feedback_signer=runner,trusted_feedback_keys=(runner.key_id,),source_root=Path(__file__).parent,repair_provider=repair)
    result,report=orch.run('RESULT-1',mf,model,experience,evaluation,lambda p,e:1.0 if str(p).strip()==str(e).strip() else 0.0)
    print(json.dumps({'manifest_digest':mf.digest,'result':asdict(result),'points':[asdict(x) for x in report.points]},indent=2))

def main():
    p=argparse.ArgumentParser(prog='egai');sp=p.add_subparsers(dest='cmd',required=True)
    for name in ('demo','status'):x=sp.add_parser(name);x.add_argument('--root',default='./egai-run')
    m=sp.add_parser('model-id');m.add_argument('--manifest',required=True)
    mm=sp.add_parser('make-model-manifest');mm.add_argument('--name',required=True);mm.add_argument('--file',action='append',required=True,dest='files');mm.add_argument('--out',required=True);mm.add_argument('--runtime-family',default='local')
    se=sp.add_parser('sequential');se.add_argument('--experience',required=True);se.add_argument('--evaluation',required=True);se.add_argument('--backend-config',required=True);se.add_argument('--checkpoints',default='0,10,50,100,250,500,1000')
    rr=sp.add_parser('research-run');rr.add_argument('--experience',required=True);rr.add_argument('--evaluation',required=True);rr.add_argument('--backend-config',required=True);rr.add_argument('--root',default='./egai-research-run');rr.add_argument('--checkpoints',default='0,10,50,100,250,500,1000');rr.add_argument('--repair-jsonl',default='');rr.add_argument('--synthetic-reference-repairs',action='store_true')
    a=p.parse_args()
    if a.cmd=='demo':demo(a.root)
    elif a.cmd=='status':print(json.dumps(Runtime(a.root).status(),indent=2))
    elif a.cmd=='model-id':model_id(a.manifest)
    elif a.cmd=='make-model-manifest':make_model_manifest(a.name,a.files,a.out,a.runtime_family)
    elif a.cmd=='research-run':research_run(a.experience,a.evaluation,a.backend_config,a.root,a.checkpoints,a.repair_jsonl,a.synthetic_reference_repairs)
    else:sequential(a.experience,a.evaluation,a.backend_config,a.checkpoints)
if __name__=='__main__':main()
