import argparse, json
from dataclasses import asdict
from pathlib import Path
from egai.cognition.model import FrozenModelIdentity
from egai.cognition.backends import load_backend
from egai.cognition.agent import SandboxAdaptiveAgent
from egai.bench.datasets import read_jsonl
from egai.bench.sequential import SequentialExperiment

def model_id(manifest_path):
    d=json.loads(Path(manifest_path).read_text());ident=FrozenModelIdentity(d['model_name'],tuple((x[0],x[1]) for x in d['files']),d.get('config_digest',''),d.get('tokenizer_digest',''),d.get('runtime_family','local'))
    print(json.dumps({'model_digest':ident.digest,'model_name':ident.model_name,'files':len(ident.files)},indent=2))

def make_model_manifest(name,files,out,runtime_family='local'):
    ident=FrozenModelIdentity.from_files(name,files,runtime_family=runtime_family)
    body={'model_name':ident.model_name,'files':list(ident.files),'config_digest':ident.config_digest,'tokenizer_digest':ident.tokenizer_digest,'runtime_family':ident.runtime_family,'model_digest':ident.digest}
    Path(out).write_text(json.dumps(body,indent=2));print(json.dumps(body,indent=2))

def sequential(experience_path,evaluation_path,backend_config,checkpoints):
    model=load_backend(backend_config);agent=SandboxAdaptiveAgent(model)
    experience=read_jsonl(experience_path);evaluation=read_jsonl(evaluation_path)
    scorer=lambda p,e:1.0 if str(p).strip()==str(e).strip() else 0.0
    cps=tuple(int(x) for x in checkpoints.split(','))
    report=SequentialExperiment(experience,evaluation,scorer,checkpoints=cps).run(model,agent)
    print(json.dumps({'model_digest':model.model_digest,'model_digest_constant':report.model_digest_constant,'leakage_clean':report.leakage_clean,'learned_procedures':report.learned_procedures,'points':[asdict(x) for x in report.points]},indent=2))

def research_demo():
    """Deterministic wiring demo; does not qualify production or general learning."""
    from egai.cognition.model import FrozenModel,FrozenModelIdentity
    from egai.bench.engine import TaskCase
    class RuleModel(FrozenModel):
        @property
        def model_digest(self):return FrozenModelIdentity('research-rule-demo',()).digest
        def generate(self,prompt):
            task=prompt.split('TASK:\n',1)[1].split('\nReturn only',1)[0]
            return task[::-1] if 'Reverse the input characters.' in prompt else 'UNKNOWN'
    model=RuleModel();learner=SandboxAdaptiveAgent(model)
    meta={'feedback_verified':True,'verifier_id':'deterministic-demo-oracle'}
    exp=[TaskCase('e1','experience','abc','cba','reverse','Reverse the input characters.',metadata=meta),
         TaskCase('e2','experience','cat','tac','reverse','Reverse the input characters.',metadata=meta)]
    ev=[TaskCase('f1','future','dog','god','reverse'),TaskCase('r1','retention','plain','UNKNOWN','other'),
        TaskCase('s1','security','retain','UNKNOWN','other')]
    report=SequentialExperiment(exp,ev,lambda a,b:float(a==b),checkpoints=(0,1,2),bootstrap_samples=100,near_leakage_threshold=None).run(model,learner)
    print(json.dumps({'scope':'deterministic research wiring only','report':asdict(report)},indent=2))


def main():
    p=argparse.ArgumentParser(prog='egai');sp=p.add_subparsers(dest='cmd',required=True)
    sp.add_parser('research-demo')
    m=sp.add_parser('model-id');m.add_argument('--manifest',required=True)
    mm=sp.add_parser('make-model-manifest');mm.add_argument('--name',required=True);mm.add_argument('--file',action='append',required=True,dest='files');mm.add_argument('--out',required=True);mm.add_argument('--runtime-family',default='local')
    se=sp.add_parser('sequential');se.add_argument('--experience',required=True);se.add_argument('--evaluation',required=True);se.add_argument('--backend-config',required=True);se.add_argument('--checkpoints',default='0,10,50,100,250,500,1000')
    a=p.parse_args()
    if a.cmd=='research-demo':research_demo()
    elif a.cmd=='model-id':model_id(a.manifest)
    elif a.cmd=='make-model-manifest':make_model_manifest(a.name,a.files,a.out,a.runtime_family)
    else:sequential(a.experience,a.evaluation,a.backend_config,a.checkpoints)
if __name__=='__main__':main()
