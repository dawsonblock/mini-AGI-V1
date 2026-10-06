from dataclasses import dataclass
from egai.cognition.model import FrozenModelGuard
from egai.skills.learner import VerifiedEpisode
from .leakage import detect_leakage
from .statistics import paired_bootstrap
from .metrics import BenchmarkPoint, fte

@dataclass(frozen=True)
class EpisodeTrace:
    case_id:str; task_kind:str; input_text:str
    attempted_output:str; repaired_output:str
    attempted_score:float; repaired_score:float
    used_skills:tuple[str,...]=()

@dataclass(frozen=True)
class SequentialPoint:
    episode:int; future_success:float; retention_success:float; security_success:float
    baseline_future_success:float; baseline_retention_success:float; baseline_security_success:float
    gain_vs_frozen:float; ci_low:float; ci_high:float; model_digest:str; learner_snapshot_digest:str
    fte:float=0.0; skill_failures:int=0; verified_repairs:int=0
    negative_transfer_rate:float=0.0; future_n:int=0

@dataclass(frozen=True)
class SequentialReport:
    points:tuple[SequentialPoint,...]; leakage_clean:bool; model_digest_constant:bool; learned_procedures:int
    verified_repairs:int=0; skill_failures:int=0; traces:tuple[EpisodeTrace,...]=()
    max_negative_transfer_rate:float=0.0; degraded_skills:int=0; retired_skills:int=0

class SequentialExperiment:
    def __init__(self,experience_cases,evaluation_cases,scorer,checkpoints=(0,50,100,250,500,1000),bootstrap_samples=1000,near_leakage_threshold=.95,verification_authority=None,receipt_validator=None):
        self.experience=list(experience_cases);self.evaluation=list(evaluation_cases);self.scorer=scorer
        self.checkpoints=tuple(sorted(set(checkpoints)));self.bootstrap_samples=bootstrap_samples
        self.verification_authority=verification_authority; self.receipt_validator=receipt_validator
        leak=detect_leakage(self.experience,self.evaluation,near_leakage_threshold)
        if not leak.clean: raise ValueError(f'benchmark leakage detected: {leak}')
        self.leakage=leak
    @staticmethod
    def _text(resp): return resp.text if hasattr(resp,'text') else resp
    def _eval(self,frozen_solver,learner):
        base_by={'future':[],'retention':[],'security':[]};cand_by={'future':[],'retention':[],'security':[]}
        for c in self.evaluation:
            b=self._text(frozen_solver(c.input,c.task_kind)); a=self._text(learner.solve(c.input,c.task_kind))
            bs=float(self.scorer(b,c.expected));cs=float(self.scorer(a,c.expected))
            base_by.setdefault(c.split,[]).append(bs);cand_by.setdefault(c.split,[]).append(cs)
        avg=lambda xs:sum(xs)/len(xs) if xs else 0.
        eff=paired_bootstrap(base_by['future'],cand_by['future'],self.bootstrap_samples,seed=0) if base_by['future'] else None
        protected=list(zip(base_by['future']+base_by['retention'],cand_by['future']+cand_by['retention']))
        neg=sum(1 for b,c in protected if c+1e-12<b)
        neg_rate=neg/len(protected) if protected else 0.
        return ({s:avg(base_by[s]) for s in base_by},{s:avg(cand_by[s]) for s in cand_by},eff,neg_rate,len(base_by['future']))
    def run(self,frozen_model,learner,checkpoint_callback=None,verification_callback=None):
        guard=FrozenModelGuard(frozen_model)
        if learner.model_digest!=guard.expected:raise ValueError('baseline and learner must use same frozen model identity')
        points=[]; cp=set(x for x in self.checkpoints if x<=len(self.experience)); verified_repairs=0; skill_failures=0; traces=[]
        def frozen_solver(inp,kind): return frozen_model.generate(f"TASK_KIND: {kind}\nVERIFIED PROCEDURES:\n(none)\nTASK:\n{inp}\nReturn only the answer.")
        def checkpoint(ep):
            guard.assert_unchanged(frozen_model)
            if learner.model_digest!=guard.expected:raise RuntimeError('learner changed base model identity')
            base,cand,effect,neg_rate,future_n=self._eval(frozen_solver,learner);gain=effect.mean_gain if effect else 0.
            bp=BenchmarkPoint(0,base['future'],base['retention'],0.,0.,0.);ap=BenchmarkPoint(ep,cand['future'],cand['retention'],0.,float(ep),0.)
            val=fte(bp,ap,max(ep,1)) if ep else 0.
            p=SequentialPoint(ep,cand['future'],cand['retention'],cand['security'],base['future'],base['retention'],base['security'],gain,effect.ci_low if effect else 0.,effect.ci_high if effect else 0.,learner.model_digest,learner.snapshot_digest,val,skill_failures,verified_repairs,neg_rate,future_n)
            points.append(p)
            if checkpoint_callback: checkpoint_callback(p)
        if 0 in cp:checkpoint(0)
        for i,c in enumerate(self.experience,1):
            resp=learner.solve(c.input,c.task_kind);text=self._text(resp);attempt_score=float(self.scorer(text,c.expected))
            if hasattr(resp,'used_skills'):
                if attempt_score < 1.0:
                    for sid in resp.used_skills: learner.skills.mark_failure(sid); skill_failures+=1
                else:
                    for sid in resp.used_skills: learner.skills.mark_success(sid)
            repaired_output=str(c.expected); verified=False; receipt_digest=''; verifier_id=''
            if self.verification_authority is not None:
                receipt=self.verification_authority.verify_repair(c,text,repaired_output)
                if self.receipt_validator is None: raise RuntimeError('verification authority requires receipt validator')
                self.receipt_validator.validate(receipt,c,text,repaired_output); verified=True; receipt_digest=receipt.digest; verifier_id=receipt.verifier_id; verified_repairs+=1
                if verification_callback: verification_callback(receipt)
            elif bool(c.feedback_verified or c.metadata.get('feedback_verified',False)) and c.procedure_hint:
                verified=True; verifier_id=str(c.metadata.get('verifier_id','legacy-benchmark-oracle')); verified_repairs+=1
            learned_output=repaired_output if verified and c.procedure_hint else str(text); learned_score=float(self.scorer(learned_output,c.expected)) if verified else attempt_score
            learner.observe(VerifiedEpisode(c.case_id,c.task_kind,str(c.input),learned_output,str(c.expected),learned_score,verified,c.procedure_hint,c.tags,str(text),attempt_score,verifier_id,receipt_digest))
            traces.append(EpisodeTrace(c.case_id,c.task_kind,str(c.input),str(text),repaired_output,attempt_score,learned_score,tuple(getattr(resp,'used_skills',()))))
            guard.assert_unchanged(frozen_model)
            if i in cp:checkpoint(i)
        health=learner.skills.health() if hasattr(learner.skills,'health') else {"degraded":0,"retired":0}
        return SequentialReport(tuple(points),True,all(p.model_digest==guard.expected for p in points),len(learner.skills.all()),verified_repairs,skill_failures,tuple(traces),max((p.negative_transfer_rate for p in points),default=0.),health.get("degraded",0),health.get("retired",0))
