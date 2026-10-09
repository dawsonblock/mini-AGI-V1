from dataclasses import dataclass
from egai.cognition.model import FrozenModelGuard
from egai.skills.learner import VerifiedEpisode
from .leakage import detect_leakage
from .statistics import paired_bootstrap
from .metrics import BenchmarkPoint, fte

@dataclass(frozen=True)
class SequentialPoint:
    episode:int
    future_success:float
    retention_success:float
    security_success:float
    gain_vs_frozen:float
    ci_low:float
    ci_high:float
    model_digest:str
    learner_snapshot_digest:str
    fte:float=0.0

@dataclass(frozen=True)
class SequentialReport:
    points:tuple[SequentialPoint,...]
    leakage_clean:bool
    model_digest_constant:bool
    learned_procedures:int

class SequentialExperiment:
    """Frozen-model continual-learning harness.

    Evaluation cases are never passed to observe(). Experience cases may contain externally
    verified feedback (`expected` + `procedure_hint`). A failed model attempt can therefore
    still teach a procedure once a verifier/teacher supplies the successful repair.
    """
    def __init__(self,experience_cases,evaluation_cases,scorer,checkpoints=(0,50,100,250,500,1000),bootstrap_samples=1000,near_leakage_threshold=.95):
        self.experience=list(experience_cases);self.evaluation=list(evaluation_cases);self.scorer=scorer
        self.checkpoints=tuple(sorted(set(checkpoints)));self.bootstrap_samples=bootstrap_samples
        leak=detect_leakage(self.experience,self.evaluation,near_leakage_threshold)
        if not leak.clean: raise ValueError(f'benchmark leakage detected: {leak}')
        self.leakage=leak
    @staticmethod
    def _text(resp): return resp.text if hasattr(resp,'text') else resp
    def _eval(self,frozen_solver,learner,guard,frozen_model):
        by={'future':[],'retention':[],'security':[]};base=[];cand=[]
        for c in self.evaluation:
            b=self._text(frozen_solver(c.input,c.task_kind)); a=self._text(learner.solve(c.input,c.task_kind))
            guard.assert_unchanged(frozen_model)
            if learner.model_digest!=guard.expected:raise RuntimeError('learner changed base model identity')
            bs=float(self.scorer(b,c.expected));cs=float(self.scorer(a,c.expected));by.setdefault(c.split,[]).append(cs)
            if c.split=='future':base.append(bs);cand.append(cs)
        def avg(xs):
            return sum(xs)/len(xs) if xs else 0.
        eff=paired_bootstrap(base,cand,self.bootstrap_samples,seed=0) if base else None
        return avg(by.get('future',[])),avg(by.get('retention',[])),avg(by.get('security',[])),eff
    def run(self,frozen_model,learner):
        guard=FrozenModelGuard(frozen_model)
        if learner.model_digest!=guard.expected:raise ValueError('baseline and learner must use same frozen model identity')
        points=[]; cp=set(x for x in self.checkpoints if x<=len(self.experience))
        def frozen_solver(inp,kind):
            prompt=f"TASK_KIND: {kind}\nVERIFIED PROCEDURES:\n(none)\nTASK:\n{inp}\nReturn only the answer."
            return frozen_model.generate(prompt)
        def checkpoint(ep):
            guard.assert_unchanged(frozen_model)
            if learner.model_digest!=guard.expected:raise RuntimeError('learner changed base model identity')
            future,retention,security,effect=self._eval(frozen_solver,learner,guard,frozen_model)
            base=effect.mean_gain if effect else 0.
            bp=BenchmarkPoint(0,max(0.,future-base),retention,0.,0.,0.)
            ap=BenchmarkPoint(ep,future,retention,0.,float(ep),0.)
            val=fte(bp,ap,max(ep,1)) if ep else 0.
            points.append(SequentialPoint(ep,future,retention,security,base,effect.ci_low if effect else 0.,effect.ci_high if effect else 0.,learner.model_digest,learner.snapshot_digest,val))
        if 0 in cp:checkpoint(0)
        for i,c in enumerate(self.experience,1):
            resp=learner.solve(c.input,c.task_kind);text=self._text(resp);attempt_score=float(self.scorer(text,c.expected))
            feedback_verified=bool(c.metadata.get('feedback_verified', c.feedback_verified))
            if feedback_verified and c.procedure_hint:
                # The successful repair/demonstration is externally verified feedback, not a claim that the model succeeded.
                learned_output=str(c.expected); learned_score=float(self.scorer(c.expected,c.expected)); verified=True
            else:
                learned_output=str(text); learned_score=attempt_score; verified=False
            learner.observe(VerifiedEpisode(c.case_id,c.task_kind,str(c.input),learned_output,str(c.expected),learned_score,verified,c.procedure_hint,c.tags,str(text),attempt_score,str(c.metadata.get('verifier_id','benchmark-oracle'))))
            guard.assert_unchanged(frozen_model)
            if learner.model_digest!=guard.expected:raise RuntimeError('learner changed base model identity')
            if i in cp:checkpoint(i)
        return SequentialReport(tuple(points),True,all(p.model_digest==guard.expected for p in points),len(learner.skills.all()))
