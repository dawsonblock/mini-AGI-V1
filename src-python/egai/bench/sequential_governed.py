from dataclasses import dataclass
from egai.cognition.model import FrozenModelGuard
from egai.skills.learner import VerifiedEpisode
from egai.skills.verification import VerificationAuthority,VerificationReceiptValidator
from egai.evidence.session import EvidenceSession
from egai.epistemic.beliefs import BeliefCompiler
from .leakage import detect_leakage
from .statistics import paired_bootstrap
from .metrics import BenchmarkPoint,fte

@dataclass(frozen=True)
class GovernedTrace:
    case_id:str;task_kind:str;input_text:str;attempted_output:str;repaired_output:str;attempted_score:float;repaired_score:float
    used_skills:tuple[str,...];cognitive_actions:tuple[str,...];evidence_root:str;verification_receipt_digest:str

@dataclass(frozen=True)
class GovernedPoint:
    episode:int;future_success:float;retention_success:float;security_success:float;baseline_future_success:float;baseline_retention_success:float;baseline_security_success:float
    gain_vs_frozen:float;ci_low:float;ci_high:float;model_digest:str;learner_snapshot_digest:str;fte:float=0.;skill_failures:int=0;verified_repairs:int=0;negative_transfer_rate:float=0.;future_n:int=0

@dataclass(frozen=True)
class GovernedReport:
    points:tuple[GovernedPoint,...];leakage_clean:bool;model_digest_constant:bool;learned_procedures:int;verified_repairs:int;skill_failures:int;traces:tuple[GovernedTrace,...]
    max_negative_transfer_rate:float;evidence_head:str;belief_snapshot_digest:str;degraded_skills:int=0;retired_skills:int=0

class GovernedSequentialExperiment:
    def __init__(self,experience,evaluation,scorer,repair_provider,evidence_ledger,feedback_signer,verifier,trusted_feedback_keys,checkpoints=(0,50,100,250,500,1000),bootstrap_samples=1000,near_leakage_threshold=.95):
        self.experience=list(experience);self.evaluation=list(evaluation);self.scorer=scorer;self.repair_provider=repair_provider;self.evidence_ledger=evidence_ledger
        self.feedback=VerificationAuthority('independent-repair-verifier',feedback_signer,scorer);self.validator=VerificationReceiptValidator(verifier,trusted_feedback_keys)
        self.checkpoints=tuple(sorted(set(checkpoints)));self.bootstrap_samples=bootstrap_samples
        leak=detect_leakage(self.experience,self.evaluation,near_leakage_threshold)
        if not leak.clean:raise ValueError(f'benchmark leakage detected: {leak}')
    @staticmethod
    def _text(r):return r.text if hasattr(r,'text') else str(r)
    def _eval(self,model,learner):
        base={'future':[],'retention':[],'security':[]};cand={'future':[],'retention':[],'security':[]}
        for c in self.evaluation:
            bp=model.generate(f"TASK_KIND: {c.task_kind}\nVERIFIED PROCEDURES:\n(none)\nTASK:\n{c.input}\nReturn only the answer.")
            cp=self._text(learner.solve(c.input,c.task_kind));bs=float(self.scorer(bp,c.expected));cs=float(self.scorer(cp,c.expected));base[c.split].append(bs);cand[c.split].append(cs)
        def avg(x): return sum(x)/len(x) if x else 0.
        eff=paired_bootstrap(base['future'],cand['future'],self.bootstrap_samples,seed=0) if base['future'] else None
        pairs=list(zip(base['future']+base['retention'],cand['future']+cand['retention']));neg=sum(1 for b,c in pairs if c+1e-12<b)/len(pairs) if pairs else 0.
        return {s:avg(base[s]) for s in base},{s:avg(cand[s]) for s in cand},eff,neg,len(base['future'])
    def run(self,model,learner,environment_digest='',checkpoint_callback=None,verification_callback=None):
        guard=FrozenModelGuard(model);session=EvidenceSession(self.evidence_ledger,model.model_digest,environment_digest);points=[];traces=[];repairs=failures=0;cp=set(x for x in self.checkpoints if x<=len(self.experience))
        def checkpoint(ep):
            guard.assert_unchanged(model);b,c,e,n,fn=self._eval(model,learner);g=e.mean_gain if e else 0.;bp=BenchmarkPoint(0,b['future'],b['retention'],0.,0.,0.);ap=BenchmarkPoint(ep,c['future'],c['retention'],0.,float(ep),0.)
            points.append(GovernedPoint(ep,c['future'],c['retention'],c['security'],b['future'],b['retention'],b['security'],g,e.ci_low if e else 0.,e.ci_high if e else 0.,model.model_digest,learner.snapshot_digest,fte(bp,ap,max(ep,1)) if ep else 0.,failures,repairs,n,fn))
            if checkpoint_callback:checkpoint_callback(points[-1])
        if 0 in cp:checkpoint(0)
        for i,c in enumerate(self.experience,1):
            resp=learner.solve(c.input,c.task_kind);attempt=self._text(resp);ascore=float(self.scorer(attempt,c.expected))
            if ascore>=1.0:
                # Successful own attempts are evidence, but do not become repair-based skill training without an independent repair receipt.
                if hasattr(resp,'used_skills'):
                    for sid in resp.used_skills:learner.skills.mark_success(sid)
                if i in cp:checkpoint(i)
                continue
            if hasattr(resp,'used_skills'):
                for sid in resp.used_skills:learner.skills.mark_failure(sid);failures+=1
            repair=self.repair_provider.repair(c,attempt);receipt=self.feedback.verify_repair(c,attempt,repair.output_text);self.validator.validate(receipt,c,attempt,repair.output_text)
            rscore=float(self.scorer(repair.output_text,c.expected));ev=session.record(c,attempt,ascore,repair,receipt,rscore);repairs+=1
            ep=VerifiedEpisode(c.case_id,c.task_kind,str(c.input),repair.output_text,'',rscore,True,'',tuple(c.tags),attempt,ascore,receipt.verifier_id)
            learner.observe(ep);traces.append(GovernedTrace(c.case_id,c.task_kind,str(c.input),attempt,repair.output_text,ascore,rscore,tuple(getattr(resp,'used_skills',())),tuple(getattr(resp,'cognitive_actions',())),ev.root,receipt.digest))
            if verification_callback:verification_callback(receipt)
            guard.assert_unchanged(model)
            if i in cp:checkpoint(i)
        beliefs=BeliefCompiler().compile(self.evidence_ledger.eligible('belief'));from egai.common.canonical import digest
        _,head=self.evidence_ledger._head();health=learner.skills.health()
        return GovernedReport(tuple(points),True,all(x.model_digest==guard.expected for x in points),len(learner.skills.all()),repairs,failures,tuple(traces),max((x.negative_transfer_rate for x in points),default=0.),head,digest([b.derivation_receipt for b in beliefs]),health.get('degraded',0),health.get('retired',0))
