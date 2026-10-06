from dataclasses import dataclass,asdict
import json
from egai.common.canonical import digest
from egai.learning.abstraction import AbstractionEngine
from egai.learning.transfer import TransferEvaluator,highest_ring
from egai.learning.proposals import PlasticityProposer
from egai.skills.model import SkillManifest

@dataclass(frozen=True)
class LearningArtifactPackage:
    skill_manifest:SkillManifest;hypothesis_digest:str;abstraction_digest:str;transfer_digest:str;proposal_digest:str;evidence_root:str
    @property
    def digest(self):return digest(self)

class LearningArtifactBuilder:
    """Converts learned sandbox procedures into evidence-backed proposal artifacts; does not promote them."""
    def __init__(self,artifact_store):self.store=artifact_store
    def build(self,procedure,report):
        roots=procedure.supporting_evidence_roots
        if len(roots)<2:raise ValueError('persistent skill candidate requires repeated evidence ancestry')
        h,a=AbstractionEngine().propose(procedure.task_kind,procedure.procedure_text,roots)
        # RC8.6 maps demonstrated hidden future success to R1; richer benchmark metadata can extend rings later.
        gain=report.points[-1].gain_vs_frozen if report.points else 0.;te=TransferEvaluator().evaluate(a.abstraction_id,{'R0':1.0,'R1':1.0 if gain>0 else 0.0})
        ring=highest_ring(te.ring_results);lp=PlasticityProposer().propose('LP-'+procedure.procedure_id,report.evidence_head,ring,skill_possible=True,expected_gain=max(gain,0.0),risk=report.max_negative_transfer_rate)
        skill=SkillManifest(procedure.procedure_id,'1',a.principle,(procedure.trigger_text,),(),(),('task',),('answer',),{'handler':'model_procedure','procedure_text':procedure.procedure_text},resource_limits={'max_invocations':1},termination_conditions=('answer produced',),success_postconditions=('independent verifier passes',),verifier={'handler':'task_verifier'},known_failure_modes=(),supporting_evidence=tuple(roots),contradicting_evidence=(),qualification_receipt='',rollback_target='')
        for obj in (h,a,te,lp,skill):self.store.put_bytes(json.dumps(asdict(obj),sort_keys=True,separators=(',',':')).encode())
        return LearningArtifactPackage(skill,digest(h),digest(a),digest(te),lp.digest,report.evidence_head)
