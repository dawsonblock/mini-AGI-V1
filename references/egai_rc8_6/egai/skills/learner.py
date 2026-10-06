from dataclasses import dataclass
from collections import defaultdict
from egai.common.canonical import digest
from .memory import LearnedProcedure
from .synthesis import SynthesisExample,PatternSkillSynthesizer

@dataclass(frozen=True)
class VerifiedEpisode:
    episode_id:str; task_kind:str; input_text:str; output_text:str; expected_text:str; score:float; verified:bool
    procedure_hint:str=''; tags:tuple[str,...]=(); attempted_output:str=''; attempted_score:float=0.0
    verifier_id:str=''; verification_receipt_digest:str=''; evidence_root:str=''

class SkillMiner:
    """Legacy compatibility miner. RC8.6 production paths use SynthesizedSkillMiner, not procedure_hint."""
    def __init__(self,min_support=2,min_score=1.0,min_distinct_inputs=1):
        self.min_support=min_support;self.min_score=min_score;self.min_distinct_inputs=min_distinct_inputs;self._groups=defaultdict(dict)
    def observe(self,e):
        if not e.verified or e.score<self.min_score or not e.procedure_hint:return None
        key=(e.task_kind,e.procedure_hint.strip());self._groups[key][e.episode_id]=e;xs=list(self._groups[key].values())
        distinct={digest({'input':x.input_text}) for x in xs}
        if len(xs)<self.min_support or len(distinct)<self.min_distinct_inputs:return None
        pid='proc-'+digest({'kind':e.task_kind,'hint':key[1]}).split(':')[1][:16]
        return LearnedProcedure(pid,e.task_kind,' '.join(e.tags) or e.task_kind,key[1],tuple(sorted(x.episode_id for x in xs)),len(xs),0,tuple(sorted(set(x.verifier_id for x in xs if x.verifier_id))),tuple(sorted(set(x.verification_receipt_digest for x in xs if x.verification_receipt_digest))),tuple(sorted(distinct)),tuple(sorted(set(x.evidence_root for x in xs if x.evidence_root))),digest({'legacy':key[1]}))

class SynthesizedSkillMiner:
    """Learns only from independently verified trajectories and synthesizes its own procedure candidate."""
    def __init__(self,synthesizer=None,min_support=2,min_score=1.0,min_distinct_inputs=2):
        self.synthesizer=synthesizer or PatternSkillSynthesizer();self.min_support=min_support;self.min_score=min_score;self.min_distinct_inputs=min_distinct_inputs;self._groups=defaultdict(dict)
    def observe(self,e):
        if not e.verified or e.score<self.min_score or not e.verification_receipt_digest or not e.evidence_root:return None
        self._groups[e.task_kind][e.episode_id]=e;xs=list(self._groups[e.task_kind].values())
        distinct={digest({'input':x.input_text}) for x in xs}
        if len(xs)<self.min_support or len(distinct)<self.min_distinct_inputs:return None
        examples=tuple(SynthesisExample(x.episode_id,x.task_kind,x.input_text,x.attempted_output,x.output_text,x.evidence_root,x.verification_receipt_digest) for x in xs)
        s=self.synthesizer.synthesize(examples)
        if not s:return None
        pid='proc-'+digest({'kind':e.task_kind,'synthesis':s.digest}).split(':')[1][:16]
        return LearnedProcedure(pid,e.task_kind,s.trigger_text,s.procedure_text,tuple(sorted(x.episode_id for x in xs)),len(xs),0,tuple(sorted(set(x.verifier_id for x in xs if x.verifier_id))),tuple(sorted(set(x.verification_receipt_digest for x in xs if x.verification_receipt_digest))),tuple(sorted(distinct)),s.supporting_evidence,s.digest)
