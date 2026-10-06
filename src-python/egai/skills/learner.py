from dataclasses import dataclass
from collections import defaultdict
from egai.common.canonical import digest
from .memory import LearnedProcedure

@dataclass(frozen=True)
class VerifiedEpisode:
    episode_id:str
    task_kind:str
    input_text:str
    output_text:str
    expected_text:str
    score:float
    verified:bool
    procedure_hint:str=''
    tags:tuple[str,...]=()
    attempted_output:str=''
    attempted_score:float=0.0
    verifier_id:str=''

class SkillMiner:
    """Conservative sandbox miner. Only repeated, independently verified successful feedback can create a procedure."""
    def __init__(self,min_support=2,min_score=1.0):
        self.min_support=min_support;self.min_score=min_score;self._groups=defaultdict(dict)
    def observe(self,e:VerifiedEpisode):
        if not e.verified or e.score<self.min_score or not e.procedure_hint: return None
        key=(e.task_kind,e.procedure_hint.strip())
        self._groups[key][e.episode_id]=e
        xs=list(self._groups[key].values())
        if len(xs)<self.min_support:return None
        pid='proc-'+digest({'kind':e.task_kind,'hint':key[1]}).split(':')[1][:16]
        return LearnedProcedure(pid,e.task_kind,' '.join(e.tags) or e.task_kind,key[1],tuple(sorted(x.episode_id for x in xs)),len(xs),0)
