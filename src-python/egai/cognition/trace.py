from dataclasses import dataclass,field,asdict
from egai.common.canonical import digest

@dataclass(frozen=True)
class CognitiveStep:
    index:int; action:str; input_digest:str; output_digest:str; evidence_refs:tuple[str,...]=(); skill_refs:tuple[str,...]=(); cost:float=0.; verified:bool=False

@dataclass
class EpisodeTrace:
    episode_id:str; model_digest:str; steps:list[CognitiveStep]=field(default_factory=list); final_score:float|None=None
    @property
    def digest(self):return digest({'episode_id':self.episode_id,'model_digest':self.model_digest,'steps':[asdict(s) for s in self.steps],'final_score':self.final_score})
