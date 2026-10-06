from dataclasses import dataclass,field
from typing import Callable
from egai.common.canonical import digest

@dataclass(frozen=True)
class TaskCase:
    case_id:str; split:str; input:object; expected:object
    task_kind:str=''; procedure_hint:str=''; tags:tuple[str,...]=(); metadata:dict=field(default_factory=dict)
    feedback_verified:bool=False
    @property
    def input_digest(self):return digest({'input':self.input})
    @property
    def case_digest(self):return digest({'input':self.input,'expected':self.expected,'task_kind':self.task_kind})

class SequentialBenchmark:
    def __init__(self,cases,scorer:Callable[[object,object],float]):self.cases=list(cases);self.scorer=scorer
    def run(self,solver,episode=0):
        groups={}
        for c in self.cases:
            pred=solver(c.input);groups.setdefault(c.split,[]).append(self.scorer(pred,c.expected))
        avg=lambda x:sum(x)/len(x) if x else 0.
        return {'episode':episode,'future_success':avg(groups.get('future',[])),'old_success':avg(groups.get('retention',[])),'security_success':avg(groups.get('security',[]))}
