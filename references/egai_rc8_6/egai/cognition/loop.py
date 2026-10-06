from dataclasses import dataclass,field
from .router import Budget,CognitiveRouter
@dataclass
class CognitiveTrace:
    steps:list[dict]=field(default_factory=list)
class CognitiveLoop:
    def __init__(self,router=None):self.router=router or CognitiveRouter()
    def step(self,state,budget,trace=None):
        trace=trace or CognitiveTrace();action=self.router.choose(state,budget);trace.steps.append({'action':action,'state_keys':sorted(state)})
        return action,trace
