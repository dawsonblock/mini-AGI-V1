from dataclasses import dataclass
from egai.common.canonical import digest
from egai.skills.memory import SandboxSkillMemory
from egai.skills.learner import SkillMiner,VerifiedEpisode,SynthesizedSkillMiner
from .loop import CognitiveLoop,CognitiveTrace
from .router import Budget

@dataclass(frozen=True)
class AgentResponse:
    text:str; used_skills:tuple[str,...]; prompt_digest:str; cognitive_actions:tuple[str,...]=()

class SandboxAdaptiveAgent:
    """Legacy-compatible frozen agent; RC8.6 orchestrator instantiates with SynthesizedSkillMiner."""
    def __init__(self,model,skill_memory=None,skill_miner=None):
        self.model=model;self.skills=skill_memory or SandboxSkillMemory();self.miner=skill_miner or SkillMiner(min_distinct_inputs=2)
    @property
    def model_digest(self):return self.model.model_digest
    @property
    def snapshot_digest(self):return self.skills.snapshot_digest
    def solve(self,task,task_kind=''):
        query=task if isinstance(task,str) else str(task);found=self.skills.retrieve(query,task_kind,3)
        skill_text='\n'.join(f'- {p.procedure_text}' for p in found) or '(none)'
        prompt=f"TASK_KIND: {task_kind}\nVERIFIED PROCEDURES:\n{skill_text}\nTASK:\n{query}\nReturn only the answer."
        out=self.model.generate(prompt);return AgentResponse(out,tuple(p.procedure_id for p in found),digest(prompt),('retrieve','reason' if not found else 'execute','verify'))
    def observe(self,e):
        p=self.miner.observe(e)
        if p:self.skills.upsert(p)
        return p
    def export_skill_bundle(self):
        import json
        return json.dumps(self.skills.export_bundle(),sort_keys=True,separators=(',',':')).encode()

class GovernedAdaptiveAgent(SandboxAdaptiveAgent):
    """Controller-mediated cognitive path used by RC8.6 experiments."""
    def __init__(self,model,skill_memory=None,skill_miner=None,loop=None,budget=None):
        super().__init__(model,skill_memory,skill_miner or SynthesizedSkillMiner());self.loop=loop or CognitiveLoop();self.budget=budget or Budget()
    def solve(self,task,task_kind=''):
        query=task if isinstance(task,str) else str(task);found=self.skills.retrieve(query,task_kind,3);trace=CognitiveTrace()
        # Controller gets a minimal state, not benchmark truth.
        state={'missing_evidence':not bool(found),'known_skill':bool(found),'needs_verification':False,'uncertain':not bool(found)}
        first,trace=self.loop.step(state,self.budget,trace)
        actions=[first]
        if first=='retrieve':
            state={'known_skill':bool(found),'missing_evidence':False,'uncertain':not bool(found)}
            second,trace=self.loop.step(state,self.budget,trace);actions.append(second)
        elif first not in ('execute','reason'): actions.append('execute' if found else 'reason')
        skill_text='\n'.join(f'- {p.procedure_text}' for p in found) or '(none)'
        prompt=f"TASK_KIND: {task_kind}\nVERIFIED PROCEDURES:\n{skill_text}\nTASK:\n{query}\nReturn only the answer."
        out=self.model.generate(prompt);actions.append('verify')
        return AgentResponse(out,tuple(p.procedure_id for p in found),digest(prompt),tuple(actions))
