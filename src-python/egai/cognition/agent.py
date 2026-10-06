from dataclasses import dataclass
from egai.common.canonical import digest
from egai.skills.memory import SandboxSkillMemory
from egai.skills.learner import SkillMiner,VerifiedEpisode

@dataclass(frozen=True)
class AgentResponse:
    text:str; used_skills:tuple[str,...]; prompt_digest:str

class SandboxAdaptiveAgent:
    """Frozen-model research agent. Learning occurs only in ephemeral skill memory until exported for qualification."""
    def __init__(self,model,skill_memory=None,skill_miner=None):
        self.model=model;self.skills=skill_memory or SandboxSkillMemory();self.miner=skill_miner or SkillMiner()
    @property
    def model_digest(self):return self.model.model_digest
    @property
    def snapshot_digest(self):return self.skills.snapshot_digest
    def solve(self,task,task_kind=''):
        query=task if isinstance(task,str) else str(task)
        found=self.skills.retrieve(query,task_kind,3)
        skill_text='\n'.join(f'- {p.procedure_text}' for p in found) or '(none)'
        prompt=f"TASK_KIND: {task_kind}\nVERIFIED PROCEDURES:\n{skill_text}\nTASK:\n{query}\nReturn only the answer."
        out=self.model.generate(prompt)
        return AgentResponse(out,tuple(p.procedure_id for p in found),digest(prompt))
    def observe(self,e:VerifiedEpisode):
        p=self.miner.observe(e)
        if p:self.skills.upsert(p)
        return p
    def export_skill_bundle(self):
        import json
        from dataclasses import asdict
        return json.dumps([asdict(p) for p in self.skills.all()],sort_keys=True,separators=(',',':')).encode()
