from dataclasses import dataclass,asdict,replace
import re
from egai.common.canonical import digest

TOKEN_RE=re.compile(r"[a-zA-Z0-9_+-]+")
def toks(s): return set(x.lower() for x in TOKEN_RE.findall(str(s)))

@dataclass(frozen=True)
class LearnedProcedure:
    procedure_id:str; task_kind:str; trigger_text:str; procedure_text:str
    supporting_episode_ids:tuple[str,...]; successes:int; failures:int=0
    @property
    def confidence(self): return (self.successes+1)/(self.successes+self.failures+2)
    @property
    def digest(self): return digest(self)

class SandboxSkillMemory:
    """Ephemeral research memory; cannot mutate the production SkillRegistry."""
    def __init__(self): self._procedures={}
    def upsert(self,p): self._procedures[p.procedure_id]=p
    def mark_failure(self,procedure_id):
        p=self._procedures.get(procedure_id)
        if p:self._procedures[procedure_id]=replace(p,failures=p.failures+1)
    def mark_success(self,procedure_id):
        p=self._procedures.get(procedure_id)
        if p:self._procedures[procedure_id]=replace(p,successes=p.successes+1)
    def health(self):
        """Skill-health summary for the governed benchmark report.

        degraded: procedures that have failed more often than they have
        succeeded (confidence below 0.5 — the same condition stated on
        the counters, so no threshold is invented here).
        retired: always 0 — this ephemeral memory has no retirement
        mechanism; the field exists for report-shape parity with a
        production registry that may retire skills."""
        degraded=sum(1 for p in self._procedures.values() if p.failures>p.successes)
        return {'degraded':degraded,'retired':0}
    def all(self): return tuple(sorted(self._procedures.values(),key=lambda p:p.procedure_id))
    def retrieve(self,query,task_kind='',k=3):
        q=toks(query); scored=[]
        for p in self._procedures.values():
            if task_kind and p.task_kind!=task_kind: continue
            pt=toks(p.trigger_text+' '+p.procedure_text); overlap=len(q&pt)/max(len(q|pt),1)
            # Matching task kind is itself meaningful; lexical overlap is only a tie-breaker.
            kind_bonus=.65 if task_kind and p.task_kind==task_kind else 0.
            score=kind_bonus + overlap + .25*p.confidence
            if score>0: scored.append((score,p))
        return [p for _,p in sorted(scored,key=lambda x:(-x[0],x[1].procedure_id))[:k]]
    @property
    def snapshot_digest(self): return digest([asdict(p) for p in self.all()])
