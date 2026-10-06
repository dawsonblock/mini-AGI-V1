from dataclasses import dataclass,asdict,replace
import re
from egai.common.canonical import digest

TOKEN_RE=re.compile(r"[a-zA-Z0-9_+-]+")
def toks(s): return set(x.lower() for x in TOKEN_RE.findall(str(s)))

@dataclass(frozen=True)
class LearnedProcedure:
    procedure_id:str; task_kind:str; trigger_text:str; procedure_text:str
    supporting_episode_ids:tuple[str,...]; successes:int; failures:int=0
    verifier_ids:tuple[str,...]=(); verification_receipt_digests:tuple[str,...]=(); distinct_input_digests:tuple[str,...]=()
    supporting_evidence_roots:tuple[str,...]=(); synthesis_digest:str=''
    @property
    def confidence(self): return (self.successes+1)/(self.successes+self.failures+2)
    @property
    def digest(self): return digest(self)

@dataclass(frozen=True)
class SkillLifecycleEvent:
    seq:int; procedure_id:str; old_state:str; new_state:str; reason:str
    @property
    def digest(self): return digest(self)

class SandboxSkillMemory:
    """Ephemeral research memory with negative-evidence lifecycle controls."""
    def __init__(self):
        self._procedures={}
        self._states={}
        self._events=[]
        self._runtime_successes={}
    def _transition(self,pid,new_state,reason):
        old=self._states.get(pid,"active")
        if old==new_state:return
        self._states[pid]=new_state
        self._events.append(SkillLifecycleEvent(len(self._events)+1,pid,old,new_state,reason))
    def upsert(self,p):
        existing=self._procedures.get(p.procedure_id)
        self._procedures[p.procedure_id]=p
        if p.procedure_id not in self._states:self._states[p.procedure_id]="active"
        # New independent support may rehabilitate a degraded research skill, never a retired one.
        if existing and self._states.get(p.procedure_id)=="degraded" and p.successes>existing.successes:
            self._transition(p.procedure_id,"active","new independently verified support")
    def mark_success(self,procedure_id):
        if procedure_id in self._procedures:
            self._runtime_successes[procedure_id]=self._runtime_successes.get(procedure_id,0)+1
    def mark_failure(self,procedure_id):
        p=self._procedures.get(procedure_id)
        if not p:return
        p=replace(p,failures=p.failures+1); self._procedures[procedure_id]=p
        # Conservative research lifecycle: repeated negative evidence degrades then retires.
        if p.failures >= p.successes + 2:
            self._transition(procedure_id,"retired","negative evidence exceeded support")
        elif p.failures >= max(2,p.successes):
            self._transition(procedure_id,"degraded","negative evidence reached support")
    def state(self,procedure_id): return self._states.get(procedure_id,"unknown")
    def lifecycle_events(self): return tuple(self._events)
    def health(self):
        counts={"active":0,"degraded":0,"retired":0}
        for pid in self._procedures: counts[self._states.get(pid,"active")]+=1
        return counts
    def all(self): return tuple(sorted(self._procedures.values(),key=lambda p:p.procedure_id))
    def retrieve(self,query,task_kind='',k=3):
        q=toks(query); scored=[]
        for p in self._procedures.values():
            state=self._states.get(p.procedure_id,"active")
            if state=="retired": continue
            if task_kind and p.task_kind!=task_kind: continue
            pt=toks(p.trigger_text+' '+p.procedure_text); overlap=len(q&pt)/max(len(q|pt),1)
            kind_bonus=.65 if task_kind and p.task_kind==task_kind else 0.
            state_penalty=.35 if state=="degraded" else 0.
            reliability=max(0.,p.confidence - .1*p.failures - state_penalty)
            score=kind_bonus + overlap + .25*reliability
            if score>0: scored.append((score,p))
        return [p for _,p in sorted(scored,key=lambda x:(-x[0],x[1].procedure_id))[:k]]
    @property
    def snapshot_digest(self):
        return digest({
            "procedures":[asdict(p) for p in self.all()],
            "states":dict(sorted(self._states.items())),
            "events":[asdict(e) for e in self._events],
            "runtime_successes":dict(sorted(self._runtime_successes.items())),
        })
    def export_bundle(self):
        return {
            "procedures":[asdict(p) for p in self.all()],
            "states":dict(sorted(self._states.items())),
            "events":[asdict(e) for e in self._events],
            "runtime_successes":dict(sorted(self._runtime_successes.items())),
            "health":self.health(),
        }
