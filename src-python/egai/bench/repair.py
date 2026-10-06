from abc import ABC, abstractmethod
from dataclasses import dataclass
from egai.common.canonical import digest

@dataclass(frozen=True)
class Repair:
    case_id:str
    output_text:str
    provider_id:str
    provenance_digest:str

class RepairProvider(ABC):
    @abstractmethod
    def repair(self,case,attempted_output:str)->Repair: ...

class ReferenceRepairProvider(RepairProvider):
    """Synthetic/test-only oracle. Keeps answer keys outside the learner and labels provenance explicitly."""
    def __init__(self,provider_id='synthetic-reference-oracle'): self.provider_id=provider_id
    def repair(self,case,attempted_output):
        return Repair(case.case_id,str(case.expected),self.provider_id,digest({'provider':self.provider_id,'case':case.case_digest}))

class CallableRepairProvider(RepairProvider):
    def __init__(self,provider_id,fn): self.provider_id=provider_id; self.fn=fn
    def repair(self,case,attempted_output):
        out=str(self.fn(case.input,attempted_output,case.task_kind))
        return Repair(case.case_id,out,self.provider_id,digest({'provider':self.provider_id,'case':case.case_digest,'repair':out}))


class JSONLRepairProvider(RepairProvider):
    """Loads externally produced repairs keyed by case_id. The repair file is separate from benchmark truth and learner context."""
    def __init__(self,path,provider_id='external-repair-jsonl'):
        import json
        from pathlib import Path
        self.provider_id=provider_id; self.rows={}
        for line in Path(path).read_text(encoding='utf-8').splitlines():
            if not line.strip(): continue
            d=json.loads(line); cid=str(d['case_id'])
            if cid in self.rows: raise ValueError('duplicate repair case_id: '+cid)
            self.rows[cid]=str(d['repair'])
    def repair(self,case,attempted_output):
        if case.case_id not in self.rows: raise KeyError('no external repair for '+case.case_id)
        out=self.rows[case.case_id]
        return Repair(case.case_id,out,self.provider_id,digest({'provider':self.provider_id,'case_id':case.case_id,'repair':out}))
