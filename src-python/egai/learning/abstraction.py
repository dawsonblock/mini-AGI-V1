from .objects import Hypothesis,AbstractionCandidate,Counterexample
from egai.common.canonical import digest

class AbstractionEngine:
    """Conservative symbolic abstraction over repeated verified procedure evidence."""
    def propose(self,task_kind,procedure_text,supporting_evidence):
        if len(set(supporting_evidence))<2: raise ValueError('abstraction requires repeated support')
        hid='H-'+digest({'kind':task_kind,'procedure':procedure_text}).split(':')[1][:16]
        aid='A-'+digest({'hypothesis':hid,'scope':task_kind}).split(':')[1][:16]
        h=Hypothesis(hid,f'For {task_kind}: {procedure_text}',tuple(sorted(set(supporting_evidence))))
        a=AbstractionCandidate(aid,(hid,),procedure_text,task_kind)
        return h,a

class CounterexampleSearch:
    def from_failures(self,abstraction,failures):
        out=[]
        for i,f in enumerate(failures):
            if getattr(f,'task_kind',None)==abstraction.scope:
                cid='CE-'+digest({'a':abstraction.abstraction_id,'i':i,'input':getattr(f,'input',str(f))}).split(':')[1][:16]
                out.append(Counterexample(cid,abstraction.abstraction_id,{'input':getattr(f,'input',str(f))},grounded=True))
        return tuple(out)
