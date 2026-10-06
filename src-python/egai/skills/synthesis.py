from dataclasses import dataclass
from abc import ABC,abstractmethod
import json
from egai.common.canonical import digest

@dataclass(frozen=True)
class SynthesisExample:
    episode_id:str; task_kind:str; input_text:str; attempted_output:str; repaired_output:str
    evidence_root:str; verification_receipt_digest:str

@dataclass(frozen=True)
class SynthesizedSkill:
    task_kind:str; trigger_text:str; procedure_text:str; rationale:str
    supporting_evidence:tuple[str,...]; verification_receipts:tuple[str,...]; synthesizer_id:str
    @property
    def digest(self): return digest(self)

class SkillSynthesizer(ABC):
    @abstractmethod
    def synthesize(self,examples:tuple[SynthesisExample,...])->SynthesizedSkill|None: ...

class PatternSkillSynthesizer(SkillSynthesizer):
    """Deterministic reference synthesizer for falsifiable experiments; no benchmark hint is consumed."""
    VERSION='pattern-synthesizer/1'
    def synthesize(self,examples):
        if len(examples)<2:return None
        kind=examples[0].task_kind
        if any(x.task_kind!=kind for x in examples):return None
        pairs=[(x.input_text,x.repaired_output) for x in examples]
        proc=None;trigger=kind or 'matching task';why=''
        if all(out==inp[::-1] for inp,out in pairs): proc='Reverse characters.'; why='all independently verified repairs equal exact character reversal'
        elif all(out==inp.upper() for inp,out in pairs): proc='Convert the input text to uppercase.'; why='all independently verified repairs equal uppercase transform'
        elif all(out==inp.lower() for inp,out in pairs): proc='Convert the input text to lowercase.'; why='all independently verified repairs equal lowercase transform'
        elif all(out==''.join(sorted(inp)) for inp,out in pairs): proc='Sort the input characters in ascending order.'; why='all independently verified repairs equal sorted characters'
        if not proc:return None
        return SynthesizedSkill(kind,trigger,proc,why,tuple(sorted(set(x.evidence_root for x in examples if x.evidence_root))),tuple(sorted(set(x.verification_receipt_digest for x in examples if x.verification_receipt_digest))),self.VERSION)

class ModelSkillSynthesizer(SkillSynthesizer):
    """Proposal-only LLM synthesizer. Output remains untrusted until independent qualification."""
    VERSION='model-synthesizer/1'
    def __init__(self,model): self.model=model
    def synthesize(self,examples):
        rows=[{'task_kind':e.task_kind,'input':e.input_text,'attempt':e.attempted_output,'verified_repair':e.repaired_output} for e in examples]
        prompt='Infer one reusable procedure from these verified repair trajectories. Return strict JSON with trigger, procedure, rationale. Do not mention examples.\\n'+json.dumps(rows,sort_keys=True)
        try:d=json.loads(self.model.generate(prompt))
        except Exception:return None
        if not all(str(d.get(k,'')).strip() for k in ('trigger','procedure','rationale')):return None
        return SynthesizedSkill(examples[0].task_kind,str(d['trigger']),str(d['procedure']),str(d['rationale']),tuple(sorted(set(e.evidence_root for e in examples if e.evidence_root))),tuple(sorted(set(e.verification_receipt_digest for e in examples if e.verification_receipt_digest))),self.VERSION+':'+self.model.model_digest)
