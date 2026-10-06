from dataclasses import dataclass, asdict
from collections import defaultdict
from egai.common.canonical import digest

@dataclass(frozen=True)
class Belief:
    claim_key: str
    proposition: str
    supporting_evidence: tuple[str,...]
    contradicting_evidence: tuple[str,...]
    confidence: float
    status: str
    compiler_version: str
    derivation_receipt: str=""

class BeliefCompiler:
    VERSION="belief-compiler/1"
    def compile(self, evidence):
        # Reference convention: payload={"claim": "...", "stance":"support"|"contradict"}
        groups=defaultdict(lambda: {"support":[],"contradict":[]})
        for e in evidence:
            claim=e.payload.get("claim")
            if not claim: continue
            stance=e.payload.get("stance","support")
            if stance not in groups[claim]: continue
            groups[claim][stance].append(e.record_hash)
        beliefs=[]
        for claim,g in sorted(groups.items()):
            s,c=len(g["support"]),len(g["contradict"])
            conf=(s+1)/(s+c+2)
            status="disputed" if s and c else ("supported" if s else "falsified")
            b=Belief(claim,claim,tuple(g["support"]),tuple(g["contradict"]),conf,status,self.VERSION)
            beliefs.append(Belief(**{**asdict(b),"derivation_receipt":digest(asdict(b))}))
        return beliefs
