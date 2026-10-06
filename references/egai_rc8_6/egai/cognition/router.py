from dataclasses import dataclass

ACTIONS=("retrieve","reason","search","simulate","execute","verify","experiment","ask","branch","halt")

@dataclass
class Budget:
    tokens:int=4096
    tool_calls:int=8
    experiments:int=2
    risk:float=1.0

class CognitiveRouter:
    """Deterministic initial policy; intentionally replaceable by qualified policy candidates."""
    def choose(self, state:dict, budget:Budget):
        if state.get("needs_verification") and budget.tool_calls>0: return "verify"
        if state.get("known_skill"): return "execute"
        if state.get("missing_evidence") and budget.tool_calls>0: return "retrieve"
        if state.get("uncertain") and budget.experiments>0: return "experiment"
        return "reason"
