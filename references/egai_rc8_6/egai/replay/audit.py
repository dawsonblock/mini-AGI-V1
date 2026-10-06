from dataclasses import dataclass,asdict
from .world import ReplayWorld,ReplayNode
from egai.common.canonical import digest

@dataclass(frozen=True)
class ReplayAudit:
    audit_id:str
    worlds:int
    grounded_transitions:int
    ungrounded_transitions:int
    prefix_violations:int
    grounded_coverage:float
    auditor_version:str="replay-auditor/1"
    @property
    def digest(self): return digest(self)

class ReplayAuditor:
    """Audits that recorded counterfactual transitions are grounded and controller views remain prefix-causal."""
    def __init__(self,signer,verifier): self.signer=signer; self.verifier=verifier
    def audit_traces(self,audit_id,traces):
        grounded=ungrounded=violations=0
        for tr in traces:
            root=ReplayNode(f"{tr.case_id}:root",0,{"input":tr.input_text,"task_kind":tr.task_kind},{},{"state":"start"})
            attempt_action={"kind":"answer","output":tr.attempted_output}
            repair_action={"kind":"answer","output":tr.repaired_output}
            nodes=[
                root,
                ReplayNode(f"{tr.case_id}:attempt",1,{},attempt_action,{"score":tr.attempted_score},root.node_id),
                ReplayNode(f"{tr.case_id}:repair",1,{},repair_action,{"score":tr.repaired_score},root.node_id),
            ]
            w=ReplayWorld(f"world:{tr.case_id}",nodes,self.signer,self.verifier)
            cap=w.issue_capability(0)
            visible=w.view(cap)
            if len(visible)!=1 or visible[0].node_id!=root.node_id: violations+=1
            for action in (attempt_action,repair_action):
                res=w.step(root.node_id,action,cap)
                if res.grounded: grounded+=1
                else: ungrounded+=1
        total=grounded+ungrounded
        return ReplayAudit(audit_id,len(traces),grounded,ungrounded,violations,grounded/total if total else 1.0)
