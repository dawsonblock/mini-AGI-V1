from dataclasses import dataclass, field
from typing import Any
from egai.common.canonical import digest

MECHANISMS=("ignore","remember","revise_belief","create_skill","reuse","compose",
            "modulate","adapt","expand","consolidate","architecture_change")
LEVEL={"ignore":0,"remember":1,"revise_belief":2,"create_skill":3,"modulate":4,
       "reuse":5,"compose":5,"adapt":6,"expand":7,"consolidate":8,"architecture_change":9}
RING_CAP={"R0":1,"R1":3,"R2":4,"R3":5,"R4":6,"R5":8,"R6":9}

@dataclass(frozen=True)
class LearningProposal:
    proposal_id:str
    origin_evidence_root:str
    hypothesis_id:str
    abstraction_id:str
    mechanism:str
    target_components:tuple[str,...]
    expected_gain:float
    expected_transfer:float
    expected_interference:float
    expected_compute:float
    expected_capacity_growth:float
    risk_estimate:float
    demonstrated_ring:str
    training_recipe:dict[str,Any]=field(default_factory=dict)
    evaluation_plan:dict[str,Any]=field(default_factory=dict)
    proposer_digest:str=""

    @property
    def digest(self): return digest(self)

class PlasticityProposer:
    """Proposal-only by construction. No registry/runtime/promotion dependency."""
    VERSION="plasticity-proposer/rules-v1"
    def propose(self, proposal_id, evidence_root, ring, repeated_gap=False,
                skill_possible=True, expected_gain=.05, risk=.1):
        cap=RING_CAP[ring]
        if skill_possible and cap>=3: mech="create_skill"
        elif repeated_gap and cap>=6: mech="adapt"
        elif cap>=5: mech="compose"
        else: mech="remember"
        p=LearningProposal(proposal_id,evidence_root,"H-"+proposal_id,"A-"+proposal_id,
            mech,(),expected_gain,expected_gain,0.0,1.0,0.0,risk,ring,
            {},{"required_ring":ring},self.VERSION)
        return p
