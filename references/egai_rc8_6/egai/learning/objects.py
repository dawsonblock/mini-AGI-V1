from dataclasses import dataclass
@dataclass(frozen=True)
class Hypothesis:
    hypothesis_id:str;statement:str;supporting_evidence:tuple[str,...];counter_evidence:tuple[str,...]=();status:str='untested'
@dataclass(frozen=True)
class AbstractionCandidate:
    abstraction_id:str;hypothesis_ids:tuple[str,...];principle:str;scope:str
@dataclass(frozen=True)
class Counterexample:
    counterexample_id:str;abstraction_id:str;task_spec:dict;grounded:bool=False
@dataclass(frozen=True)
class TransferEvaluation:
    abstraction_id:str;ring_results:dict[str,bool];scores:dict[str,float]
@dataclass(frozen=True)
class CapabilityGap:
    gap_id:str;description:str;repeated_failures:int;available_skill_failed:bool;transfer_scope:str
