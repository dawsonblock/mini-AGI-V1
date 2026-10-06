from .objects import TransferEvaluation
RINGS=("R0","R1","R2","R3","R4","R5","R6")
RING_DESC={"R0":"same episode","R1":"new instance","R2":"new difficulty","R3":"new composition","R4":"new task family","R5":"new domain","R6":"extrapolation"}

def highest_ring(results:dict[str,bool])->str:
    highest="R0"
    for r in RINGS:
        if results.get(r,False):highest=r
        else:break
    return highest

class TransferEvaluator:
    def evaluate(self,abstraction_id,ring_scores,threshold=.5):
        results={r:float(ring_scores.get(r,0.))>=threshold for r in RINGS}
        # Enforce contiguous evidence: R4 cannot count if R2 failed.
        seen_failure=False
        for r in RINGS:
            if seen_failure:results[r]=False
            if not results[r]:seen_failure=True
        return TransferEvaluation(abstraction_id,results,{r:float(ring_scores.get(r,0.)) for r in RINGS})
