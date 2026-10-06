from dataclasses import dataclass

@dataclass(frozen=True)
class AblationResult:
    name:str
    future_success:float
    retention_success:float
    security_success:float
    delta_future:float

def run_ablations(evaluate_fn,variants):
    """evaluate_fn(name, variant)->dict with future/retention/security success."""
    rows=[];baseline=None
    for name,variant in variants:
        m=evaluate_fn(name,variant)
        if baseline is None: baseline=float(m['future_success'])
        rows.append(AblationResult(name,float(m['future_success']),float(m.get('retention_success',0)),float(m.get('security_success',0)),float(m['future_success'])-baseline))
    return tuple(rows)
