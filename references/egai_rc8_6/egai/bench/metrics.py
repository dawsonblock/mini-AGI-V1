from dataclasses import dataclass

@dataclass(frozen=True)
class BenchmarkPoint:
    episode:int
    future_success:float
    old_success:float
    calibration_error:float
    compute:float
    capacity_growth:float

def fte(before:BenchmarkPoint, after:BenchmarkPoint, verified_experience:float,
        lam=.001, mu=.01):
    denom=verified_experience + lam*max(after.compute-before.compute,0) + mu*max(after.capacity_growth-before.capacity_growth,0)
    return (after.future_success-before.future_success)/denom if denom else 0.0
