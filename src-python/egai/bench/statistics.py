from dataclasses import dataclass
import random

@dataclass(frozen=True)
class EffectEstimate:
    mean_gain:float; ci_low:float; ci_high:float; n:int

def paired_bootstrap(baseline,candidate,samples=2000,seed=0,alpha=.05):
    if len(baseline)!=len(candidate) or not baseline:raise ValueError('paired non-empty scores required')
    diffs=[c-b for b,c in zip(baseline,candidate)];rng=random.Random(seed);means=[];n=len(diffs)
    for _ in range(samples): means.append(sum(diffs[rng.randrange(n)] for _ in range(n))/n)
    means.sort();lo=means[int((alpha/2)*samples)];hi=means[min(samples-1,int((1-alpha/2)*samples))]
    return EffectEstimate(sum(diffs)/n,lo,hi,n)
