from dataclasses import dataclass
from enum import IntEnum
class TransferRing(IntEnum): R0_KNOWN=0;R1_NEW_INSTANCE=1;R2_NEW_DIFFICULTY=2;R3_NOVEL_COMPOSITION=3;R4_NEW_TASK_FAMILY=4;R5_NEW_ENVIRONMENT=5;R6_EXTRAPOLATION=6
@dataclass(frozen=True)
class RingScore:
    ring:TransferRing; baseline:float; candidate:float; n:int
    @property
    def delta(self):return self.candidate-self.baseline
def forward_transfer(xs):
    a=[x.delta for x in xs if x.ring>=TransferRing.R1_NEW_INSTANCE];return sum(a)/len(a) if a else 0.0
def transfer_breadth(xs,eps=0.0):return sum(x.delta>eps for x in xs)/len(xs) if xs else 0.0
