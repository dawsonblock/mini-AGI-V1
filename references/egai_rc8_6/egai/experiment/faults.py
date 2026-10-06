from dataclasses import dataclass

@dataclass(frozen=True)
class FaultPlan:
    mutate_model_after_episode:int|None=None
    corrupt_journal_after_episode:int|None=None
    fail_backend_after_calls:int|None=None

class FaultInjector:
    def __init__(self, plan=None): self.plan=plan or FaultPlan(); self.calls=0
    def before_model_call(self):
        self.calls+=1
        if self.plan.fail_backend_after_calls is not None and self.calls>self.plan.fail_backend_after_calls:
            raise RuntimeError('injected backend failure')
