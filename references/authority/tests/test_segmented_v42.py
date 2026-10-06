import torch
from minagi.segmented import SegmentedCausalStepper


class Pool:
    admission_mode='causal_prefix_vote'
    admission_prefix_tokens=64


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.w=torch.nn.Parameter(torch.tensor(1.0)); self.pool=Pool(); self.calls=[]
    def empty_caches(self): return [{'k':None,'v':None}]
    def pool_aux(self): return torch.tensor(0.0)
    def pool_balance(self): return torch.tensor(0.0)
    def forward(self,x,y,caches=None,pos_offset=0):
        self.calls.append((int(pos_offset),int(x.shape[1]),self.pool.admission_mode,self.pool.admission_prefix_tokens))
        pred=self.w*x.float(); loss=((pred-y.float())**2).mean()
        # populate a cache tensor so the stepper exercises detach_caches
        caches[0]['k']=pred.view(1,1,-1,1); caches[0]['v']=pred.view(1,1,-1,1)
        return pred.unsqueeze(-1),loss


def test_segmented_step_updates_at_safe_boundaries_and_restores_policy():
    m=Tiny(); opt=torch.optim.SGD(m.parameters(),lr=.01)
    x=torch.tensor([[1,2,3,4]]); y=torch.tensor([[2,4,6,8]])
    r=SegmentedCausalStepper(m,opt,segment_tokens=1,clip=1).step(x,y)
    assert r.segments==4 and r.exact_token_admission
    assert [c[0] for c in m.calls] == [0,1,2,3]
    assert all(c[2:] == ('prefix_causal',1) for c in m.calls)
    assert m.pool.admission_mode=='causal_prefix_vote' and m.pool.admission_prefix_tokens==64
