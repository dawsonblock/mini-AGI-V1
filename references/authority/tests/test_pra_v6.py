import hashlib
import numpy as np
import torch

from minagi.pra import (
    PRAExample, ProcessRewardAgent, train_process_reward_agent,
    BoundProcessRewardAgent, save_pra, load_pra,
)
from minagi.verifier import ProcessVerifier, Trace


def _encoder(texts, d=12):
    # Deterministic local encoder suitable for unit tests. It intentionally
    # exposes lexical label cues so the small head can learn in a few epochs.
    out=[]
    for t in texts:
        v=np.zeros(d,dtype=np.float32)
        for tok in t.lower().split():
            h=int(hashlib.sha256(tok.encode()).hexdigest()[:8],16)
            v[h%d]+=1.0
        n=np.linalg.norm(v)
        out.append(v/(n if n else 1))
    return np.stack(out)


def test_trainable_process_reward_agent_binds_to_online_verifier(tmp_path):
    ex=[]
    for i in range(40):
        good=(i%2)==0
        ex.append(PRAExample(
            question="medical test",
            partial_trace="known evidence",
            step=("supported correct safe" if good else "unsupported wrong unsafe"),
            evidence=("guideline support",) if good else (),
            reward_label=int(good), search_label=int(not good),
        ))
    agent=ProcessRewardAgent(12, hidden=24, dropout=0.0)
    r=train_process_reward_agent(agent,_encoder,ex,epochs=30,batch_size=8,lr=1e-2,seed=1)
    assert r.reward_accuracy > 0.9
    bound=BoundProcessRewardAgent(agent,_encoder,search_threshold=0.5)
    pv=ProcessVerifier(bound.scorer, retriever=lambda q,t,s,k:["guideline support"],
                       search_policy=bound.search_policy)
    good=pv.evaluate("medical test",Trace(),"supported correct safe")
    bad=pv.evaluate("medical test",Trace(),"unsupported wrong unsafe")
    assert good.reward > bad.reward

    save_pra(tmp_path/"pra",agent,extra={"dataset":"unit"})
    got,extra=load_pra(tmp_path/"pra")
    assert extra["dataset"]=="unit"
    x=torch.from_numpy(_encoder(["supported correct safe"]))
    assert torch.allclose(agent.probabilities(x)[0], got.probabilities(x)[0], atol=1e-6)
