from minagi.verifier import ProcessVerifier, BeamReasoner


def test_process_verifier_guides_beam_and_retrieves():
    retrieved=[]
    def retr(q,t,s,k): retrieved.append(s); return ['evidence:'+s]
    def score(q,t,s,e):
        return 2.0 if s in ('good','done') else -1.0
    def propose(q,t,b):
        return ['done'] if 'good' in t else ['bad','good']
    v=ProcessVerifier(score,retriever=retr)
    r=BeamReasoner(propose,v,beam_width=2,branching=2,max_steps=2,
                   complete=lambda q,t:'done' in t).run('q')
    assert [s.text for s in r.steps] == ['good','done']
    assert all(s.searched for s in r.steps)
    assert retrieved
