from minagi.dream import DiscoveryNode, DiscoveryTree, ReplaySimulator, best_policy


def tree():
    return DiscoveryTree([
        DiscoveryNode('r',None,0), DiscoveryNode('a','r',1),
        DiscoveryNode('b','r',3), DiscoveryNode('a2','a',2),
        DiscoveryNode('b2','b',5)])


def first(eligible,t,revealed,w): return eligible[:w]
def root_then_best(eligible,t,revealed,w):
    # root opens recorded branches in order; then choose the revealed leaf with
    # highest observed score.
    leaves=[x for x in eligible if x!='r']
    return ([max(leaves,key=lambda x:t.nodes[x].score)] if leaves else ['r'])[:w]


def test_replay_never_invents_nodes_and_scores_policy():
    sim=ReplaySimulator(beta_cost=0.1,max_rounds=5)
    r=sim.evaluate(tree(),root_then_best,workers=1)
    assert set(r.revealed) <= set(tree().nodes)
    assert r.attempts <= 4
    assert r.best_score >= 1


def test_best_policy_returns_scored_candidate():
    sim=ReplaySimulator(max_rounds=4)
    got=best_policy({'first':first,'best':root_then_best},[tree()],sim)
    assert got and got[1] in {'first','best'} and got[2]
