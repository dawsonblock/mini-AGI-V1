from minagi.dream import DiscoveryNode, DiscoveryTree, ReplaySimulator, qualify_policy


def world(a, b):
    return DiscoveryTree([
        DiscoveryNode('r', None, 0),
        DiscoveryNode('a', 'r', a),
        DiscoveryNode('b', 'r', b),
    ])


def baseline(eligible, tree, revealed, workers):
    return ['r']


def deep(eligible, tree, revealed, workers):
    # After opening a root branch, continue it before opening another.
    nonroot=[x for x in eligible if x != 'r']
    return nonroot[:workers] or ['r']


def test_replay_policy_requires_holdout_evidence():
    # Deep has no advantage on these shallow worlds, so it cannot be promoted
    # just because selection order happens to favor it.
    sim=ReplaySimulator(max_rounds=2)
    r=qualify_policy({'baseline': baseline, 'deep': deep},
                     [world(1,0), world(2,0)],
                     [world(1,0), world(2,0)], sim,
                     baseline='baseline', require_lcb=False,
                     min_holdout_gain=.1)
    assert not r.promoted


def test_replay_policy_can_pass_paired_holdout_gate():
    def stop(eligible, tree, revealed, workers):
        return []
    def explore(eligible, tree, revealed, workers):
        return ['r'] if 'a' not in revealed else []
    sim=ReplaySimulator(max_rounds=1)
    train=[world(2,0), world(3,0)]
    hold=[world(2,0), world(3,0), world(4,0)]
    r=qualify_policy({'baseline': stop, 'explore': explore}, train, hold, sim,
                     baseline='baseline', min_holdout_gain=.5, require_lcb=True)
    assert r.name == 'explore'
    assert r.promoted
    assert r.holdout_delta >= 2
