import pytest
from minagi.verifier import ProcessVerifier, BeamReasoner, Trace, VerifiedStep


def test_mean_scoring_avoids_positive_reward_length_bias():
    short = Trace([VerifiedStep('done', .9)])
    long = Trace([VerifiedStep('a', .8), VerifiedStep('b', .8), VerifiedStep('c', .8)])
    assert long.score > short.score          # legacy sum has the bias
    assert short.aggregate('mean') > long.aggregate('mean')


def test_completed_trace_is_kept_without_forced_expansion():
    def proposer(q, t, b):
        return ['done', 'padding'] if not t else ['padding']
    v = ProcessVerifier(lambda q,t,s,e: 1.0 if s == 'done' else .6)
    r = BeamReasoner(proposer, v, beam_width=2, branching=2, max_steps=4,
                     complete=lambda q,t: t.endswith('done'),
                     score_mode='mean').run('q')
    assert r.text == 'done'


def test_nonfinite_reward_is_rejected():
    v = ProcessVerifier(lambda *a: float('nan'))
    with pytest.raises(ValueError):
        v.evaluate('q', Trace(), 'x')
