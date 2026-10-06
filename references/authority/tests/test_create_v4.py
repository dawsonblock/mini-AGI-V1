import json
from minagi.create import create


def test_create_preserves_full_depth_policy(tmp_path):
    out=tmp_path/'w'
    create(str(out),verbose=False,force=True,experts=2,resident=1,d_ff=16,
           d_model=8,trunk_d_ff=16,n_head=2,block=16,max_steps=3,
           n_prelude=1,n_recur=1,n_coda=0,min_steps=2,
           train_steps_mean=2.5,bptt_window=2,halt_prior=.2,
           halt_thresh=.8,halt_freeze=True,top_k=1,
           pool_hierarchical_index=True,pool_index_group_size=2,
           pool_index_top_groups=1,pool_index_refresh=7,
           pool_index_max_candidates=2)
    m=json.load(open(out/'manifest.json'))['cfg']
    assert m['min_steps']==2 and m['train_steps_mean']==2.5
    assert m['bptt_window']==2 and m['halt_prior']==.2
    assert m['halt_thresh']==.8 and m['halt_freeze'] is True
    assert m['tokenizer']['kind']=='byte'
    assert m['pool_hierarchical_index'] is True
    assert m['pool_index_group_size']==2 and m['pool_index_top_groups']==1
    assert m['pool_index_refresh']==7 and m['pool_index_max_candidates']==2
