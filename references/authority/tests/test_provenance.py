from pathlib import Path
from minagi.provenance import sha256_file, content_split, source_records, build_run_manifest


def test_content_hash_split_prevents_renamed_duplicate_leakage(tmp_path):
    a=tmp_path/'a.txt'; b=tmp_path/'renamed.txt'; c=tmp_path/'c.txt'
    a.write_text('same'); b.write_text('same'); c.write_text('different')
    recs=source_records([a,b,c], tmp_path, val_permille=500, deduplicate=True)
    ra, rb = recs[0], recs[2] if recs[1]['path']=='c.txt' else recs[1]
    same=[r for r in recs if r['sha256']==sha256_file(a)]
    assert len(same)==2
    assert len({r['split'] for r in same}) == 1
    assert sum(r['included'] for r in same) == 1


def test_content_split_is_deterministic():
    d='a'*64
    assert content_split(d, 5) == content_split(d, 5)


def test_run_manifest_binds_source_and_config(tmp_path):
    (tmp_path/'x.py').write_text('x=1\n')
    cfg=tmp_path/'config.yaml'; cfg.write_text('a: 1\n')
    m=build_run_manifest(tmp_path, config_path=cfg)
    assert len(m['source_tree_sha256']) == 64
    assert len(m['manifest_sha256']) == 64
    assert m['config']['sha256'] == sha256_file(cfg)
