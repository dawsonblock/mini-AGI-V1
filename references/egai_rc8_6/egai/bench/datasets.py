import json
from pathlib import Path
from dataclasses import asdict
from egai.common.canonical import canonical_bytes, sha256_bytes
from .engine import TaskCase

SPLITS=('experience','future','retention','security')

def cases_bytes(cases, split=None):
    rows=[]
    for c in cases:
        if split is None or c.split==split:
            rows.append(asdict(c))
    rows=sorted(rows,key=lambda x:x['case_id'])
    return canonical_bytes(rows)

def cases_digest(cases, split=None): return sha256_bytes(cases_bytes(cases,split))

def write_jsonl(path,cases):
    p=Path(path); p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('w',encoding='utf-8') as f:
        for c in cases: f.write(json.dumps(asdict(c),sort_keys=True,ensure_ascii=False)+'\n')

def read_jsonl(path):
    out=[]
    with Path(path).open(encoding='utf-8') as f:
        for i,line in enumerate(f,1):
            if not line.strip(): continue
            d=json.loads(line); d['tags']=tuple(d.get('tags',[])); d['metadata']=d.get('metadata',{})
            c=TaskCase(**d)
            if c.split not in SPLITS: raise ValueError(f'line {i}: invalid split {c.split!r}')
            out.append(c)
    ids=[c.case_id for c in out]
    if len(ids)!=len(set(ids)): raise ValueError('duplicate case ids')
    return out
