import json
from dataclasses import asdict
from .engine import TaskCase

ALLOWED_SPLITS=('future','retention','security')

def encode_cases(cases):
    rows=[];seen=set()
    for c in sorted(cases,key=lambda x:x.case_id):
        if c.case_id in seen:raise ValueError('duplicate case id')
        seen.add(c.case_id)
        d=asdict(c);d['tags']=list(c.tags);rows.append(d)
    return json.dumps(rows,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()

def decode_cases(data):
    rows=json.loads(data);out=[]
    for d in rows:
        d=dict(d);d['tags']=tuple(d.get('tags',[]));out.append(TaskCase(**d))
    return out

def persist_benchmark_sets(store,cases):
    groups={s:[] for s in ALLOWED_SPLITS}
    for c in cases:
        if c.split not in groups:raise ValueError('benchmark case split must be future/retention/security')
        groups[c.split].append(c)
    if not groups['future'] or not groups['retention']:raise ValueError('future and retention sets required')
    return tuple(store.put_bytes(encode_cases(groups[s])) for s in ALLOWED_SPLITS)

def load_benchmark_sets(store,benchmark):
    digests=(benchmark.task_set_digest,benchmark.retention_set_digest,benchmark.security_set_digest)
    expected=('future','retention','security');out=[]
    for split,d in zip(expected,digests):
        xs=decode_cases(store.get_bytes(d))
        if any(c.split!=split for c in xs):raise ValueError(f'case split mismatch in {split} artifact')
        out.extend(xs)
    return out
