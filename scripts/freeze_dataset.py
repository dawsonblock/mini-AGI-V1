#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src-python'))
from egai.common.canonical import sha256_bytes
from minagi.v161.dataset_manifest import DatasetMember, DatasetMembershipManifest, DatasetPartitionSet

def member(row):
    b=json.dumps(row,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()
    return DatasetMember(str(row['id']),str(row['family']),sha256_bytes(b),str(row.get('source','local')),str(row.get('generator','manual')))
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--input',required=True); ap.add_argument('--output',required=True); ap.add_argument('--family-disjoint-hidden',action='store_true'); a=ap.parse_args()
    rows=[json.loads(x) for x in Path(a.input).read_text().splitlines() if x.strip()]
    parts={n:[r for r in rows if r['split']==n] for n in ('train','validation','hidden')}
    ps=DatasetPartitionSet(*(DatasetMembershipManifest(n,tuple(member(r) for r in parts[n])) for n in ('train','validation','hidden')),require_family_disjoint_hidden=a.family_disjoint_hidden)
    out={'schema':'mini-agi-v16.1-frozen-dataset-v1','partition_set_digest':ps.digest,'proof':ps.proof(),'partitions':{n:{'digest':getattr(ps,n).digest,'members':[m.__dict__ for m in getattr(ps,n).members]} for n in ('train','validation','hidden')}}
    Path(a.output).write_text(json.dumps(out,indent=2,sort_keys=True)); print(json.dumps({'partition_set_digest':ps.digest,'verified':True},indent=2))
if __name__=='__main__': main()
