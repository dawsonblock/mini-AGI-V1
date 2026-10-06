#!/usr/bin/env bash
set -euo pipefail
ENDPOINT="${KVCONTINUAL_GATEWAY_ENDPOINT:-http://127.0.0.1:8088/v1}"
MODEL="${KVCONTINUAL_MODEL_NAME:-local-qwen}"
PROMPT="${QW3_BENCH_PROMPT:-Return exactly the word READY.}"
N="${QW3_BENCH_RUNS:-3}"
python3 - "$ENDPOINT" "$MODEL" "$PROMPT" "$N" <<'PY'
import json,sys,time,urllib.request
base,model,prompt,n=sys.argv[1],sys.argv[2],sys.argv[3],int(sys.argv[4])
lat=[]
for i in range(n):
    body=json.dumps({"model":model,"messages":[{"role":"user","content":prompt}],"max_tokens":16,"temperature":0}).encode()
    req=urllib.request.Request(base.rstrip('/')+'/chat/completions',data=body,headers={'Content-Type':'application/json'})
    t=time.perf_counter()
    with urllib.request.urlopen(req,timeout=300) as r: data=json.loads(r.read())
    dt=time.perf_counter()-t; lat.append(dt)
    print(f"run={i+1} latency_s={dt:.3f} text={data.get('choices',[{}])[0].get('message',{}).get('content','')[:80]!r}")
print(f"mean_latency_s={sum(lat)/len(lat):.3f}")
PY
