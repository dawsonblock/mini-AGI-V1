#!/usr/bin/env python3
from __future__ import annotations
import argparse, gc, hashlib, json, os, random, sys
from dataclasses import asdict
from pathlib import Path
import yaml

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src-python"))

from egai.common.canonical import digest, sha256_bytes
from egai.common.crypto import Ed25519Signer, Ed25519Verifier
from minagi.platforms.colab.environment import probe_environment
from minagi.platforms.colab.storage import ColabStorage
from minagi.platforms.cuda.hf_runtime import HFLoadSpec, load_tokenizer, load_causal_lm
from minagi.platforms.cuda.peft_trainer import LoraTrainSpec, train_lora
from minagi.v161.dataset_manifest import DatasetMember, DatasetMembershipManifest, DatasetPartitionSet
from minagi.v161.evaluator_registry import EvaluatorArtifact, EvaluatorRegistry
from minagi.v161.evaluators import exact_match, retention_score, security_regression
from minagi.v161.campaign_plan import ColabCampaignPlanV161
from minagi.v161.executed_run import ExecutedRunReceiptV161
from minagi.v161.runtime_closure3 import sha256_path

ZERO="sha256:"+"0"*64

def load_rows(path: Path):
    rows=[]
    for line in path.read_text().splitlines():
        if line.strip(): rows.append(json.loads(line))
    return rows

def member(row):
    payload=json.dumps(row,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode()
    return DatasetMember(str(row["id"]),str(row["family"]),sha256_bytes(payload),str(row.get("source","local")),str(row.get("generator","manual")))

def generate(model,tokenizer,prompt,max_new_tokens):
    import torch
    with torch.inference_mode():
        device=next(model.parameters()).device
        batch=tokenizer(prompt,return_tensors="pt")
        batch={k:v.to(device) for k,v in batch.items()}
        out=model.generate(**batch,max_new_tokens=max_new_tokens,do_sample=False,pad_token_id=tokenizer.eos_token_id)
        text=tokenizer.decode(out[0,batch["input_ids"].shape[1]:],skip_special_tokens=True).strip()
        return text

def evaluate(model,tokenizer,rows,max_new_tokens,scorer):
    scores=[]; outputs=[]
    for row in rows:
        pred=generate(model,tokenizer,str(row["prompt"]),max_new_tokens)
        score=float(scorer(pred,str(row["expected"])))
        scores.append(score); outputs.append({"id":row["id"],"prediction":pred,"expected":row["expected"],"score":score})
    return (sum(scores)/len(scores) if scores else 0.0),outputs

def model_identity(model,spec):
    commit=str(getattr(getattr(model,"config",None),"_commit_hash","") or spec.revision)
    cfg=getattr(model,"config",None)
    cfgdoc=cfg.to_dict() if cfg is not None and hasattr(cfg,"to_dict") else {}
    return digest({"model_id":spec.model_id,"resolved_revision":commit,"config":cfgdoc})

def tokenizer_identity(tokenizer,spec):
    return digest({"model_id":spec.model_id,"revision":str(getattr(tokenizer,"_commit_hash","") or spec.revision),"class":tokenizer.__class__.__name__,"vocab_size":len(tokenizer)})

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--config",default="configs/smoke.yaml"); ap.add_argument("--storage",default="/content/minagi_work"); args=ap.parse_args()
    cfg=yaml.safe_load((ROOT/args.config).read_text())
    storage=ColabStorage(args.storage); env=probe_environment(str(storage.root));
    if cfg.get("require_cuda",True) and not env.cuda_available: raise SystemExit("CUDA GPU required by this campaign config")
    dataset=ROOT/cfg["dataset"]
    rows=load_rows(dataset); train=[r for r in rows if r["split"]=="train"]; val=[r for r in rows if r["split"]=="validation"]; hidden=[r for r in rows if r["split"]=="hidden"]
    parts=DatasetPartitionSet(DatasetMembershipManifest("train",tuple(member(r) for r in train)),DatasetMembershipManifest("validation",tuple(member(r) for r in val)),DatasetMembershipManifest("hidden",tuple(member(r) for r in hidden)),require_family_disjoint_hidden=bool(cfg.get("require_family_disjoint_hidden",False)))
    registry=EvaluatorRegistry(storage.root/"evidence"/"evaluators")
    arts=[EvaluatorArtifact.from_callable("exact-match-v1",exact_match),EvaluatorArtifact.from_callable("retention-v1",retention_score),EvaluatorArtifact.from_callable("security-v1",security_regression)]
    for a in arts: registry.register(a)
    score_fn=registry.resolve(arts[0].digest); retention_fn=registry.resolve(arts[1].digest); security_fn=registry.resolve(arts[2].digest)
    model_cfg=cfg["model"]; spec=HFLoadSpec(model_cfg["id"],str(model_cfg.get("revision","main")),str(model_cfg.get("dtype","auto")),str(model_cfg.get("quantization","none")),bool(model_cfg.get("trust_remote_code",False)))
    seeds=tuple(int(x) for x in cfg.get("seeds",[0]))
    plan=ColabCampaignPlanV161(str(cfg["campaign_id"]),spec.model_id,spec.revision,parts.digest,arts[0].digest,arts[1].digest,arts[2].digest,seeds,float(cfg.get("minimum_forward_transfer",0.0)),float(cfg.get("minimum_retention",0.0)),True)
    campaign_dir=storage.root/"campaigns"/cfg["campaign_id"]; campaign_dir.mkdir(parents=True,exist_ok=True)
    (campaign_dir/"CAMPAIGN_PLAN.json").write_text(json.dumps({"value":asdict(plan),"digest":plan.digest},indent=2,sort_keys=True)); (campaign_dir/"DATASET_PROOF.json").write_text(json.dumps(parts.proof(),indent=2,sort_keys=True))
    signer=Ed25519Signer.generate("colab-execution-witness"); verifier=Ed25519Verifier(); verifier.register(signer.key_id,signer.public_bytes())
    (campaign_dir/"EXECUTION_PUBLIC_KEY.bin").write_bytes(signer.public_bytes()); (campaign_dir/"EXECUTION_KEY_ID.txt").write_text(signer.key_id+"\n")
    all_results=[]
    for seed in seeds:
        run_dir=campaign_dir/f"seed-{seed}"; run_dir.mkdir(parents=True,exist_ok=True)
        seed_result_path=run_dir/"SEED_RESULT.json"
        if (run_dir/"COMPLETE").is_file() and seed_result_path.is_file():
            all_results.append(json.loads(seed_result_path.read_text())); continue
        random.seed(seed)
        tokenizer=load_tokenizer(spec); model=load_causal_lm(spec)
        model_digest=model_identity(model,spec); tok_digest=tokenizer_identity(tokenizer,spec)
        a0_score,a0_outputs=evaluate(model,tokenizer,hidden,int(cfg.get("max_new_tokens",8)),score_fn)
        a0=ExecutedRunReceiptV161.sign(signer=signer,campaign_digest=plan.digest,arm="A0",seed=seed,environment_digest=env.digest,model_digest=model_digest,tokenizer_digest=tok_digest,adapter_digest=ZERO,dataset_digest=parts.hidden.digest,evaluator_digest=arts[0].digest,metrics={"hidden_exact_match":a0_score})
        assert a0.verify(verifier)
        (run_dir/"A0.json").write_text(json.dumps({"receipt":asdict(a0),"outputs":a0_outputs},indent=2,sort_keys=True))
        train_spec=LoraTrainSpec(**cfg.get("lora",{}),seed=seed)
        adapter_dir=storage.root/"adapters"/cfg["campaign_id"]/f"seed-{seed}"
        texts=[str(r.get("train_text") or (str(r["prompt"])+" "+str(r["expected"]))) for r in train]
        model,training_receipt=train_lora(model=model,tokenizer=tokenizer,texts=texts,output_dir=adapter_dir,spec=train_spec)
        adapter_digest=sha256_path(adapter_dir)
        del model; gc.collect()
        try:
            import torch
            if torch.cuda.is_available(): torch.cuda.empty_cache()
        except Exception: pass
        model=load_causal_lm(spec,adapter_path=str(adapter_dir)); a1_score,a1_outputs=evaluate(model,tokenizer,hidden,int(cfg.get("max_new_tokens",8)),score_fn)
        retention_rows=train[:max(1,min(len(train),int(cfg.get("retention_samples",2))))]
        retention,ret_outputs=evaluate(model,tokenizer,retention_rows,int(cfg.get("max_new_tokens",8)),retention_fn)
        security_count=sum(int(security_fn(x["prediction"],x["expected"])) for x in a1_outputs)
        a1=ExecutedRunReceiptV161.sign(signer=signer,campaign_digest=plan.digest,arm="A1",seed=seed,environment_digest=env.digest,model_digest=model_digest,tokenizer_digest=tok_digest,adapter_digest=adapter_digest,dataset_digest=parts.hidden.digest,evaluator_digest=arts[0].digest,metrics={"hidden_exact_match":a1_score,"retention":retention,"security_regressions":security_count})
        assert a1.verify(verifier)
        (run_dir/"A1.json").write_text(json.dumps({"receipt":asdict(a1),"outputs":a1_outputs,"training":training_receipt},indent=2,sort_keys=True))
        seed_result={"seed":seed,"a0":a0_score,"a1":a1_score,"delta":a1_score-a0_score,"retention":retention,"security_regressions":security_count,"a0_receipt":a0.digest,"a1_receipt":a1.digest,"adapter_digest":adapter_digest}
        seed_result_path.write_text(json.dumps(seed_result,indent=2,sort_keys=True)); (run_dir/"COMPLETE").write_text("complete\n"); all_results.append(seed_result)
        del model; gc.collect()
    mean_delta=sum(x["delta"] for x in all_results)/len(all_results); mean_ret=sum(x["retention"] for x in all_results)/len(all_results)
    decision="PASS" if mean_delta>=plan.minimum_forward_transfer and mean_ret>=plan.minimum_retention else "BLOCK"
    summary={"schema":"mini-agi-v16.1-colab-campaign-result-v1","campaign_plan_digest":plan.digest,"environment_digest":env.digest,"results":all_results,"mean_forward_transfer_delta":mean_delta,"mean_retention":mean_ret,"decision":decision,"promotion_ready":decision=="PASS","note":"PASS is experimental qualification only; it is not production promotion authority."}
    summary["digest"]=digest(summary); (campaign_dir/"RESULT.json").write_text(json.dumps(summary,indent=2,sort_keys=True)); print(json.dumps(summary,indent=2,sort_keys=True)); return 0
if __name__=="__main__": raise SystemExit(main())
