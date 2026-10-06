from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
import json, random, time
from egai.common.canonical import digest

@dataclass(frozen=True)
class LoraTrainSpec:
    rank: int = 8
    alpha: int = 16
    dropout: float = 0.05
    target_modules: tuple[str,...] = ("c_attn",)
    learning_rate: float = 2e-4
    steps: int = 20
    max_length: int = 128
    seed: int = 0
    schema: str = "mini-agi-v16.1-lora-train-spec-v1"
    @property
    def digest(self): return digest(self)


def train_lora(*, model, tokenizer, texts: list[str], output_dir: str | Path, spec: LoraTrainSpec):
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    if not texts: raise ValueError("training texts required")
    random.seed(spec.seed); torch.manual_seed(spec.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(spec.seed)
    try: model=prepare_model_for_kbit_training(model)
    except Exception: pass
    cfg=LoraConfig(r=spec.rank,lora_alpha=spec.alpha,lora_dropout=spec.dropout,target_modules=list(spec.target_modules),task_type="CAUSAL_LM")
    model=get_peft_model(model,cfg); model.train()
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=spec.learning_rate)
    device=next(model.parameters()).device
    losses=[]; started=time.time_ns()
    for step in range(spec.steps):
        text=texts[step%len(texts)]
        batch=tokenizer(text,return_tensors="pt",truncation=True,max_length=spec.max_length)
        batch={k:v.to(device) for k,v in batch.items()}; labels=batch["input_ids"].clone()
        out=model(**batch,labels=labels); loss=out.loss; loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
        losses.append(float(loss.detach().cpu()))
    outdir=Path(output_dir); outdir.mkdir(parents=True,exist_ok=True); model.save_pretrained(outdir,safe_serialization=True)
    receipt={"schema":"mini-agi-v16.1-training-receipt-v1","train_spec_digest":spec.digest,"steps":spec.steps,"loss_first":losses[0],"loss_last":losses[-1],"started_ns":started,"finished_ns":time.time_ns()}
    (outdir/"TRAINING_RECEIPT.json").write_text(json.dumps(receipt,indent=2,sort_keys=True))
    (outdir/"TRAINING_SPEC.json").write_text(json.dumps(asdict(spec),indent=2,sort_keys=True))
    return model, receipt
