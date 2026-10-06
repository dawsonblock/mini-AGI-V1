from __future__ import annotations
import torch

@torch.inference_mode()
def generate_text(model, tokenizer, prompt: str, *, max_new_tokens: int = 16) -> str:
    device=next(model.parameters()).device
    inputs=tokenizer(prompt,return_tensors="pt"); inputs={k:v.to(device) for k,v in inputs.items()}
    out=model.generate(**inputs,max_new_tokens=max_new_tokens,do_sample=False,pad_token_id=tokenizer.eos_token_id)
    new=out[0,inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new,skip_special_tokens=True).strip()

def exact_match(prediction: str, expected: str) -> float:
    return float(prediction.strip().casefold()==expected.strip().casefold())
