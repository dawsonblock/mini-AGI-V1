from __future__ import annotations
from dataclasses import dataclass
from typing import Any

@dataclass(frozen=True)
class HFLoadSpec:
    model_id: str
    revision: str
    dtype: str = "auto"
    quantization: str = "none"
    trust_remote_code: bool = False


def load_tokenizer(spec: HFLoadSpec):
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(spec.model_id,revision=spec.revision,trust_remote_code=spec.trust_remote_code)
    if tok.pad_token_id is None and tok.eos_token_id is not None: tok.pad_token=tok.eos_token
    return tok


def load_causal_lm(spec: HFLoadSpec, *, adapter_path: str | None = None):
    import torch
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    kwargs={"revision":spec.revision,"trust_remote_code":spec.trust_remote_code}
    if spec.dtype=="bfloat16": kwargs["torch_dtype"]=torch.bfloat16
    elif spec.dtype=="float16": kwargs["torch_dtype"]=torch.float16
    elif spec.dtype=="float32": kwargs["torch_dtype"]=torch.float32
    else: kwargs["torch_dtype"]="auto"
    if spec.quantization in {"4bit","8bit"}:
        kwargs["quantization_config"]=BitsAndBytesConfig(load_in_4bit=spec.quantization=="4bit",load_in_8bit=spec.quantization=="8bit")
        kwargs["device_map"]="auto"
    elif torch.cuda.is_available(): kwargs["device_map"]="auto"
    model=AutoModelForCausalLM.from_pretrained(spec.model_id,**kwargs)
    if adapter_path:
        from peft import PeftModel
        model=PeftModel.from_pretrained(model,adapter_path)
    return model
