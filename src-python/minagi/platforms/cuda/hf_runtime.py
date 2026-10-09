"""CUDA/HF loading helpers.

v16.4.1: `load_causal_lm` refuses to open an adapter directory for
SERVING. Serving must go through `TrustedRuntimeLauncher` and load from
an `MeasuredSnapshot` (`load_causal_lm(spec, snapshot=snap)`), which
binds the exact authorized bytes. The research plane — evaluating a
candidate artifact, which is the experiment itself, not the runtime —
must opt in explicitly with `purpose="research"`; that flag is the
reviewable record that the load is not a serving activation.
"""
from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class HFLoadSpec:
    model_id: str
    revision: str
    dtype: str = "auto"
    quantization: str = "none"
    trust_remote_code: bool = False


def load_tokenizer(spec: HFLoadSpec, *, path: str | None = None):
    from transformers import AutoTokenizer
    source = path or spec.model_id
    if path:
        tok = AutoTokenizer.from_pretrained(source, local_files_only=True)
    else:
        tok = AutoTokenizer.from_pretrained(
            source, revision=spec.revision,
            trust_remote_code=spec.trust_remote_code)
    if tok.pad_token_id is None and tok.eos_token_id is not None: tok.pad_token=tok.eos_token
    return tok


def _model_kwargs(spec: HFLoadSpec) -> dict:
    import torch
    from transformers import BitsAndBytesConfig
    kwargs={"trust_remote_code":spec.trust_remote_code}
    if spec.dtype=="bfloat16": kwargs["torch_dtype"]=torch.bfloat16
    elif spec.dtype=="float16": kwargs["torch_dtype"]=torch.float16
    elif spec.dtype=="float32": kwargs["torch_dtype"]=torch.float32
    else: kwargs["torch_dtype"]="auto"
    if spec.quantization in {"4bit","8bit"}:
        kwargs["quantization_config"]=BitsAndBytesConfig(load_in_4bit=spec.quantization=="4bit",load_in_8bit=spec.quantization=="8bit")
        kwargs["device_map"]="auto"
    elif torch.cuda.is_available(): kwargs["device_map"]="auto"
    return kwargs


def load_causal_lm(spec: HFLoadSpec, *, adapter_path: str | None = None,
                   purpose: str = "serving", snapshot=None):
    """Load a causal LM.

    Serving (default): pass `snapshot=<MeasuredSnapshot>` to load the
    model/tokenizer/adapter from the launcher-staged immutable copy.
    A serving load WITHOUT a snapshot is refused (a bare model load is
    not an admitted artifact), as is `adapter_path` in serving mode — a
    raw path is not an authorized artifact.

    Research-plane evaluation that must open an adapter directory
    directly (the candidate under evaluation) passes
    `purpose="research"` explicitly.
    """
    if purpose not in ("serving", "research"):
        raise ValueError("purpose must be 'serving' or 'research'")
    if purpose == "serving":
        if adapter_path is not None:
            raise PermissionError(
                "serving must load adapters from an MeasuredSnapshot staged "
                "by TrustedRuntimeLauncher — a raw adapter path is not an "
                "authorized artifact (research-plane evaluation must pass "
                "purpose='research' explicitly)")
        if snapshot is None:
            raise PermissionError(
                "serving requires snapshot=<MeasuredSnapshot> staged by "
                "TrustedRuntimeLauncher — a bare model load is not an "
                "admitted artifact (research-plane evaluation must pass "
                "purpose='research' explicitly)")
    from transformers import AutoModelForCausalLM
    if purpose == "serving":
        if snapshot is not None:
            model = AutoModelForCausalLM.from_pretrained(
                str(snapshot.path("model")), local_files_only=True,
                **_model_kwargs(spec))
            from peft import PeftModel
            return PeftModel.from_pretrained(
                model, str(snapshot.path("adapter")))
    elif snapshot is not None:
        raise ValueError("purpose='research' does not accept a snapshot")

    kwargs=_model_kwargs(spec)
    kwargs["revision"]=spec.revision
    model=AutoModelForCausalLM.from_pretrained(spec.model_id,**kwargs)
    if adapter_path:
        from peft import PeftModel
        model=PeftModel.from_pretrained(model,adapter_path)
    return model
