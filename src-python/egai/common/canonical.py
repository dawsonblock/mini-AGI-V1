import hashlib
import json
import math
import re
from dataclasses import asdict, is_dataclass
from enum import Enum
from datetime import datetime, timezone

DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
CANONICAL_VERSION = "egai-json-v1"

def validate_digest(value:str)->str:
    if not isinstance(value,str) or not DIGEST_RE.fullmatch(value):
        raise ValueError("invalid sha256 digest")
    return value

def _norm(x):
    if is_dataclass(x): x=asdict(x)
    if isinstance(x,Enum): return x.value
    if isinstance(x,float):
        if not math.isfinite(x): raise ValueError("non-finite numbers forbidden in canonical objects")
        return x
    if isinstance(x,datetime):
        if x.tzinfo is None: raise ValueError("naive datetime forbidden")
        return x.astimezone(timezone.utc).isoformat().replace('+00:00','Z')
    if isinstance(x,dict): return {str(k):_norm(v) for k,v in sorted(x.items(),key=lambda kv:str(kv[0]))}
    if isinstance(x,(list,tuple)): return [_norm(v) for v in x]
    if x is None or isinstance(x,(str,int,bool)): return x
    raise TypeError(f"unsupported canonical type: {type(x).__name__}")

def canonical_bytes(x)->bytes:
    body={"canonical_version":CANONICAL_VERSION,"value":_norm(x)}
    return json.dumps(body,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode()

def digest(x)->str:
    return "sha256:"+hashlib.sha256(canonical_bytes(x)).hexdigest()

def sha256_bytes(b:bytes)->str:
    return "sha256:"+hashlib.sha256(b).hexdigest()
