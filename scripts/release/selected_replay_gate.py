#!/usr/bin/env python3
"""Exercise an above-budget agent-safe selected-replay request."""
from __future__ import annotations

import json
import sys
import urllib.request

base = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:18080"


def get_json(path: str) -> dict:
    with urllib.request.urlopen(base + path, timeout=30) as response:
        return json.load(response)


def post_json(path: str, obj: dict, timeout: int = 60) -> dict:
    req = urllib.request.Request(
        base + path,
        data=json.dumps(obj).encode(),
        headers={"content-type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.load(response)


status = get_json("/v1/kvmem/status")
assert status["profile"] == "agent-safe", status
assert status["state_coherence"] == "selected-replay", status
budget = int(status["selection_budget_tokens"])
ctx = int(status["logical_context_tokens"])
gen = int(status["generation_budget_tokens"])

# Build enough historical *messages* to cross the semantic selection budget while
# keeping the final current query small. Use QW3's own tokenizer endpoint rather
# than assuming a words->tokens ratio. Chat framing will only add more tokens.
messages: list[dict[str, str]] = []
plain_count = 0
for i in range(2000):
    messages.append({
        "role": "user" if i % 2 == 0 else "assistant",
        "content": (
            f"Historical tool round {i}: alpha beta gamma delta epsilon zeta eta theta. "
            f"Evidence marker E{i:04d}. The provider observation remains non-authoritative."
        ),
    })
    if i % 100 == 99:
        joined = "\n".join(m["content"] for m in messages)
        plain_count = int(post_json("/v1/tokenize", {"content": joined})["count"])
        if plain_count > budget + max(1024, budget // 8):
            break
else:
    raise AssertionError("could not construct an above-budget release-gate prompt")

# Leave generous room for chat-template framing and generation.
if plain_count + gen + 4096 >= ctx:
    raise AssertionError(
        f"release gate prompt too close to ctx limit: plain={plain_count} gen={gen} ctx={ctx}"
    )

messages.append({"role": "user", "content": "Reply with exactly OK."})
out = post_json(
    "/v1/chat/completions",
    {"model": "qw3", "messages": messages, "max_tokens": 8},
    timeout=300,
)
receipt = out.get("kvmem") or {}
assert receipt.get("profile") == "agent-safe", receipt
assert receipt.get("state_coherence") == "selected-replay", receipt
logical = int(receipt.get("logical_prompt_tokens", 0))
if logical <= budget:
    raise AssertionError(
        f"selected-replay gate did not cross selection budget: logical={logical} budget={budget}"
    )
assert receipt.get("hybrid_state_exact_for_request") is True, receipt
assert receipt.get("retrieval_fallback_permitted") is False, receipt
assert receipt.get("query_replay") is True, receipt
print(
    f"selected-replay GPU gate: PASS logical={logical} budget={budget} "
    f"history_plain_tokens={plain_count}"
)
