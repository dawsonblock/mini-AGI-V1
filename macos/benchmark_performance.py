#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import time
import urllib.request
from pathlib import Path


def _request_stream(url: str, body: dict, timeout: float) -> dict:
    payload = json.dumps(body).encode()
    req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json", "Accept": "text/event-stream"})
    start = time.perf_counter()
    first = None
    chars = 0
    usage_tokens = None
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                obj = json.loads(data)
            except json.JSONDecodeError:
                continue
            try:
                delta = obj["choices"][0]["delta"].get("content") or ""
            except Exception:
                delta = ""
            if delta:
                if first is None:
                    first = time.perf_counter()
                chars += len(delta)
            usage = obj.get("usage") or {}
            if usage.get("completion_tokens") is not None:
                usage_tokens = int(usage["completion_tokens"])
    end = time.perf_counter()
    ttft = (first - start) if first is not None else end - start
    decode_s = max(1e-9, end - (first or start))
    tokens = usage_tokens if usage_tokens is not None else max(1, round(chars / 4))
    return {
        "ttft_s": ttft,
        "total_s": end - start,
        "completion_tokens": tokens,
        "completion_tokens_source": "usage" if usage_tokens is not None else "approx_chars_div4",
        "decode_tokens_per_s": tokens / decode_s,
        "output_chars": chars,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Measure streaming TTFT and decode throughput on the local Apple runtime.")
    ap.add_argument("--endpoint", default="http://127.0.0.1:8088/v1/chat/completions")
    ap.add_argument("--model", default="local-qwen")
    ap.add_argument("--prompt", default="Explain in three concise paragraphs why caching reduces LLM inference latency.")
    ap.add_argument("--prompt-file")
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--timeout", type=float, default=300)
    ap.add_argument("--json-out")
    args = ap.parse_args()
    prompt = Path(args.prompt_file).read_text() if args.prompt_file else args.prompt
    rows = []
    for i in range(args.runs):
        body = {
            "model": args.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": args.max_tokens,
            "temperature": 0,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        row = _request_stream(args.endpoint, body, args.timeout)
        row["run"] = i + 1
        rows.append(row)
        print(f"run={i+1} ttft_s={row['ttft_s']:.3f} total_s={row['total_s']:.3f} decode_tps={row['decode_tokens_per_s']:.2f}")
    summary = {
        "runs": rows,
        "summary": {
            "ttft_median_s": statistics.median(r["ttft_s"] for r in rows),
            "ttft_mean_s": statistics.fmean(r["ttft_s"] for r in rows),
            "decode_tps_median": statistics.median(r["decode_tokens_per_s"] for r in rows),
            "decode_tps_mean": statistics.fmean(r["decode_tokens_per_s"] for r in rows),
            "total_median_s": statistics.median(r["total_s"] for r in rows),
        },
        "method": "OpenAI-compatible SSE; completion tokens use backend usage when available, otherwise chars/4 estimate",
    }
    print(json.dumps(summary["summary"], indent=2, sort_keys=True))
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
