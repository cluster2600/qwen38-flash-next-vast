#!/usr/bin/env python3
"""Run a local single-stream decode gate against the vLLM endpoint."""

import json
import os
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


PORT = int(os.environ.get("PORT", "8000"))
MODEL = os.environ.get(
    "SERVED_MODEL_NAME", "qwen3.8-flash-next-uncensored-nvfp4"
)
MIN_TOKENS_PER_SECOND = float(os.environ.get("MIN_TOKENS_PER_SECOND", "60"))
BENCHMARK_TOKENS = int(os.environ.get("BENCHMARK_TOKENS", "512"))
URL = f"http://127.0.0.1:{PORT}/v1/completions"


def completion(tokens: int) -> tuple[int, float]:
    payload = {
        "model": MODEL,
        "prompt": "Continue with a compact numbered technical checklist:\n1.",
        "max_tokens": tokens,
        "min_tokens": tokens,
        "ignore_eos": True,
        "temperature": 0.0,
    }
    request = Request(
        URL,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    with urlopen(request, timeout=900) as response:
        result = json.load(response)
    elapsed = time.perf_counter() - started
    generated = int(result.get("usage", {}).get("completion_tokens", 0))
    if generated <= 0:
        raise RuntimeError("response did not include completion token usage")
    return generated, elapsed


def main() -> None:
    if BENCHMARK_TOKENS < 128:
        raise SystemExit("BENCHMARK_TOKENS must be at least 128")

    print("warming up the decode path...")
    completion(64)

    generated, elapsed = completion(BENCHMARK_TOKENS)
    rate = generated / elapsed
    print(
        f"single-stream: {generated} tokens in {elapsed:.2f}s = {rate:.1f} tokens/s"
    )
    if rate < MIN_TOKENS_PER_SECOND:
        print(
            f"FAIL: {rate:.1f} tokens/s is below the {MIN_TOKENS_PER_SECOND:.1f} target",
            file=sys.stderr,
        )
        raise SystemExit(1)
    print(f"PASS: target >= {MIN_TOKENS_PER_SECOND:.1f} tokens/s")


if __name__ == "__main__":
    try:
        main()
    except (HTTPError, URLError, OSError, ValueError, RuntimeError) as exc:
        print(f"benchmark failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
