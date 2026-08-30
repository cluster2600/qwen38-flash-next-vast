#!/usr/bin/env python3
"""Check the loopback OpenAI-compatible endpoint without exposing secrets."""

import json
import os
import sys
from urllib.error import URLError
from urllib.request import urlopen


port = int(os.environ.get("PORT", "8000"))
url = f"http://127.0.0.1:{port}/v1/models"

try:
    with urlopen(url, timeout=5) as response:
        payload = json.load(response)
except (OSError, URLError, ValueError) as exc:
    print(f"not ready: {exc}", file=sys.stderr)
    raise SystemExit(1) from exc

models = [item.get("id") for item in payload.get("data", [])]
if not models:
    print("not ready: no served model", file=sys.stderr)
    raise SystemExit(1)

print(models[0])
