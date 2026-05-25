from __future__ import annotations

import sys
import time


def emit_progress(message: str, *, component: str = "benchmark") -> None:
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [{component}] {message}", file=sys.stderr, flush=True)