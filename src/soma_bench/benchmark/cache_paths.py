from __future__ import annotations

import os
from pathlib import Path


def benchmark_cache_root() -> Path:
    configured = os.getenv("SOMA_BENCHMARK_CACHE_DIR")
    if isinstance(configured, str) and configured.strip():
        return Path(configured).expanduser().resolve()

    xdg_cache_home = os.getenv("XDG_CACHE_HOME")
    if isinstance(xdg_cache_home, str) and xdg_cache_home.strip():
        cache_home = Path(xdg_cache_home).expanduser().resolve()
    else:
        cache_home = (Path.home() / ".cache").resolve()

    return cache_home / "soma-bench" / "benchmark-cache"