from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .backends import DEFAULT_RUNTIME_BACKEND

DEFAULT_WORKSPACE = "docker"
DEFAULT_MAX_ITERATIONS = 100
DEFAULT_CONCURRENCY = 1
DEFAULT_OUTPUT_DIR = "outputs/soma-bench-local"
SUPPORTED_WORKSPACES = ("docker",)


@dataclass(slots=True)
class BenchmarkDefaults:
    runtime_backend: str = DEFAULT_RUNTIME_BACKEND
    workspace: str = DEFAULT_WORKSPACE
    max_iterations: int = DEFAULT_MAX_ITERATIONS
    concurrency: int = DEFAULT_CONCURRENCY
    output_dir: str = DEFAULT_OUTPUT_DIR

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


DEFAULTS = BenchmarkDefaults()


def normalize_instance_tag(instance_id: str) -> str:
    return instance_id.replace("__", "_1776_").replace("/", "_").lower()


def build_agent_image_ref(
    instance_id: str,
    *,
    image: str,
    target: str,
) -> str:
    return f"{image}:{normalize_instance_tag(instance_id)}-{target}"


def default_output_path(output_dir: str | Path) -> Path:
    return Path(output_dir) / "output.jsonl"
