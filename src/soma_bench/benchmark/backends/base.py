from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from ..manifest import BenchmarkInstance


@dataclass(frozen=True, slots=True)
class RuntimeExecutionContext:
    backend_name: str
    instance: "BenchmarkInstance"
    run_payload: dict[str, Any]
    llm_config: dict[str, Any]
    workspace: str
    max_iterations: int
    output_dir: Path
    instance_dir: Path


@dataclass(frozen=True, slots=True)
class RuntimeExecutionResult:
    benchmark_id: str
    instance_id: str
    backend_name: str
    status: str
    error: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "benchmark_id": self.benchmark_id,
            "instance_id": self.instance_id,
            "runtime_backend": self.backend_name,
            "status": self.status,
            "metadata": self.metadata,
        }
        if self.error:
            payload["error"] = self.error
        return payload


RuntimeExecutor = Callable[[RuntimeExecutionContext], RuntimeExecutionResult]
RuntimeBatchExecutor = Callable[
    [list[RuntimeExecutionContext], int],
    list[RuntimeExecutionResult],
]


@dataclass(frozen=True, slots=True)
class RuntimeBackend:
    name: str
    default_image: str
    default_build_target: str
    execution_hint: str
    execute: RuntimeExecutor
    execute_batch: RuntimeBatchExecutor | None = None
