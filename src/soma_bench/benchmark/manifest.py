from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


class BenchmarkManifestError(ValueError):
    """Raised when a benchmark manifest row is invalid."""


def _coerce_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _coerce_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    raise BenchmarkManifestError(f"Expected a mapping, got {type(value).__name__}")


@dataclass(slots=True)
class BenchmarkInstance:
    benchmark_id: str
    trajectory_id: str
    instance_id: str
    repo: str
    base_commit: str = ""
    environment_setup_commit: str = ""
    install_config: dict[str, Any] = field(default_factory=dict)
    requirements: list[Any] = field(default_factory=list)
    environment: dict[str, Any] = field(default_factory=dict)
    fail_to_pass: list[Any] = field(default_factory=list)
    pass_to_pass: list[Any] = field(default_factory=list)
    user_prompt: str = ""
    stage_boundary: str = ""
    expected_next_stage: str = ""
    expected_next_action: str = ""
    resume_prompt: str = ""
    hidden_eval: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "BenchmarkInstance":
        required_fields = ("trajectory_id", "instance_id", "repo")
        missing = [name for name in required_fields if not payload.get(name)]
        if missing:
            raise BenchmarkManifestError(
                "Manifest row is missing required fields: " + ", ".join(sorted(missing))
            )

        return cls(
            benchmark_id=_derive_benchmark_id(payload),
            trajectory_id=str(payload["trajectory_id"]),
            instance_id=str(payload["instance_id"]),
            repo=str(payload["repo"]),
            base_commit=str(payload.get("base_commit", "")),
            environment_setup_commit=str(payload.get("environment_setup_commit", "")),
            install_config=_coerce_dict(payload.get("install_config")),
            requirements=_coerce_list(payload.get("requirements")),
            environment=_coerce_dict(payload.get("environment")),
            fail_to_pass=_coerce_list(payload.get("fail_to_pass")),
            pass_to_pass=_coerce_list(payload.get("pass_to_pass")),
            user_prompt=str(payload.get("user_prompt", "")),
            stage_boundary=str(payload.get("stage_boundary", "")),
            expected_next_stage=str(payload.get("expected_next_stage", "")),
            expected_next_action=str(payload.get("expected_next_action", "")),
            resume_prompt=str(payload.get("resume_prompt", "")),
            hidden_eval=_coerce_dict(payload.get("hidden_eval")),
            metadata=_coerce_dict(payload.get("metadata")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "benchmark_id": self.benchmark_id,
            "trajectory_id": self.trajectory_id,
            "instance_id": self.instance_id,
            "repo": self.repo,
            "base_commit": self.base_commit,
            "environment_setup_commit": self.environment_setup_commit,
            "install_config": self.install_config,
            "requirements": self.requirements,
            "environment": self.environment,
            "fail_to_pass": self.fail_to_pass,
            "pass_to_pass": self.pass_to_pass,
            "user_prompt": self.user_prompt,
            "stage_boundary": self.stage_boundary,
            "expected_next_stage": self.expected_next_stage,
            "expected_next_action": self.expected_next_action,
            "resume_prompt": self.resume_prompt,
            "hidden_eval": self.hidden_eval,
            "metadata": self.metadata,
        }


def load_selected_instance_ids(path: str | Path) -> set[str]:
    selected_ids: set[str] = set()
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            value = line.strip()
            if value:
                selected_ids.add(value)
    return selected_ids


def _derive_benchmark_id(payload: dict[str, Any]) -> str:
    explicit = payload.get("benchmark_id")
    if isinstance(explicit, str) and explicit.strip():
        return explicit.strip()
    instance_id = str(payload.get("instance_id", "")).strip() or "unknown-instance"
    source_stage = str(payload.get("stage_boundary", "")).strip() or "unknown-source"
    resume_stage = str(payload.get("expected_next_stage", "")).strip() or "unknown-resume"
    return f"{instance_id}:{source_stage}->{resume_stage}"


def load_manifest(
    path: str | Path,
    limit: int | None = None,
    selected_instance_ids: set[str] | None = None,
) -> list[BenchmarkInstance]:
    entries: list[BenchmarkInstance] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if limit is not None and limit > 0 and len(entries) >= limit:
                break
            raw_line = line.strip()
            if not raw_line:
                continue
            try:
                payload = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise BenchmarkManifestError(
                    f"Invalid JSON on manifest line {line_number}: {exc.msg}"
                ) from exc
            instance = BenchmarkInstance.from_dict(payload)
            if selected_instance_ids and instance.instance_id not in selected_instance_ids:
                continue
            entries.append(instance)
    return entries


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")