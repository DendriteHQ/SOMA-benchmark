"""Runtime backend that solves benchmark instances with the GitHub Copilot CLI.

Only the Copilot-specific part lives here - its compose service, CLI arguments and event
stream. The run itself, and the sandbox stack around the agent, is harness_base's.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
from pathlib import Path

from ..base import RuntimeBackend, RuntimeExecutionContext, RuntimeExecutionResult
from ..harness_base import (
    AgentHarness,
    AgentLaunch,
    _resolve_benchmark_trajectory_path,
    _resolve_runtime_option,
    run_harness_batch,
    run_harness_instance,
)

COPILOT_DEFAULT_COMPOSE_FILE = str(
    (Path(__file__).resolve().parent / "copilot-cli-container" / "docker-compose.yml").resolve()
)
COPILOT_DEFAULT_SERVICE = "copilot"
COPILOT_DEFAULT_IMAGE = "local/copilot-cli:latest"
COPILOT_DEFAULT_OUTPUT_FORMAT = "json"
COPILOT_PROJECT_NAME_PREFIX = "soma-copilot-"


def _resolve_compose_file(context: RuntimeExecutionContext) -> Path:
    for value in (
        _resolve_runtime_option(context, "copilot_compose_file"),
        os.getenv("SOMA_COPILOT_COMPOSE_FILE"),
        COPILOT_DEFAULT_COMPOSE_FILE,
    ):
        if isinstance(value, str) and value.strip():
            path = Path(value.strip()).expanduser().resolve()
            if path.is_file():
                return path
    raise RuntimeError(
        "Copilot docker-compose file was not found. Set SOMA_COPILOT_COMPOSE_FILE "
        "or runtime_options.copilot_compose_file."
    )


def _resolve_compose_service(context: RuntimeExecutionContext) -> str:
    for value in (
        _resolve_runtime_option(context, "copilot_service"),
        os.getenv("SOMA_COPILOT_SERVICE"),
        COPILOT_DEFAULT_SERVICE,
    ):
        if isinstance(value, str) and value.strip():
            return value.strip()
    return COPILOT_DEFAULT_SERVICE


def _resolve_extra_args(context: RuntimeExecutionContext) -> list[str]:
    for value in (
        _resolve_runtime_option(context, "copilot_args"),
        os.getenv("SOMA_COPILOT_ARGS"),
    ):
        if isinstance(value, str) and value.strip():
            return shlex.split(value)
        if isinstance(value, list) and all(isinstance(item, str) for item in value):
            return list(value)
    return ["--allow-all", "--no-ask-user"]


def _resolve_output_format(context: RuntimeExecutionContext) -> str:
    for value in (
        _resolve_runtime_option(context, "copilot_output_format"),
        os.getenv("SOMA_COPILOT_OUTPUT_FORMAT"),
        COPILOT_DEFAULT_OUTPUT_FORMAT,
    ):
        if isinstance(value, str) and value.strip():
            normalized = value.strip().lower()
            if normalized in {"text", "json"}:
                return normalized
    return COPILOT_DEFAULT_OUTPUT_FORMAT


def _contains_output_format_arg(args: list[str]) -> bool:
    for idx, item in enumerate(args):
        if item == "--output-format":
            return True
        if item.startswith("--output-format="):
            return True
        if item == "-o" and idx + 1 < len(args):
            return True
    return False


def _contains_stream_arg(args: list[str]) -> bool:
    for idx, item in enumerate(args):
        if item == "--stream" and idx + 1 < len(args):
            return True
        if item.startswith("--stream="):
            return True
    return False


def _count_trajectory_steps(trajectory_path: Path) -> int | None:
    """Count agent steps from a copilot trajectory JSONL file.

    Each agent runs its own 0-indexed ``turnId`` sequence: the main agent's
    turns carry no ``agentId``, while every sub-agent spawned via the ``task``
    tool emits its own turns tagged with the spawning tool call's ``agentId``.
    A single global "last turnId" therefore misses all sub-agent steps, so we
    group turns by ``agentId`` and sum ``max(turnId) + 1`` across every agent.

    Falls back to counting assistant.turn_start events if no turn carries a
    usable ``turnId``.
    """
    try:
        if not trajectory_path.is_file():
            return None
        max_turn_id: dict[object, int] = {}
        turn_start_count = 0
        with trajectory_path.open(encoding="utf-8") as fh:
            for raw_line in fh:
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                entry_type = entry.get("type")
                if entry_type == "assistant.turn_start":
                    turn_start_count += 1
                elif entry_type != "assistant.turn_end":
                    continue
                # Both turn_start and turn_end carry the turnId; using either
                # keeps the count correct even if one side is missing.
                turn_id_raw = (entry.get("data") or {}).get("turnId")
                if turn_id_raw is None:
                    continue
                try:
                    turn_id = int(turn_id_raw)
                except (TypeError, ValueError):
                    continue
                agent_id = entry.get("agentId")
                if agent_id not in max_turn_id or turn_id > max_turn_id[agent_id]:
                    max_turn_id[agent_id] = turn_id
        if max_turn_id:
            return sum(turn_id + 1 for turn_id in max_turn_id.values())
        if turn_start_count > 0:
            return turn_start_count
        return None
    except Exception:
        return None


def _extract_stream_error(stdout: str) -> str | None:
    """Scan Copilot's JSONL event stream for the actual failure reason.

    docker compose's own stdout/stderr only ever contains provisioning noise
    ("Volume X Created", "Container Y Created"); the real reason a run failed
    (e.g. a 404 from the model provider) is emitted by the Copilot CLI as a
    `session.error` / `model.call_failure` event on stdout.
    """
    session_error: str | None = None
    call_failure: str | None = None
    for line in stdout.split("\n"):
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue
        data = event.get("data") if isinstance(event.get("data"), dict) else {}
        if event.get("type") == "session.error":
            message = data.get("message")
            if isinstance(message, str) and message.strip():
                session_error = message.strip()
        elif event.get("type") == "model.call_failure":
            error_message = data.get("errorMessage")
            if isinstance(error_message, str) and error_message.strip():
                call_failure = error_message.strip()
    return session_error or call_failure


def _error_summary(stderr: str, stdout: str) -> str:
    stream_error = _extract_stream_error(stdout)
    if stream_error:
        return " ".join(stream_error.split())[:400]
    for candidate in (stderr, stdout):
        normalized = " ".join(candidate.split())
        if normalized:
            return normalized[:400]
    return "Copilot execution failed"


class CopilotHarness(AgentHarness):
    name = "copilot"
    display_name = "Copilot"
    default_image = COPILOT_DEFAULT_IMAGE
    compose_project_prefix = COPILOT_PROJECT_NAME_PREFIX

    def resolve_compose_file(self, context: RuntimeExecutionContext) -> Path:
        return _resolve_compose_file(context)

    def resolve_compose_service(self, context: RuntimeExecutionContext) -> str:
        return _resolve_compose_service(context)

    def prepare_launch(
        self,
        context: RuntimeExecutionContext,
        env: dict[str, str],
        *,
        prompt: str,
        proxy_url: str | None,
        credential: str | None,
        repo_path: str,
        tmp_run_dir: Path,
    ) -> AgentLaunch:
        extra_args = _resolve_extra_args(context)
        output_format = _resolve_output_format(context)
        if not _contains_output_format_arg(extra_args):
            extra_args = [*extra_args, "--output-format", output_format]
        if not _contains_stream_arg(extra_args):
            extra_args = [*extra_args, "--stream", "off"]

        if proxy_url is not None:
            env["COPILOT_PROVIDER_BASE_URL"] = f"{proxy_url}/"
        elif isinstance(context.llm_config.get("base_url"), str):
            env["COPILOT_PROVIDER_BASE_URL"] = str(context.llm_config["base_url"])
        if isinstance(context.llm_config.get("model"), str):
            env["COPILOT_MODEL"] = str(context.llm_config["model"])
        if credential is not None:
            env["COPILOT_PROVIDER_API_KEY"] = credential

        return AgentLaunch(
            run_args=[],
            env_names=["COPILOT_PROVIDER_BASE_URL", "COPILOT_MODEL", "COPILOT_PROVIDER_API_KEY"],
            service_args=["-p", prompt, *extra_args],
            # Text output has no event stream to keep.
            trajectory_path=(
                _resolve_benchmark_trajectory_path(context, harness_name=self.name) if output_format == "json" else None
            ),
            metadata={
                "command": "docker compose --project-name <name> run copilot -p ...",
                "args": extra_args,
                "copilot_output_format": output_format,
            },
        )

    def count_steps(self, trajectory_path: Path) -> int | None:
        return _count_trajectory_steps(trajectory_path)

    def error_summary(
        self,
        *,
        result: subprocess.CompletedProcess[str],
        trajectory_path: Path | None,
    ) -> str:
        return _error_summary(result.stderr or "", result.stdout or "")


COPILOT_HARNESS = CopilotHarness()


def run_copilot_instance(context: RuntimeExecutionContext) -> RuntimeExecutionResult:
    return run_harness_instance(context, COPILOT_HARNESS)


def run_copilot_batch(
    contexts: list[RuntimeExecutionContext],
    concurrency: int,
) -> list[RuntimeExecutionResult]:
    return run_harness_batch(contexts, concurrency, COPILOT_HARNESS)


COPILOT_BACKEND = RuntimeBackend(
    name="copilot",
    default_image=COPILOT_DEFAULT_IMAGE,
    default_build_target="container",
    execution_hint=(
        "Use --execute to run benchmark instances through GitHub Copilot CLI in Docker Compose. "
        "By default this backend runs docker compose -f "
        "src/soma_bench/benchmark/backends/copilot/copilot-cli-container/docker-compose.yml "
        "run --rm copilot -p <prompt> --allow-all --no-ask-user."
    ),
    execute=run_copilot_instance,
    execute_batch=run_copilot_batch,
)
