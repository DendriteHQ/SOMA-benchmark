"""Runtime backend that solves benchmark instances with Claude Code.

The agent is driven through the Claude Agent SDK for Python
(https://github.com/anthropics/claude-agent-sdk-python), which is Claude Code packaged as
a library: it spawns the `claude` CLI as a subprocess and speaks its stream-json protocol.
Because it is a library and not a command, the container's entrypoint is our own
`claude-code-container/agent_runner.py` rather than a vendor binary; that script is the
only piece of this backend that runs inside the sandbox.

A run is the copilot backend's run with a different agent in the agent container, so the
two are comparable measurements of the same compression setup rather than two different
experiments:

    claude-code ──► proxy ──► compression-service ──► upstream provider
         │
         └── swe-exec ──► swe-exec-broker ──► dind ──► swe-sandbox (task image)

Everything in that diagram except the first box - and the checkout, workspace seeding,
patch capture and evaluation around it - is harness_base's `run_harness_instance`, configured
by the same `SOMA_COPILOT_*` / `copilot_*` options. This module is only the
`AgentHarness`: how the agent is wired to its provider, started, and read back.

Two things about Claude Code specifically that the stack has to accommodate, both
documented at code.claude.com/docs/en/llm-gateway-protocol:

* It speaks the **Anthropic Messages API** and posts to `/v1/messages`, appending that
  path to `ANTHROPIC_BASE_URL` itself. The proxy is a path-preserving passthrough, so the
  configured upstream must be the provider's root, not its `/v1` prefix -
  `_resolve_proxy_upstream_base_url` is what reconciles that with an `llm.base_url` written
  for the OpenAI-shaped backends.
* It **streams** every inference request and watches the byte stream: a gateway that
  buffers the whole response before relaying it stalls the agent, and a stream that goes
  silent for 300s is aborted client-side. `compression_service/app/proxy.py` therefore
  relays `text/event-stream` responses incrementally.
"""

from __future__ import annotations

import json
import os
import subprocess
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from ..base import RuntimeBackend, RuntimeExecutionContext, RuntimeExecutionResult
from ...progress import emit_progress
from ..harness_base import (
    COPILOT_PROXY_UPSTREAM_ENV,
    AgentHarness,
    AgentLaunch,
    _resolve_benchmark_trajectory_path,
    _resolve_runtime_option,
    run_harness_batch,
    run_harness_instance,
)

CLAUDE_CODE_CONTAINER_DIR = Path(__file__).resolve().parent / "claude-code-container"
CLAUDE_CODE_DEFAULT_COMPOSE_FILE = str((CLAUDE_CODE_CONTAINER_DIR / "docker-compose.yml").resolve())
CLAUDE_CODE_DEFAULT_SERVICE = "claude-code"
CLAUDE_CODE_DEFAULT_IMAGE = "local/claude-code-agent:latest"
# Listed in harness_base's STACK_PROJECT_NAME_PREFIXES, so its stale-stack sweep covers these.
CLAUDE_CODE_PROJECT_NAME_PREFIX = "soma-claude-code-"
CLAUDE_CODE_PROMPT_CONTAINER_PATH = "/tmp/soma-prompt.txt"
CLAUDE_CODE_DEFAULT_PERMISSION_MODE = "bypassPermissions"
# Stands in for a credential when the proxy holds the real one; see prepare_launch.
CLAUDE_CODE_PROXY_PLACEHOLDER_TOKEN = "soma-benchmark-proxy"

# `claude` re-appends this to ANTHROPIC_BASE_URL on every request, so an upstream that
# already carries it would be reached at /v1/v1/messages through the path-preserving proxy.
_ANTHROPIC_API_PATH_PREFIX = "v1"

# Everything the agent container is given. Cleared from the inherited environment before
# being set, so a variable the host happens to export (an ANTHROPIC_BASE_URL from someone's
# own Claude Code setup) never reaches the agent in place of what this run decided.
_AGENT_ENV_NAMES = (
    "ANTHROPIC_BASE_URL",
    "ANTHROPIC_AUTH_TOKEN",
    "ANTHROPIC_MODEL",
    "CLAUDE_CODE_WORKDIR",
    "CLAUDE_CODE_PERMISSION_MODE",
    "CLAUDE_CODE_NETWORK_ISOLATED",
    "CLAUDE_CODE_MAX_TURNS",
    "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC",
    "CLAUDE_CODE_ATTRIBUTION_HEADER",
    "CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS",
    "CLAUDE_CODE_EXTRA_DISALLOWED_TOOLS",
)


# --------------------------------------------------------------------------------------
# Runtime options
# --------------------------------------------------------------------------------------


def _resolve_compose_file(context: RuntimeExecutionContext) -> Path:
    for value in (
        _resolve_runtime_option(context, "claude_code_compose_file"),
        os.getenv("SOMA_CLAUDE_CODE_COMPOSE_FILE"),
        CLAUDE_CODE_DEFAULT_COMPOSE_FILE,
    ):
        if isinstance(value, str) and value.strip():
            path = Path(value.strip()).expanduser().resolve()
            if path.is_file():
                return path
    raise RuntimeError(
        "Claude Code docker-compose file was not found. Set SOMA_CLAUDE_CODE_COMPOSE_FILE "
        "or runtime_options.claude_code_compose_file."
    )


def _resolve_str_option(context: RuntimeExecutionContext, *, option: str, env: str, default: str) -> str:
    for value in (_resolve_runtime_option(context, option), os.getenv(env), default):
        if isinstance(value, str) and value.strip():
            return value.strip()
    return default


def _resolve_compose_service(context: RuntimeExecutionContext) -> str:
    return _resolve_str_option(
        context,
        option="claude_code_service",
        env="SOMA_CLAUDE_CODE_SERVICE",
        default=CLAUDE_CODE_DEFAULT_SERVICE,
    )


def _resolve_permission_mode(context: RuntimeExecutionContext) -> str:
    return _resolve_str_option(
        context,
        option="claude_code_permission_mode",
        env="SOMA_CLAUDE_CODE_PERMISSION_MODE",
        default=CLAUDE_CODE_DEFAULT_PERMISSION_MODE,
    )


def _resolve_extra_disallowed_tools(context: RuntimeExecutionContext) -> str | None:
    """Comma-separated tool names to deny, replacing the runner's benchmark default.

    An empty string is a meaningful value - it clears the default deny list - so this
    returns None for "not configured" rather than falling back to a default here.
    """
    for value in (
        _resolve_runtime_option(context, "claude_code_disallowed_tools"),
        os.getenv("SOMA_CLAUDE_CODE_DISALLOWED_TOOLS"),
    ):
        if isinstance(value, str):
            return value.strip()
        if isinstance(value, (list, tuple)):
            return ",".join(str(item).strip() for item in value)
    return None


def _resolve_max_turns(context: RuntimeExecutionContext) -> int | None:
    """Turn ceiling handed to the SDK, or None (no ceiling) unless one is set explicitly.

    Deliberately not derived from `max_iterations`: the copilot harness ignores it and runs
    until the agent finishes or the wall-clock timeout fires, and a Claude Code run cut off
    at a turn count Copilot never had would not be the same measurement.
    """
    for value in (
        _resolve_runtime_option(context, "claude_code_max_turns"),
        os.getenv("SOMA_CLAUDE_CODE_MAX_TURNS"),
    ):
        if value is None:
            continue
        try:
            parsed = int(str(value).strip())
        except (TypeError, ValueError):
            continue
        if parsed > 0:
            return parsed
    return None


# --------------------------------------------------------------------------------------
# Provider wiring
# --------------------------------------------------------------------------------------


def _resolve_proxy_upstream_base_url(raw_base_url: str) -> str:
    """Upstream root the proxy forwards to, with Claude Code's own `/v1` removed.

    The proxy is path-preserving: it appends the client's request path to this URL. Claude
    Code posts `/v1/messages`, so an upstream configured as `https://host/api/v1` - which
    is what `LLM_BASE_URL` looks like when it was written for an OpenAI-shaped client -
    would be reached at `https://host/api/v1/v1/messages`. Dropping one trailing `v1`
    segment makes the same setting serve both client shapes.

    Override with SOMA_CLAUDE_CODE_UPSTREAM_BASE_URL when the provider's Anthropic-format
    endpoint does not sit at the root of its OpenAI-format one.
    """
    override = os.getenv("SOMA_CLAUDE_CODE_UPSTREAM_BASE_URL", "").strip()
    if override:
        return _normalize_base_url(override)

    normalized = _normalize_base_url(raw_base_url)
    parsed = urlsplit(normalized)
    segments = [segment for segment in parsed.path.split("/") if segment]
    if segments and segments[-1].lower() == _ANTHROPIC_API_PATH_PREFIX:
        segments.pop()
        trimmed = "/" + "/".join(segments)
        if not trimmed.endswith("/"):
            trimmed = f"{trimmed}/"
        rebuilt = urlunsplit((parsed.scheme, parsed.netloc, trimmed, "", ""))
        emit_progress(
            f"[claude-code] proxy upstream {normalized} -> {rebuilt} "
            "(Claude Code appends /v1/messages itself; override with "
            "SOMA_CLAUDE_CODE_UPSTREAM_BASE_URL)",
            component="claude-code",
        )
        return rebuilt
    return normalized


def _normalize_base_url(value: str) -> str:
    parsed = urlsplit(value.strip())
    if not parsed.scheme or not parsed.netloc:
        raise RuntimeError(
            f"Claude Code LLM base URL must be an absolute http(s) URL. Received: {value!r}"
        )
    path = parsed.path or "/"
    if not path.endswith("/"):
        path = f"{path}/"
    return urlunsplit((parsed.scheme, parsed.netloc, path, "", ""))


def _openai_translation_default(context: RuntimeExecutionContext) -> str:
    """"1" unless the run's model is served by Anthropic itself.

    Claude Code speaks one contract - Anthropic Messages - so every other provider needs
    the proxy to translate. Defaulting this off made a working OpenRouter run depend on
    remembering an environment variable, and forgetting it does not fail loudly: the
    request still gets an answer (OpenRouter's Anthropic-compatible endpoint serves
    non-Anthropic models too), it just takes a different path whose prompt-cache
    behaviour does not match the baseline the run is scored against.

    An explicit PROXY_OPENAI_TRANSLATION still wins - this only decides the default.
    """
    model = context.llm_config.get("model")
    if not isinstance(model, str) or not model.strip():
        return "0"  # no model to judge by: keep the historical behaviour
    name = model.strip().lower()
    vendor, _, rest = name.partition("/")
    served_by_anthropic = vendor == "anthropic" or (not rest and name.startswith("claude"))
    return "0" if served_by_anthropic else "1"


# --------------------------------------------------------------------------------------
# Trajectory
# --------------------------------------------------------------------------------------


def _iter_trajectory_entries(trajectory_path: Path) -> Iterator[dict[str, Any]]:
    if not trajectory_path.is_file():
        return
    with trajectory_path.open(encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(entry, dict):
                yield entry


def _count_trajectory_steps(trajectory_path: Path) -> int | None:
    """Agent steps from the runner's JSONL.

    `ResultMessage.num_turns` is the SDK's own count and is preferred; counting
    `AssistantMessage` lines is the fallback for a run that was killed before the SDK
    emitted its terminal message, where the count still has to mean something.
    """
    try:
        assistant_messages = 0
        num_turns: int | None = None
        for entry in _iter_trajectory_entries(trajectory_path):
            entry_type = entry.get("type")
            if entry_type == "AssistantMessage":
                assistant_messages += 1
            elif entry_type == "ResultMessage":
                raw_turns = entry.get("num_turns")
                if isinstance(raw_turns, int) and raw_turns > 0:
                    num_turns = raw_turns
        if num_turns is not None:
            return num_turns
        return assistant_messages or None
    except Exception:
        return None


def _extract_sdk_usage(trajectory_path: Path) -> dict[str, Any]:
    """Token usage and cost as the SDK reported them, normalised to the proxy's key names.

    Kept alongside the proxy's own accounting rather than replacing it: the proxy counts
    what crossed the compression boundary, which is the number the benchmark scores, while
    this is what the agent itself believes it spent. When they disagree, that difference is
    the finding.
    """
    totals = {
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_read_tokens": 0,
        "cache_creation_tokens": 0,
    }
    total_cost_usd = 0.0
    saw_usage = False
    try:
        for entry in _iter_trajectory_entries(trajectory_path):
            if entry.get("type") != "ResultMessage":
                continue
            cost = entry.get("total_cost_usd")
            if isinstance(cost, (int, float)):
                total_cost_usd += float(cost)
            usage = entry.get("usage")
            if not isinstance(usage, dict):
                continue
            saw_usage = True
            totals["input_tokens"] += int(usage.get("input_tokens") or 0)
            totals["output_tokens"] += int(usage.get("output_tokens") or 0)
            totals["cache_read_tokens"] += int(usage.get("cache_read_input_tokens") or 0)
            totals["cache_creation_tokens"] += int(usage.get("cache_creation_input_tokens") or 0)
    except Exception:
        return {}
    if not saw_usage:
        return {}
    if total_cost_usd:
        totals["total_cost_usd"] = round(total_cost_usd, 6)
    return totals


def _extract_stream_error(trajectory_path: Path) -> str | None:
    """The reason a run failed, which docker compose's own output never carries.

    compose stdout/stderr only ever holds provisioning noise ("Container X Created"); a
    404 from the provider or a missing CLI surfaces as a runner.error line, or as an error
    ResultMessage, in the trajectory.
    """
    runner_error: str | None = None
    result_error: str | None = None
    try:
        for entry in _iter_trajectory_entries(trajectory_path):
            entry_type = entry.get("type")
            if entry_type == "runner.error":
                message = entry.get("error")
                if isinstance(message, str) and message.strip():
                    runner_error = message.strip()
            elif entry_type == "ResultMessage" and entry.get("is_error"):
                for key in ("result", "terminal_reason", "subtype"):
                    message = entry.get(key)
                    if isinstance(message, str) and message.strip():
                        result_error = message.strip()
                        break
                errors = entry.get("errors")
                if isinstance(errors, list) and errors:
                    result_error = "; ".join(str(item) for item in errors)[:400]
    except Exception:
        return None
    return runner_error or result_error


def _error_summary(*, trajectory_path: Path | None, stderr: str, stdout: str) -> str:
    if trajectory_path is not None:
        stream_error = _extract_stream_error(trajectory_path)
        if stream_error:
            return " ".join(stream_error.split())[:400]
    for candidate in (stderr, stdout):
        normalized = " ".join(candidate.split())
        if normalized:
            return normalized[:400]
    return "Claude Code execution failed"


# --------------------------------------------------------------------------------------
# Harness
# --------------------------------------------------------------------------------------


class ClaudeCodeHarness(AgentHarness):
    name = "claude-code"
    display_name = "Claude Code"
    default_image = CLAUDE_CODE_DEFAULT_IMAGE
    compose_project_prefix = CLAUDE_CODE_PROJECT_NAME_PREFIX

    def resolve_compose_file(self, context: RuntimeExecutionContext) -> Path:
        return _resolve_compose_file(context)

    def resolve_compose_service(self, context: RuntimeExecutionContext) -> str:
        return _resolve_compose_service(context)

    def proxy_upstream_base_url(self, raw_base_url: str) -> str:
        return _resolve_proxy_upstream_base_url(raw_base_url)

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
        # Mounted rather than passed as argv: benchmark prompts routinely run to thousands of
        # characters with embedded newlines and quotes, and a file survives `compose run`
        # argument handling byte-for-byte.
        host_prompt_path = tmp_run_dir / "prompt.txt"
        host_prompt_path.write_text(prompt, encoding="utf-8")

        # Anthropic -> OpenAI wire translation in the proxy sidecar. Forwarded explicitly
        # because compose interpolates from this process's environment, so a variable left
        # unset here silently takes the compose default instead of what the caller asked for.
        for name, default in (
            ("PROXY_OPENAI_TRANSLATION", _openai_translation_default(context)),
            ("PROXY_OPENAI_CARRY_REASONING", "1"),
            ("PROXY_OPENAI_SORT_TOOLS", "1"),
            ("PROXY_OPENAI_REASONING_EFFORT", ""),
        ):
            env[name] = os.getenv(name, default)

        for name in _AGENT_ENV_NAMES:
            env.pop(name, None)

        network_isolated = proxy_url is not None
        if network_isolated:
            # No trailing /v1: the CLI appends /v1/messages to whatever this names.
            env["ANTHROPIC_BASE_URL"] = proxy_url
        elif isinstance(context.llm_config.get("base_url"), str):
            env["ANTHROPIC_BASE_URL"] = _resolve_proxy_upstream_base_url(
                str(context.llm_config["base_url"])
            ).rstrip("/")
        if isinstance(context.llm_config.get("model"), str):
            env["ANTHROPIC_MODEL"] = str(context.llm_config["model"])

        # ANTHROPIC_AUTH_TOKEN puts the credential in `Authorization: Bearer`, which is the
        # header both the compression proxy and the sticky router read. ANTHROPIC_API_KEY
        # would send `x-api-key` instead and never reach them.
        if credential:
            env["ANTHROPIC_AUTH_TOKEN"] = credential
        elif network_isolated:
            # The proxy holds the real upstream credential (or the upstream needs none), but
            # Claude Code still has to see *a* gateway credential: without one it falls back
            # to an interactive claude.ai login, which in a non-TTY container is a hang, not
            # an error. The placeholder only ever reaches the proxy.
            env["ANTHROPIC_AUTH_TOKEN"] = CLAUDE_CODE_PROXY_PLACEHOLDER_TOKEN

        permission_mode = _resolve_permission_mode(context)
        max_turns = _resolve_max_turns(context)
        extra_disallowed_tools = _resolve_extra_disallowed_tools(context)
        env["CLAUDE_CODE_WORKDIR"] = repo_path
        env["CLAUDE_CODE_PERMISSION_MODE"] = permission_mode
        env["CLAUDE_CODE_NETWORK_ISOLATED"] = "true" if network_isolated else "false"
        if max_turns is not None:
            env["CLAUDE_CODE_MAX_TURNS"] = str(max_turns)
        # Version checks, telemetry and release notes all go to hosts the sandbox network
        # cannot route to; off, they stop showing up as blocked connections and startup delay.
        env["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"] = "1"
        # The compression service rewrites the system array, which defeats the positional
        # strip of Claude Code's attribution block and would leak it into the compressed prompt.
        env["CLAUDE_CODE_ATTRIBUTION_HEADER"] = "0"
        # Pre-release capabilities pair a beta header with a body field; an upstream that is
        # not api.anthropic.com rejects the pair with a hard 400 rather than ignoring it.
        env["CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS"] = "1"
        if extra_disallowed_tools is not None:
            env["CLAUDE_CODE_EXTRA_DISALLOWED_TOOLS"] = extra_disallowed_tools

        return AgentLaunch(
            run_args=["-v", f"{host_prompt_path.resolve()}:{CLAUDE_CODE_PROMPT_CONTAINER_PATH}:ro"],
            env_names=[name for name in _AGENT_ENV_NAMES if name in env],
            service_args=["--prompt-file", CLAUDE_CODE_PROMPT_CONTAINER_PATH, "--cwd", repo_path],
            trajectory_path=_resolve_benchmark_trajectory_path(context, harness_name=self.name),
            metadata={
                "command": "docker compose --project-name <name> run claude-code --prompt-file ...",
                "proxy_upstream_base_url": env.get(COPILOT_PROXY_UPSTREAM_ENV, ""),
                "proxy_openai_translation": env["PROXY_OPENAI_TRANSLATION"],
                "anthropic_base_url": env.get("ANTHROPIC_BASE_URL", ""),
                "anthropic_model": env.get("ANTHROPIC_MODEL", ""),
                "permission_mode": permission_mode,
                "max_turns": max_turns,
                "disallowed_tools": extra_disallowed_tools,
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
        return _error_summary(
            trajectory_path=trajectory_path,
            stderr=result.stderr or "",
            stdout=result.stdout or "",
        )

    def result_metadata(self, *, trajectory_path: Path | None) -> dict[str, Any]:
        return {
            "sdk_token_usage": _extract_sdk_usage(trajectory_path) if trajectory_path is not None else {},
        }


CLAUDE_CODE_HARNESS = ClaudeCodeHarness()


def run_claude_code_instance(context: RuntimeExecutionContext) -> RuntimeExecutionResult:
    return run_harness_instance(context, CLAUDE_CODE_HARNESS)


def run_claude_code_batch(
    contexts: list[RuntimeExecutionContext],
    concurrency: int,
) -> list[RuntimeExecutionResult]:
    return run_harness_batch(contexts, concurrency, CLAUDE_CODE_HARNESS)


CLAUDE_CODE_BACKEND = RuntimeBackend(
    name="claude_code",
    default_image=CLAUDE_CODE_DEFAULT_IMAGE,
    default_build_target="container",
    execution_hint=(
        "Use --execute to run benchmark instances through Claude Code (driven by the "
        "Claude Agent SDK for Python) in Docker Compose, on the same sandbox stack as the "
        "copilot backend. By default this backend runs docker compose -f "
        "src/soma_bench/benchmark/backends/claude_code/claude-code-container/docker-compose.yml "
        "run --rm claude-code --prompt-file ..., with the agent's Anthropic Messages traffic "
        "routed through the compression proxy."
    ),
    execute=run_claude_code_instance,
    execute_batch=run_claude_code_batch,
)
