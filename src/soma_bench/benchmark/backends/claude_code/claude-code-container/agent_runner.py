#!/usr/bin/env python3
"""Drive Claude Code through the Claude Agent SDK and emit a JSONL trajectory on stdout.

This is the container entrypoint for the `claude_code` runtime backend. It exists
because the Agent SDK is a library, not a CLI: unlike `copilot -p <prompt>`, there is
nothing to exec that both takes a prompt and prints a machine-readable trajectory. The
host-side backend therefore runs this script instead of a vendor binary, and parses the
JSONL it writes exactly the way it parses Copilot's `--output-format json` stream.

Contract with the host (soma_bench/benchmark/backends/claude_code/claude_code.py):

  stdout  one JSON object per line, nothing else. Every line carries a ``type`` naming
          the SDK message class it came from (``AssistantMessage``, ``ResultMessage``,
          ...), plus ``runner.*`` lines for this script's own lifecycle events.
  stderr  free-form. The `claude` CLI's own stderr is forwarded here verbatim.
  exit    0 on a clean result, 1 when the agent ran but ended in an error result
          (`ResultError` - an API failure, a turn-limit stop), 2 when the harness itself
          failed (no CLI, unparseable protocol, a stream that ended with no result).

Everything else is configured through the environment so the compose file stays the
single place that describes the container; see `_options_from_env`.
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

try:
    from claude_agent_sdk import (
        ClaudeAgentOptions,
        ClaudeSDKError,
        CLIJSONDecodeError,
        CLINotFoundError,
        ProcessError,
        ResultError,
        query,
    )
except ImportError as exc:  # pragma: no cover - container build guarantees this
    sys.stderr.write(
        f"claude-agent-sdk is not importable inside the agent container: {exc}\n"
    )
    raise SystemExit(2) from exc


# Tools that can only ever fail behind the benchmark's internal-only sandbox network.
# Left in the tool list they cost the model a turn each time it rediscovers that, so they
# are dropped up front whenever the host says the run is network-isolated.
OFFLINE_DISALLOWED_TOOLS = ("WebSearch", "WebFetch")

# Session-management and orchestration tools the CLI ships with that have no meaning for
# one sandboxed benchmark instance, and that would distort what the run measures if the
# model reached for them. `Workflow` is the sharpest case: it can fan out many agents,
# every one of them billing through the same proxy, so the instance's token total would
# stop describing the trajectory it is scored on. Override with
# CLAUDE_CODE_EXTRA_DISALLOWED_TOOLS (comma-separated) or set it empty to allow them.
BENCHMARK_DISALLOWED_TOOLS = (
    "Workflow",
    "CronCreate",
    "CronDelete",
    "CronList",
    "ScheduleWakeup",
    "SendMessage",
    "ListAgents",
    "EnterWorktree",
    "ExitWorktree",
)


def _emit(payload: dict[str, Any]) -> None:
    """Write one trajectory line. Flushed per line so a killed run keeps its prefix."""
    sys.stdout.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
    sys.stdout.flush()


def _jsonable(value: Any) -> Any:
    """Best-effort conversion of an SDK message into plain JSON types.

    The SDK's messages and content blocks are dataclasses holding other dataclasses,
    so `dataclasses.asdict` covers most of it; anything exotic (an exception, a Path)
    degrades to its string form rather than failing the whole line.
    """
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        payload = {
            field.name: _jsonable(getattr(value, field.name))
            for field in dataclasses.fields(value)
        }
        # Content blocks carry their `type` as a class-level Literal rather than a
        # dataclass field, so asdict-style conversion drops the one key that says whether
        # a block is text, a tool call or thinking. Naming the class keeps every nested
        # block self-describing, and matches how messages are tagged at the top level.
        payload.setdefault("type", type(value).__name__)
        return payload
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _env_str(name: str) -> str | None:
    value = os.environ.get(name, "").strip()
    return value or None


def _env_bool(name: str, *, default: bool) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    return default


def _env_int(name: str) -> int | None:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return None
    try:
        parsed = int(raw)
    except ValueError:
        return None
    return parsed if parsed > 0 else None


def _resolve_prompt(args: argparse.Namespace) -> str:
    """Prompt text, from a file when given, else argv, else stdin.

    The file form is what the backend actually uses: benchmark prompts routinely carry
    newlines and quotes, and a mounted file survives `docker compose run` argument
    handling untouched.
    """
    if args.prompt_file:
        return Path(args.prompt_file).read_text(encoding="utf-8")
    if args.prompt:
        return args.prompt
    return sys.stdin.read()


def _resolve_disallowed_tools() -> list[str]:
    raw_override = os.environ.get("CLAUDE_CODE_EXTRA_DISALLOWED_TOOLS")
    if raw_override is None:
        disallowed = list(BENCHMARK_DISALLOWED_TOOLS)
    else:
        disallowed = [name.strip() for name in raw_override.split(",") if name.strip()]
    if _env_bool("CLAUDE_CODE_NETWORK_ISOLATED", default=True):
        disallowed.extend(
            name for name in OFFLINE_DISALLOWED_TOOLS if name not in disallowed
        )
    return disallowed


def _options_from_env(*, cwd: str) -> ClaudeAgentOptions:

    return ClaudeAgentOptions(
        cwd=cwd,
        model=_env_str("ANTHROPIC_MODEL"),
        # Unattended benchmark run: there is no one to answer a permission prompt, and a
        # blocked tool call burns the instance's whole turn budget waiting.
        permission_mode=os.environ.get("CLAUDE_CODE_PERMISSION_MODE", "bypassPermissions"),
        max_turns=_env_int("CLAUDE_CODE_MAX_TURNS"),
        disallowed_tools=_resolve_disallowed_tools(),
        # `setting_sources=None` (the SDK default) loads no settings.json from disk, so a
        # stray ~/.claude in the image cannot change the agent's behaviour between runs.
        setting_sources=None,
        # Forwarded rather than swallowed: without it, a CLI that dies on startup leaves
        # the host with an empty trajectory and no reason.
        stderr=lambda line: print(line, file=sys.stderr, flush=True),
    )


async def _run(prompt: str, *, cwd: str) -> int:
    options = _options_from_env(cwd=cwd)
    started = time.monotonic()

    _emit(
        {
            "type": "runner.start",
            "cwd": cwd,
            "model": options.model,
            "permission_mode": options.permission_mode,
            "max_turns": options.max_turns,
            "disallowed_tools": options.disallowed_tools,
            "prompt_chars": len(prompt),
        }
    )

    exit_code = 0
    saw_result = False
    try:
        async for message in query(prompt=prompt, options=options):
            payload = _jsonable(message)
            payload["type"] = type(message).__name__
            _emit(payload)
            if type(message).__name__ == "ResultMessage":
                saw_result = True
                if getattr(message, "is_error", False):
                    exit_code = 1
    except ResultError as exc:
        # The agent ran; its own result says the run failed. Distinguished from the
        # harness failures below so the host can tell "the model could not finish" from
        # "the container could not start the model".
        _emit(
            {
                "type": "runner.error",
                "error_class": type(exc).__name__,
                "error": str(exc),
                "subtype": exc.subtype,
                "terminal_reason": exc.terminal_reason,
                "api_error_status": exc.api_error_status,
            }
        )
        return 1
    except (CLINotFoundError, CLIJSONDecodeError, ProcessError, ClaudeSDKError) as exc:
        _emit({"type": "runner.error", "error_class": type(exc).__name__, "error": str(exc)})
        return 2
    except Exception as exc:  # noqa: BLE001 - the host only ever sees stdout/exit code
        _emit({"type": "runner.error", "error_class": type(exc).__name__, "error": str(exc)})
        return 2

    if not saw_result:
        # The stream ended without the SDK's terminal message, which means the CLI went
        # away mid-run. Reported as a failure so the host does not score a partial run.
        _emit({"type": "runner.error", "error": "SDK stream ended without a ResultMessage"})
        exit_code = 2

    _emit(
        {
            "type": "runner.end",
            "exit_code": exit_code,
            "duration_seconds": round(time.monotonic() - started, 3),
        }
    )
    return exit_code


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Claude Code over the Claude Agent SDK.")
    parser.add_argument("-p", "--prompt", help="Prompt text. Prefer --prompt-file.")
    parser.add_argument("--prompt-file", help="File holding the prompt text.")
    parser.add_argument(
        "--cwd",
        default=os.environ.get("CLAUDE_CODE_WORKDIR") or os.getcwd(),
        help="Working directory the agent operates in (the repository checkout).",
    )
    args = parser.parse_args()

    prompt = _resolve_prompt(args).strip()
    if not prompt:
        sys.stderr.write("agent_runner: refusing to run with an empty prompt\n")
        return 2

    return asyncio.run(_run(prompt, cwd=args.cwd))


if __name__ == "__main__":
    raise SystemExit(main())
