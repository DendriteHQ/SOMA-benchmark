from __future__ import annotations

import json
from typing import Any

# Example Jev use (soma-compressor-services/1, see app/services.py). Every turn on which
# the proxy offers Jev, ask it one cheap yes/no question about the latest tool output and
# print the answer: stdout lands in the run's [compression-service][compressor.exec] log,
# and the proxy logs the call and its billed tokens. The messages are always returned
# unchanged, so this stays an identity compressor that only exercises the Jev path. Set
# ASK_JEV = False for a pure identity compressor that never spends Jev tokens.
ASK_JEV = True
PREVIEW_CHARS = 600  # what Jev sees of the tool output; its input tokens are billed


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            part.get("text", "") for part in content if isinstance(part, dict) and isinstance(part.get("text"), str)
        )
    return ""


def _ask_jev(messages: list[Any], metadata: dict[str, Any] | None, services: Any) -> None:
    last_tool = next(
        (m for m in reversed(messages) if isinstance(m, dict) and m.get("role") == "tool"),
        None,
    )
    if last_tool is None:
        return
    state = {
        "task": str((metadata or {}).get("task") or "")[:1000],
        "last_tool_output": _text(last_tool.get("content"))[:PREVIEW_CHARS],
    }
    # Keys are ids we choose; Jev answers under the same keys in result["answers"].
    questions = {
        "last_tool_output_failed": "Does last_tool_output show a command error or failure? Answer yes or no.",
    }
    try:
        result = services.jev.decide(state=state, questions=questions)
    except services.ServiceError as exc:
        # Carry on without the answer: a compressor must never break the agent's request.
        print(f"[base-miner][jev] call failed: {exc.code} {exc.message}")
        return
    print(
        f"[base-miner][jev] model={result.get('model')} "
        f"answers={json.dumps(result.get('answers'))[:500]} usage={json.dumps(result.get('usage'))}"
    )


def compress_messages(
    messages: list[Any] | None = None,
    path: str | None = None,
    metadata: dict[str, Any] | None = None,
    services: Any = None,
) -> list[Any]:
    """Identity compressor: return incoming messages unchanged (optionally consulting Jev)."""
    del path
    if not isinstance(messages, list):
        return []
    if ASK_JEV and services is not None and services.offers("jev"):
        _ask_jev(messages, metadata, services)
    return messages
