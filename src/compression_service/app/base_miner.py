from __future__ import annotations

from typing import Any


def compress_messages(
    messages: list[Any] | None = None,
    path: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> list[Any]:
    """Identity compressor: return incoming messages unchanged."""
    del path, metadata
    if isinstance(messages, list):
        return messages
    return []
