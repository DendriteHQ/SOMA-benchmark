from __future__ import annotations

from typing import Any

TOKEN_USAGE_KEYS = (
    "input_tokens",
    "output_tokens",
    "cache_read_tokens",
    "cache_creation_tokens",
)


def empty_usage_totals() -> dict[str, int]:
    return {key: 0 for key in TOKEN_USAGE_KEYS}


def normalize_provider_usage(usage: dict[str, Any] | None) -> dict[str, int]:
    totals = empty_usage_totals()
    if not isinstance(usage, dict):
        return totals

    prompt_details = usage.get("prompt_tokens_details")
    if not isinstance(prompt_details, dict):
        prompt_details = {}

    if "input_tokens" in usage:
        raw_input = usage.get("input_tokens") or 0
        cache_read = usage.get("cache_read_input_tokens") or 0
        cache_write = usage.get("cache_creation_input_tokens") or 0
    else:
        raw_prompt = usage.get("prompt_tokens") or 0
        cache_read = prompt_details.get("cached_tokens") or 0
        cache_write = prompt_details.get("cache_write_tokens") or 0
        raw_input = raw_prompt - cache_read

    totals["input_tokens"] = max(int(raw_input), 0)
    totals["output_tokens"] = int(usage.get("output_tokens") or usage.get("completion_tokens") or 0)
    totals["cache_read_tokens"] = int(cache_read or 0)
    totals["cache_creation_tokens"] = int(cache_write or 0)
    return totals


def add_usage_totals(base: dict[str, int], delta: dict[str, int] | None) -> dict[str, int]:
    updated = {
        key: int(base.get(key, 0))
        for key in TOKEN_USAGE_KEYS
    }
    if not isinstance(delta, dict):
        return updated
    for key in TOKEN_USAGE_KEYS:
        updated[key] += max(int(delta.get(key, 0) or 0), 0)
    return updated
