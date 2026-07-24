from __future__ import annotations

import json
import os
import time
import threading
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request as UrlRequest
from urllib.request import urlopen

from app.prompt_catalog import list_prompt_ids, load_prompt_template
from app.token_usage import add_usage_totals, empty_usage_totals, normalize_provider_usage

_THREAD_LOCAL = threading.local()


@dataclass(frozen=True)
class GeneratedTextResult:
    artifact_id: str
    text: str
    prompt_id: str


class _RequestToolContext:
    def __init__(self, *, request_id: str, path: str, query: str, payload: dict[str, Any] | None = None) -> None:
        self.request_id = request_id
        self.path = path
        self.query = query
        self.payload = payload if isinstance(payload, dict) else {}
        self.generated_texts: dict[str, str] = {}
        self.generated_text_prompt_ids: dict[str, str] = {}
        self.usage_totals = empty_usage_totals()
        self._artifact_counter = 0

    def register_generated_text(self, *, text: str, prompt_id: str) -> GeneratedTextResult:
        self._artifact_counter += 1
        artifact_id = f"llm-text-{self._artifact_counter}"
        self.generated_texts[artifact_id] = text
        self.generated_text_prompt_ids[artifact_id] = prompt_id
        return GeneratedTextResult(artifact_id=artifact_id, text=text, prompt_id=prompt_id)

    def add_usage(self, usage: dict[str, int]) -> None:
        self.usage_totals = add_usage_totals(self.usage_totals, usage)


def activate_request_tool_context(
    *,
    request_id: str,
    path: str,
    query: str,
    payload: dict[str, Any] | None = None,
) -> _RequestToolContext:
    context = _RequestToolContext(request_id=request_id, path=path, query=query, payload=payload)
    _THREAD_LOCAL.request_tool_context = context
    return context


def clear_request_tool_context() -> None:
    if hasattr(_THREAD_LOCAL, "request_tool_context"):
        delattr(_THREAD_LOCAL, "request_tool_context")


def get_current_request_tool_context() -> _RequestToolContext:
    context = getattr(_THREAD_LOCAL, "request_tool_context", None)
    if context is None:
        raise RuntimeError("miner tools are only available during an active /transform request")
    return context


def available_prompt_ids() -> list[str]:
    return list_prompt_ids()


def summarize_text(text: str, *, prompt_id: str = "summary_brief") -> GeneratedTextResult:
    return rewrite_text_with_prompt(text, prompt_id=prompt_id)


def compress_text(text: str, *, prompt_id: str = "compression_faithful") -> GeneratedTextResult:
    return rewrite_text_with_prompt(text, prompt_id=prompt_id)


def rewrite_text_with_prompt(text: str, *, prompt_id: str) -> GeneratedTextResult:
    if not isinstance(text, str) or not text:
        raise ValueError("text must be a non-empty string")
    context = get_current_request_tool_context()
    rewritten_text, usage = _call_llm_for_text_rewrite(
        text=text,
        prompt_id=prompt_id,
        request_id=context.request_id,
        path=context.path,
        query=context.query,
    )
    context.add_usage(usage)
    return context.register_generated_text(text=rewritten_text, prompt_id=prompt_id)


def _resolve_llm_request_url() -> str:
    base_url = os.getenv("PROXY_UPSTREAM_BASE_URL", "").strip()
    if not base_url:
        raise RuntimeError("PROXY_UPSTREAM_BASE_URL must be configured to use LLM miner tools")
    return base_url.rstrip("/") + "/chat/completions"


def _resolve_llm_model() -> str:
    model = os.getenv("COMPRESSION_LLM_MODEL", "").strip()
    if model:
        return model
    context = getattr(_THREAD_LOCAL, "request_tool_context", None)
    if context is not None:
        payload_model = context.payload.get("model")
        if isinstance(payload_model, str) and payload_model.strip():
            return payload_model.strip()
    model = os.getenv("COPILOT_MODEL", "").strip()
    if not model:
        raise RuntimeError("COMPRESSION_LLM_MODEL or request payload model must be configured to use LLM miner tools")
    return model


def _resolve_llm_timeout_seconds() -> float:
    raw = os.getenv("COMPRESSION_LLM_TIMEOUT_SECONDS", "30").strip()
    try:
        value = float(raw)
    except ValueError:
        value = 30.0
    return value if value > 0 else 30.0


def _resolve_llm_max_retries() -> int:
    raw = os.getenv("COMPRESSION_LLM_MAX_RETRIES", "3").strip()
    try:
        value = int(raw)
    except ValueError:
        value = 3
    return max(0, value)


def _resolve_llm_retry_backoff_seconds() -> float:
    raw = os.getenv("COMPRESSION_LLM_RETRY_BACKOFF_SECONDS", "1").strip()
    try:
        value = float(raw)
    except ValueError:
        value = 1.0
    return value if value >= 0 else 1.0


def _build_prompt_messages(*, prompt_id: str, text: str, path: str, query: str, request_id: str) -> list[dict[str, str]]:
    template = load_prompt_template(prompt_id)
    user_prompt = (
        f"Request ID: {request_id}\n"
        f"Path: {path}\n"
        f"Query: {query}\n"
        "Rewrite the following span according to the system instructions.\n"
        "<span>\n"
        f"{text}\n"
        "</span>"
    )
    return [
        {"role": "system", "content": template},
        {"role": "user", "content": user_prompt},
    ]


def _extract_response_text(payload: dict[str, Any]) -> str:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise RuntimeError("LLM response did not include choices")
    first = choices[0]
    if not isinstance(first, dict):
        raise RuntimeError("LLM response choice had unexpected shape")
    message = first.get("message")
    if not isinstance(message, dict):
        raise RuntimeError("LLM response did not include a message object")
    content = message.get("content")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        pieces: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text" and isinstance(item.get("text"), str):
                pieces.append(item["text"])
        joined = "".join(pieces).strip()
        if joined:
            return joined
    raise RuntimeError("LLM response did not include text content")


def _call_llm_for_text_rewrite(
    *,
    text: str,
    prompt_id: str,
    request_id: str,
    path: str,
    query: str,
) -> tuple[str, dict[str, int]]:
    request_url = _resolve_llm_request_url()
    model = _resolve_llm_model()
    api_key = os.getenv("COMPRESSION_LLM_API_KEY", "").strip() or os.getenv("PROXY_PROVIDER_API_KEY", "").strip()
    run_id_header_value = os.getenv("PROXY_RUN_ID_HEADER_VALUE", "").strip()
    timeout_seconds = _resolve_llm_timeout_seconds()
    max_retries = _resolve_llm_max_retries()
    retry_backoff_seconds = _resolve_llm_retry_backoff_seconds()
    headers = {"content-type": "application/json"}
    if api_key:
        headers["authorization"] = f"Bearer {api_key}"
    if run_id_header_value:
        headers["X-Run-Id"] = run_id_header_value

    body = json.dumps(
        {
            "model": model,
            "temperature": 0,
            "messages": _build_prompt_messages(
                prompt_id=prompt_id,
                text=text,
                path=path,
                query=query,
                request_id=request_id,
            ),
        },
        ensure_ascii=False,
    ).encode("utf-8")
    last_error: Exception | None = None
    total_attempts = max_retries + 1
    for attempt in range(1, total_attempts + 1):
        request = UrlRequest(request_url, data=body, headers=headers, method="POST")
        try:
            with urlopen(request, timeout=timeout_seconds) as response:
                raw = response.read()
            payload = json.loads(raw)
            if not isinstance(payload, dict):
                raise RuntimeError("LLM rewrite response must be a JSON object")
            rewritten_text = _extract_response_text(payload)
            usage = normalize_provider_usage(payload.get("usage"))
            return rewritten_text, usage
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            last_error = RuntimeError(f"LLM rewrite request failed with HTTP {exc.code}: {detail}")
            if exc.code < 500 or attempt >= total_attempts:
                raise last_error from exc
        except URLError as exc:
            last_error = RuntimeError(f"LLM rewrite request failed: {exc}")
            if attempt >= total_attempts:
                raise last_error from exc
        except json.JSONDecodeError as exc:
            last_error = RuntimeError("LLM rewrite response was not valid JSON")
            if attempt >= total_attempts:
                raise last_error from exc
        except RuntimeError as exc:
            last_error = exc
            if attempt >= total_attempts:
                raise

        if retry_backoff_seconds > 0:
            time.sleep(retry_backoff_seconds)

    if last_error is not None:
        raise last_error
    raise RuntimeError("LLM rewrite request failed without an explicit error")
