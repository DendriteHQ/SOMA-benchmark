from __future__ import annotations

import asyncio
import codecs
import gzip
import json
import os
import time
import uuid
import zlib
from concurrent.futures import ThreadPoolExecutor
from collections.abc import AsyncIterator
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit, urlunsplit
from urllib.request import Request as UrlRequest
from urllib.request import urlopen

import anyio
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

DEFAULT_UPSTREAM_TIMEOUT_SECONDS = 120.0
DEFAULT_COMPRESSION_TIMEOUT_SECONDS = 30.0
DEFAULT_COMPRESSION_BASE_URL = "http://compression-service:8000/"
SSE_CONTENT_TYPE = "text/event-stream"
SSE_RELAY_CHUNK_BYTES = 8192
HOP_BY_HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
    "host",
    "content-length",
}

app = FastAPI(title="SOMA Copilot Custom Proxy", version="0.1.0")

_token_lock = asyncio.Lock()
_token_totals: dict[str, int] = {
    "input_tokens": 0,
    "output_tokens": 0,
    "cache_read_tokens": 0,
    "cache_creation_tokens": 0,
}

TOKEN_USAGE_LOG_MARKER = "[proxy][token-usage] "
SERVICE_CALL_LOG_MARKER = "[proxy][service-call] "
SERVICE_USAGE_LOG_MARKER = "[proxy][service-usage] "

#: Compressor services. The compression service cannot open a
#: connection anywhere, so a miner's service call comes back in its /transform answer
#: as `pending`, this proxy executes it, and /transform/resume carries the result in.
COMPRESSOR_SERVICES_PROTOCOL = "soma-compressor-services/1"
#: What a miner may ask of Jev on one agent turn. The proxy is the only place these
#: are enforced: the miner is a file the run mounts and can be anything.
#: The deadline is the real limit (the agent waits for the whole turn); rounds and
#: calls are only a guard against a looping miner, since Jev's cost is in the score.
JEV_OFFER: dict[str, Any] = {
    "ops": ["decide"],
    "default_model": "jev-latest",
    "models": ["jev-latest", "jev-1.13"],
    "max_rounds": 50,
    "max_calls": 500,
    "max_calls_per_round": 16,
    "max_request_bytes": 262144,
    "deadline_ms": 20000,
}
#: OpenRouter's System One endpoint, relative to the upstream base URL.
JEV_UPSTREAM_PATH = "systemone"
_SERVICE_POOL = ThreadPoolExecutor(max_workers=JEV_OFFER["max_calls_per_round"])
#: Cumulative cost of compressor services for this run (one proxy per run), kept
#: apart from _token_totals: it is part of what the run spent, not what the agent did.
#: Jev bills input only, so input_tokens is the whole of its token cost.
_service_totals: dict[str, dict[str, float]] = {}

MESSAGE_LOG_MARKER = "[proxy][messages.{}] "


def _emit_message_event(*, stage: str, request_id: str, path: str, query: str,
                        run_id: str, payload: dict[str, Any]) -> None:
    """Record the messages at the proxy boundary as one JSON log line per request."""
    messages = payload.get("messages")
    if not isinstance(messages, list):
        return
    entry = {
        "request_id": request_id,
        "run_id": run_id,
        "path": path,
        "query": query,
        "model": payload.get("model"),
        "messages": messages,
    }
    print(f"{MESSAGE_LOG_MARKER.format(stage)}{json.dumps(entry, ensure_ascii=False)}", flush=True)


def _decompress_for_parsing(body: bytes, content_encoding: str) -> bytes:
    """Best-effort decompression of `body` for local usage-extraction only.

    Standalone runs talk to the upstream LLM directly (no gateway in between to
    absorb `Content-Encoding` the way `SOMA/gateway`'s httpx client does), so a
    compressed response body reaches us as-is. Never raises — a decode failure
    falls back to the original bytes, which downstream json.loads() rejects the
    same way it already does for any other unparseable payload.
    """
    encoding = (content_encoding or "").strip().lower()
    try:
        if encoding == "gzip" or encoding == "x-gzip":
            return gzip.decompress(body)
        if encoding == "deflate":
            try:
                return zlib.decompress(body)
            except zlib.error:
                # Some servers send raw deflate without the zlib header.
                return zlib.decompress(body, -zlib.MAX_WBITS)
    except Exception:
        return body
    return body


def _extract_usage_from_response(response_body: bytes, content_encoding: str = "") -> dict | None:
    decoded_body = _decompress_for_parsing(response_body, content_encoding)
    try:
        data = json.loads(decoded_body)
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    usage = data.get("usage")
    if not isinstance(usage, dict):
        return None
    return usage


async def _accumulate_token_usage(usage: dict) -> None:
    prompt_details = usage.get("prompt_tokens_details")
    if not isinstance(prompt_details, dict):
        prompt_details = {}

    # Anthropic format: input_tokens = non-cached only, cache_read_input_tokens = cached
    # OpenAI/Qwen format: prompt_tokens = total (includes cached), cached_tokens = subset
    # Normalize to Anthropic semantics so total = input + cached + output (no double-count)
    if "input_tokens" in usage:
        # `or 0`, not `get(..., 0)`: these keys are PRESENT with a null value on an
        # Anthropic-format response that did no cache write (OpenRouter's Anthropic
        # endpoint sends `"cache_creation_input_tokens": null`), and a default only
        # applies to a missing key. The `+=` below then raised TypeError and the whole
        # request came back 500 - which is what a non-streaming Anthropic-shaped call,
        # such as Claude Code's /v1/messages/count_tokens, ran into on every attempt.
        raw_input = usage.get("input_tokens") or 0
        cache_read = usage.get("cache_read_input_tokens") or 0
        cache_write = usage.get("cache_creation_input_tokens") or 0
    else:
        raw_prompt = usage.get("prompt_tokens") or 0
        cache_read = prompt_details.get("cached_tokens") or 0
        cache_write = prompt_details.get("cache_write_tokens") or 0
        raw_input = raw_prompt - cache_read  # non-cached portion only

    async with _token_lock:
        _token_totals["input_tokens"] += max(raw_input, 0)
        _token_totals["output_tokens"] += (
            usage.get("output_tokens") or usage.get("completion_tokens") or 0
        )
        _token_totals["cache_read_tokens"] += cache_read
        _token_totals["cache_creation_tokens"] += cache_write
        snapshot = dict(_token_totals)
    print(f"{TOKEN_USAGE_LOG_MARKER}{json.dumps(snapshot)}", flush=True)


def _sse_usage_fields(payload: dict) -> dict[str, int] | None:
    """Anthropic-shaped token counts carried by one SSE event, or None.

    Anthropic splits them across the stream: `message_start` carries the input and cache
    counts under `message.usage`, `message_delta` carries the running output count under a
    top-level `usage`. OpenAI-shaped streams put everything on the final chunk.
    """
    usage = payload.get("usage")
    if not isinstance(usage, dict):
        message = payload.get("message")
        usage = message.get("usage") if isinstance(message, dict) else None
    if not isinstance(usage, dict):
        return None

    if "input_tokens" in usage or "output_tokens" in usage:
        return {
            "input_tokens": int(usage.get("input_tokens") or 0),
            "output_tokens": int(usage.get("output_tokens") or 0),
            "cache_read_input_tokens": int(usage.get("cache_read_input_tokens") or 0),
            "cache_creation_input_tokens": int(usage.get("cache_creation_input_tokens") or 0),
        }

    prompt_details = usage.get("prompt_tokens_details")
    if not isinstance(prompt_details, dict):
        prompt_details = {}
    cache_read = int(prompt_details.get("cached_tokens") or usage.get("cached_tokens") or 0)
    prompt_tokens = int(usage.get("prompt_tokens") or 0)
    return {
        # Normalized to Anthropic semantics (non-cached input only) so the totals stay
        # additive whichever shape the upstream speaks.
        "input_tokens": max(prompt_tokens - cache_read, 0),
        "output_tokens": int(usage.get("completion_tokens") or 0),
        "cache_read_input_tokens": cache_read,
        "cache_creation_input_tokens": int(prompt_details.get("cache_write_tokens") or 0),
    }


def _merge_sse_usage(totals: dict[str, int], fields: dict[str, int]) -> None:
    """Fold one event's counts into the response total by taking the maximum.

    Summing would double-count: every `message_delta` repeats the running output total,
    and later events often repeat the input counts verbatim.
    """
    for key, value in fields.items():
        if value > totals.get(key, 0):
            totals[key] = value


def _scan_sse_chunk_for_usage(buffer: str, totals: dict[str, int]) -> str:
    """Consume whole `data:` lines from `buffer`, returning the unparsed remainder."""
    while True:
        newline_index = buffer.find("\n")
        if newline_index < 0:
            return buffer
        line, buffer = buffer[:newline_index].strip(), buffer[newline_index + 1:]
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if not payload or payload == "[DONE]":
            continue
        try:
            event = json.loads(payload)
        except (json.JSONDecodeError, ValueError):
            continue
        if not isinstance(event, dict):
            continue
        fields = _sse_usage_fields(event)
        if fields:
            _merge_sse_usage(totals, fields)


async def _relay_event_stream(upstream_response) -> AsyncIterator[bytes]:
    """Forward an SSE body to the client as it arrives, tallying usage on the way past.

    Streaming clients watch the byte stream: Claude Code, for one, aborts a response that
    goes silent for 300s, and the upstream's keep-alive pings are the only traffic during
    a long thinking pause. Buffering the whole body the way the JSON path does would turn
    every such pause into a stall and, past the timeout, a client-side abort.

    The bytes are forwarded untouched; the decode is only for reading usage off the events,
    and is incremental so a multi-byte character split across two reads is not mangled.
    """
    totals: dict[str, int] = {}
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    pending = ""
    try:
        while True:
            chunk = await anyio.to_thread.run_sync(
                upstream_response.read, SSE_RELAY_CHUNK_BYTES
            )
            if not chunk:
                break
            yield chunk
            pending = _scan_sse_chunk_for_usage(pending + decoder.decode(chunk), totals)
    finally:
        upstream_response.close()

    # Deliberately after the loop rather than in the `finally`: when the client hangs up
    # mid-stream the generator is closed at the `yield`, and awaiting anything during that
    # unwind raises "async generator ignored GeneratorExit". An aborted response also never
    # delivered its final usage event, so there is no complete total to record anyway.
    if totals:
        await _accumulate_token_usage(totals)


def _coerce_bool(raw_value: Any, *, default: bool = False) -> bool:
    if isinstance(raw_value, bool):
        return raw_value
    if isinstance(raw_value, str):
        value = raw_value.strip().lower()
        if value in {"1", "true", "yes", "on"}:
            return True
        if value in {"0", "false", "no", "off"}:
            return False
    return default


def _normalize_base_url(raw_url: str, *, error_label: str) -> str:
    parsed = urlsplit(raw_url)
    if not parsed.scheme or not parsed.netloc:
        raise RuntimeError(f"{error_label} must be absolute http(s). Received: {raw_url!r}")
    path = parsed.path or "/"
    if not path.endswith("/"):
        path = f"{path}/"
    return urlunsplit((parsed.scheme, parsed.netloc, path, "", ""))


def _resolve_upstream_base_url() -> str:
    for env_name in (
        "PROXY_UPSTREAM_BASE_URL",
        "COMPACT_BENCH_LLM_BASE_URL",
    ):
        candidate = os.getenv(env_name, "").strip()
        if candidate:
            return _normalize_base_url(candidate, error_label=env_name)
    raise RuntimeError("Proxy upstream base URL is not configured. Set PROXY_UPSTREAM_BASE_URL.")


# Resolve upstream URL once at module load — never changes at runtime. Defined AFTER
# the resolver so the module-level call is not a forward reference (was a NameError).
_UPSTREAM_BASE_URL: str = _resolve_upstream_base_url()
_UPSTREAM_NETLOC: str = urlsplit(_UPSTREAM_BASE_URL).netloc


def _resolve_compression_base_url() -> str:
    candidate = os.getenv("PROXY_COMPRESSION_BASE_URL", DEFAULT_COMPRESSION_BASE_URL).strip() or DEFAULT_COMPRESSION_BASE_URL
    return _normalize_base_url(candidate, error_label="PROXY_COMPRESSION_BASE_URL")


def _resolve_compression_route_map() -> dict[str, str]:
    raw = os.getenv("PROXY_COMPRESSION_ROUTE_MAP", "").strip()
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError("PROXY_COMPRESSION_ROUTE_MAP must be valid JSON") from exc
    if not isinstance(parsed, dict):
        raise RuntimeError("PROXY_COMPRESSION_ROUTE_MAP must be a JSON object of run_id to base URL")

    normalized: dict[str, str] = {}
    for key, value in parsed.items():
        run_id = str(key).strip()
        if not run_id:
            continue
        if not isinstance(value, str) or not value.strip():
            continue
        normalized[run_id] = _normalize_base_url(value.strip(), error_label=f"PROXY_COMPRESSION_ROUTE_MAP[{run_id}]")
    return normalized


def _resolve_compression_base_url_template() -> str:
    return os.getenv("PROXY_COMPRESSION_BASE_URL_TEMPLATE", "").strip()


def _resolve_timeout_seconds(env_name: str, default: float) -> float:
    raw = os.getenv(env_name, "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    return value if value > 0 else default


def _resolve_upstream_timeout_seconds() -> float:
    return _resolve_timeout_seconds("PROXY_UPSTREAM_TIMEOUT_SECONDS", DEFAULT_UPSTREAM_TIMEOUT_SECONDS)


def _resolve_compression_timeout_seconds() -> float:
    return _resolve_timeout_seconds("PROXY_COMPRESSION_TIMEOUT_SECONDS", DEFAULT_COMPRESSION_TIMEOUT_SECONDS)


def _resolve_provider_api_key_override() -> str:
    return os.getenv("PROXY_PROVIDER_API_KEY", "").strip()


def _extract_run_id_from_bearer_token(token: str) -> str:
    normalized = token.strip()
    if normalized.lower().startswith("bearer "):
        normalized = normalized[7:].strip()
    return normalized


def _resolve_run_id_for_routing(request: Request, *, override_api_key: str) -> str:
    raw_auth = request.headers.get("authorization", "")
    if isinstance(raw_auth, str) and raw_auth.strip():
        return _extract_run_id_from_bearer_token(raw_auth)
    if override_api_key:
        return _extract_run_id_from_bearer_token(override_api_key)
    return ""


def _resolve_compression_base_url_for_run(*, run_id: str) -> str:
    route_map = _resolve_compression_route_map()
    if run_id and run_id in route_map:
        return route_map[run_id]

    template = _resolve_compression_base_url_template()
    if run_id and template:
        candidate = template.replace("{run_id}", run_id)
        return _normalize_base_url(candidate, error_label="PROXY_COMPRESSION_BASE_URL_TEMPLATE")

    return _resolve_compression_base_url()


def _resolve_run_id_header_value() -> str:
    return os.getenv("PROXY_RUN_ID_HEADER_VALUE", "").strip()


def _resolve_compression_enabled() -> bool:
    return _coerce_bool(os.getenv("PROXY_COMPRESSION_ENABLED", "true"), default=True)


def _build_upstream_url(*, base_url: str, path: str, query: str) -> str:
    base = urlsplit(base_url)
    base_path = base.path if base.path else "/"
    if not base_path.endswith("/"):
        base_path = f"{base_path}/"
    normalized_path = path.lstrip("/")
    full_path = f"{base_path}{normalized_path}" if normalized_path else base_path
    return urlunsplit((base.scheme, base.netloc, full_path, query, ""))


def _forwardable_headers(request: Request) -> dict[str, str]:
    payload: dict[str, str] = {}
    for key, value in request.headers.items():
        if key.lower() in HOP_BY_HOP_HEADERS:
            continue
        payload[key] = value
    return payload


def _response_headers_from_upstream(headers: list[tuple[str, str]]) -> dict[str, str]:
    payload: dict[str, str] = {}
    for key, value in headers:
        if key.lower() in HOP_BY_HOP_HEADERS:
            continue
        payload[key] = value
    return payload


# "developer" is the o-series alias for the system role in the OpenAI API.
_PROTECTED_MESSAGE_ROLES = {"system", "developer"}


def _strip_protected_prompts(payload: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Remove system prompts and the first user prompt from `payload` before compression.

    Every system/developer message is protected, but only the first user
    message in the trajectory is — later user-role messages (e.g. injected
    loop-detection notices) must stay visible to the compressor.

    Returns (stripped_payload, protected). `protected` keeps the removed
    messages with their original indices plus the top-level `system` field
    (Anthropic-style payloads), so `_restore_protected_prompts` can put them
    back after the compression service returns.
    """
    protected_messages: list[tuple[int, Any]] = []
    remaining_messages: list[Any] = []
    first_user_protected = False
    messages = payload.get("messages")
    if isinstance(messages, list):
        for index, message in enumerate(messages):
            role = message.get("role") if isinstance(message, dict) else None
            if role in _PROTECTED_MESSAGE_ROLES:
                protected_messages.append((index, message))
            elif role == "user" and not first_user_protected:
                first_user_protected = True
                protected_messages.append((index, message))
            else:
                remaining_messages.append(message)

    stripped_payload = dict(payload)
    if isinstance(messages, list):
        stripped_payload["messages"] = remaining_messages
    system_field = stripped_payload.pop("system", None)
    protected = {"messages": protected_messages, "system": system_field}
    return stripped_payload, protected


def _restore_protected_prompts(payload: dict[str, Any], protected: dict[str, Any]) -> dict[str, Any]:
    restored_payload = dict(payload)
    messages = restored_payload.get("messages")
    # Drop any system/developer messages the compressor injected — only the
    # originals held by the proxy may carry these roles. User-role messages
    # pass through: apart from the protected first one, they legitimately
    # flow through compression (e.g. loop-detection notices).
    restored_messages = [
        message
        for message in (messages if isinstance(messages, list) else [])
        if not (isinstance(message, dict) and message.get("role") in _PROTECTED_MESSAGE_ROLES)
    ]
    # Ascending original indices; clamp in case the compressor changed the count.
    for index, message in protected["messages"]:
        restored_messages.insert(min(index, len(restored_messages)), message)
    if restored_messages or protected["messages"]:
        restored_payload["messages"] = restored_messages
    if protected["system"] is not None:
        restored_payload["system"] = protected["system"]
    return restored_payload


def _task_context(protected: dict[str, Any]) -> dict[str, Any]:
    """Read-only view of the protected first user prompt, for a compressor to rank by.

    The prompt itself stays stripped from the payload, so the compressor still cannot
    change it; it only learns what the agent was asked to do.
    """
    for _, message in protected.get("messages") or []:
        if isinstance(message, dict) and message.get("role") == "user":
            content = message.get("content")
            if isinstance(content, list):
                content = "\n".join(
                    part.get("text", "") for part in content if isinstance(part, dict) and isinstance(part.get("text"), str)
                )
            if isinstance(content, str) and content:
                return {"task": content}
    return {}


def _resolve_compressor_services_offer() -> dict[str, Any] | None:
    """The `services` field of /transform: what this proxy lets the miner call."""
    raw = os.getenv("PROXY_COMPRESSOR_SERVICES", "jev")
    names = {n.strip().lower() for n in raw.split(",") if n.strip()}
    offer: dict[str, Any] = {}
    if "jev" in names:
        offer["jev"] = dict(JEV_OFFER)
    return {"protocol": COMPRESSOR_SERVICES_PROTOCOL, "offer": offer} if offer else None


def _service_error(call_id: Any, code: str, message: str) -> dict[str, Any]:
    return {"id": call_id, "ok": False, "error": {"code": code, "message": message}}


def _execute_jev_call(body: dict[str, Any], *, headers: dict[str, str], timeout: float) -> tuple[bool, Any]:
    url = _build_upstream_url(base_url=_UPSTREAM_BASE_URL, path=JEV_UPSTREAM_PATH, query="")
    if urlsplit(url).netloc != _UPSTREAM_NETLOC:
        return False, {"code": "invalid_request", "message": "upstream host mismatch"}
    request = UrlRequest(
        url,
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={**headers, "content-type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            parsed = json.loads(response.read())
    except HTTPError as exc:
        detail = exc.read()[:300].decode("utf-8", "replace")
        return False, {"code": "upstream_error", "message": f"HTTP {exc.code}: {detail}"}
    except (URLError, TimeoutError, OSError) as exc:
        return False, {"code": "timeout" if "timed out" in str(exc) else "upstream_error", "message": str(exc)}
    except ValueError as exc:
        return False, {"code": "upstream_error", "message": f"invalid JSON: {exc}"}
    if not isinstance(parsed, dict) or not isinstance(parsed.get("answers"), dict):
        return False, {"code": "upstream_error", "message": "response has no answers"}
    usage = parsed.get("usage")
    input_tokens = usage.get("input_tokens") if isinstance(usage, dict) else None
    if not (isinstance(input_tokens, int) and not isinstance(input_tokens, bool) and input_tokens > 0):
        return False, {"code": "upstream_error", "message": "response has no usable usage"}
    return True, {k: parsed[k] for k in ("model", "answers", "usage") if k in parsed}


class _ServiceBudget:
    """Limits of one agent turn, counted across all its rounds."""

    def __init__(self, offer: dict[str, Any]) -> None:
        self.offer = offer
        self.started = time.monotonic()
        self.rounds = 0
        self.calls = 0

    def remaining_s(self, service: str) -> float:
        limit = (self.offer.get(service) or {}).get("deadline_ms", 0) / 1000.0
        return limit - (time.monotonic() - self.started)


def _run_service_round(
    pending: dict[str, Any],
    *,
    budget: _ServiceBudget,
    headers: dict[str, str],
    request_id: str,
) -> list[dict[str, Any]]:
    """Answer every call of one `pending` round. Never raises for a bad call: a call
    the proxy will not or cannot execute gets an error result and the turn goes on."""
    calls = pending.get("calls") if isinstance(pending.get("calls"), list) else []
    budget.rounds += 1
    results: dict[Any, dict[str, Any]] = {}
    runnable: list[tuple[Any, dict[str, Any]]] = []
    for call in calls:
        cid = call.get("id") if isinstance(call, dict) else None
        service = call.get("service") if isinstance(call, dict) else None
        offer = budget.offer.get(service) if isinstance(service, str) else None
        req = call.get("request") if isinstance(call, dict) else None
        if offer is None:
            results[cid] = _service_error(cid, "not_offered", f"service {service!r} is not offered")
            continue
        if call.get("op") not in offer["ops"] or not isinstance(req, dict):
            results[cid] = _service_error(cid, "invalid_request", "unknown op or missing request")
            continue
        body = {"model": req.get("model") or offer["default_model"], "state": req.get("state"),
                "questions": req.get("questions")}
        if not isinstance(body["model"], str) or body["model"] not in offer["models"]:
            results[cid] = _service_error(cid, "invalid_request", f"model {body['model']!r} is not offered")
            continue
        if body["state"] is None or not isinstance(body["questions"], dict) or not body["questions"]:
            results[cid] = _service_error(cid, "invalid_request", "state and questions are required")
            continue
        if len(json.dumps(body, ensure_ascii=False)) > offer["max_request_bytes"]:
            results[cid] = _service_error(cid, "invalid_request", "request too large")
            continue
        if (budget.rounds > offer["max_rounds"] or budget.calls >= offer["max_calls"]
                or len(runnable) >= offer["max_calls_per_round"] or budget.remaining_s(service) <= 0.2):
            results[cid] = _service_error(cid, "budget_exhausted", "rounds, calls or deadline of this turn used up")
            continue
        budget.calls += 1
        runnable.append((cid, body))

    def run(item: tuple[Any, dict[str, Any]]) -> tuple[Any, bool, Any, float]:
        cid, body = item
        started = time.monotonic()
        ok, value = _execute_jev_call(body, headers=headers, timeout=max(budget.remaining_s("jev"), 0.2))
        return cid, ok, value, (time.monotonic() - started) * 1000.0

    for cid, ok, value, ms in _SERVICE_POOL.map(run, runnable):
        body = dict(runnable)[cid]
        results[cid] = {"id": cid, "ok": True, "response": value} if ok else {"id": cid, **_service_error(cid, value["code"], value["message"])}
        entry = {
            "request_id": request_id, "session": pending.get("session"), "round": pending.get("round"),
            "id": cid, "service": "jev", "model": value.get("model") if ok else body["model"], "ok": ok,
            "ms": round(ms), "state_chars": len(body["state"]) if isinstance(body["state"], str) else len(json.dumps(body["state"])),
            "questions": len(body["questions"]),
        }
        if ok:
            entry["usage"] = value.get("usage")
            usage = value.get("usage") if isinstance(value.get("usage"), dict) else {}
            totals = _service_totals.setdefault("jev", {"calls": 0, "input_tokens": 0, "cost": 0.0})
            totals["calls"] += 1
            totals["input_tokens"] += int(usage.get("input_tokens") or 0)
            totals["cost"] = round(totals["cost"] + float(usage.get("cost") or 0.0), 10)
        else:
            entry["error"] = value
        print(f"{SERVICE_CALL_LOG_MARKER}{json.dumps(entry)}", flush=True)
    if runnable:
        print(f"{SERVICE_USAGE_LOG_MARKER}{json.dumps(_service_totals)}", flush=True)
    for cid, result in results.items():
        if not result.get("ok") and cid not in dict(runnable):
            print(f"{SERVICE_CALL_LOG_MARKER}{json.dumps({'request_id': request_id, 'session': pending.get('session'), 'round': pending.get('round'), 'id': cid, 'ok': False, 'error': result['error']})}", flush=True)
    return [results[c.get("id") if isinstance(c, dict) else None] for c in calls]


def _post_compression_service(url: str, body: dict[str, Any]) -> Any:
    request = UrlRequest(
        url,
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={"content-type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=_resolve_compression_timeout_seconds()) as response:
        return json.loads(response.read())


def _service_auth_headers(request: Request, *, override_api_key: str) -> dict[str, str]:
    """Credentials for a service call: the same identity as the agent request it
    serves, so the cost lands on that run's key. Never shown to the miner."""
    headers: dict[str, str] = {}
    auth = f"Bearer {override_api_key}" if override_api_key else request.headers.get("authorization", "")
    if auth:
        headers["Authorization"] = auth
    run_id_header_value = _resolve_run_id_header_value()
    if run_id_header_value:
        headers["X-Run-Id"] = run_id_header_value
    return headers


def _transform_payload_via_compression_service(
    *,
    path: str,
    query: str,
    payload: dict[str, Any],
    request_id: str,
    compression_base_url: str,
    context: dict[str, Any] | None = None,
    service_headers: dict[str, str] | None = None,
) -> dict[str, Any]:
    transform_url = _build_upstream_url(base_url=compression_base_url, path="transform", query="")
    services = _resolve_compressor_services_offer() if service_headers is not None else None
    body: dict[str, Any] = {
        "path": f"/{path}" if path else "/",
        "query": query,
        "payload": payload,
        "request_id": request_id,
        "context": context or {},
    }
    if services is not None:
        body["services"] = services
    parsed = _post_compression_service(transform_url, body)
    if services is not None:
        budget = _ServiceBudget(services["offer"])
        resume_url = _build_upstream_url(base_url=compression_base_url, path="transform/resume", query="")
        # Bounded by the service: every call over budget is answered with an error,
        # and each answer makes the miner either finish or ask again.
        while isinstance(parsed, dict) and isinstance(parsed.get("pending"), dict):
            # A miner that keeps asking after its budget is gone gets error results
            # for free; past twice the round limit or the deadline it is a failed
            # compression, handled like any other.
            if budget.rounds >= 2 * max(o["max_rounds"] for o in budget.offer.values()) or \
                    min(budget.remaining_s(n) for n in budget.offer) < -5.0:
                raise RuntimeError("compressor kept requesting services past its budget")
            pending = parsed["pending"]
            results = _run_service_round(pending, budget=budget, headers=service_headers or {}, request_id=request_id)
            parsed = _post_compression_service(resume_url, {"session": pending.get("session"), "results": results})
    transformed = parsed.get("payload") if isinstance(parsed, dict) else None
    if not isinstance(transformed, dict):
        raise RuntimeError("Compression service returned invalid payload shape; expected object payload.")
    return transformed



# =============================================================================
# READ COMPRESSION (experiment)
# =============================================================================
# --- Anthropic -> OpenAI request translation ------------------------------------------
#
# Off unless PROXY_OPENAI_TRANSLATION=1. This half turns the body Claude Code sends into
# an OpenAI `chat/completions` body; the two sections below turn the response and the SSE
# stream back into the Anthropic shapes the client is waiting for. All three are needed
# for a run to work, and the request is also re-addressed to `chat/completions` - an
# OpenAI payload posted to `/v1/messages` comes back as a flat 400.
#
# Claude Code has no OpenAI mode to switch on: its CLI binary contains 58 references to
# `v1/messages` and none at all to `chat/completions`, and the Agent SDK is only a
# wrapper that spawns that CLI. So the conversion has to live somewhere in our stack, and
# the proxy is the one place that already parses and rewrites this body.
#
# REASONING IS NOT OPTIONAL TO CARRY. DeepSeek's thinking mode requires the
# `reasoning_content` of an assistant turn that made a tool call to be passed back
# verbatim on every later request; drop it and the second tool-calling turn fails with
# `400 The reasoning_content in the thinking mode must be passed back to the API`. That
# is the bug currently breaking most coding agents on DeepSeek V4, and it is why
# `thinking` blocks are mapped onto `reasoning_content` here rather than discarded.

OPENAI_TRANSLATION_LOG_MARKER = "[proxy][openai-xlat] "

# Where a translated request is sent. Translating the body without re-addressing the
# request leaves an OpenAI payload pointed at the Anthropic endpoint, which upstreams
# answer with a flat 400 - measured, not hypothetical.
OPENAI_CHAT_COMPLETIONS_PATH = "v1/chat/completions"

# Anthropic tool_choice -> OpenAI tool_choice.
_TOOL_CHOICE_MAP = {"auto": "auto", "any": "required", "none": "none"}


def _openai_translation_enabled() -> bool:
    return _coerce_bool(os.getenv("PROXY_OPENAI_TRANSLATION"), default=False)


def _carry_reasoning() -> bool:
    """Whether the reasoning channel is used at all.

    On (the default) the agent's thinking is forwarded as `reasoning_content` and the
    model's reasoning comes back as a `thinking` block. DeepSeek *requires* that round
    trip once it is thinking: drop it and the next tool-calling turn fails with
    `The reasoning_content in the thinking mode must be passed back to the API`.

    Off, the whole channel is disabled - nothing is forwarded, nothing is reconstructed,
    and the upstream is asked not to think. That exists to measure one thing: whether
    thinking mode is what costs the prompt cache. The comparison is copilot, which sends
    no reasoning at all and gets ~98% cache on this same model where we get ~18%.
    """
    return _coerce_bool(os.getenv("PROXY_OPENAI_CARRY_REASONING"), default=True)


def _reasoning_effort() -> str | None:
    """OpenRouter reasoning effort for translated requests (`low`/`medium`/`high`), or None.

    Claude Code's own thinking settings do not survive translation, so without this the
    provider's default effort applies. Only used while the reasoning channel is carried.
    """
    effort = (os.getenv("PROXY_OPENAI_REASONING_EFFORT") or "").strip().lower()
    return effort or None


def _is_anthropic_messages_request(path: str, payload: Any) -> bool:
    """Whether this body is an Anthropic Messages request worth translating.

    Deliberately checks the body and not just the path: the agent reaches this proxy at
    whatever path its base URL produces, and a copilot-style `chat/completions` body must
    never be run through the translator. An Anthropic body is the one that has
    `messages` plus either a top-level `system` (a field OpenAI does not have) or content
    blocks rather than plain strings.
    """
    if not isinstance(payload, dict):
        return False
    messages = payload.get("messages")
    if not isinstance(messages, list) or not messages:
        return False
    if "chat/completions" in path:
        return False
    # `/v1/messages/count_tokens` carries a body almost identical to a real request -
    # same model, messages, system and tools - so nothing in the shape distinguishes it.
    # Translating it would turn every token count into a billed completion, silently.
    if "count_tokens" in path:
        return False
    if isinstance(payload.get("system"), (str, list)):
        return True
    if any(
        isinstance(m, dict) and isinstance(m.get("content"), list) and any(
            isinstance(b, dict) and b.get("type") in
            ("tool_use", "tool_result", "thinking", "redacted_thinking")
            for b in m["content"]
        )
        for m in messages
    ):
        return True
    # A `tools` list carrying `input_schema` is Anthropic's shape; OpenAI nests it under
    # `function.parameters`.
    tools = payload.get("tools")
    return isinstance(tools, list) and any(
        isinstance(t, dict) and "input_schema" in t for t in tools
    )


def _text_of(blocks: Any) -> str:
    """Concatenated text of a content value that is either a string or a block list."""
    if isinstance(blocks, str):
        return blocks
    if not isinstance(blocks, list):
        return ""
    parts = [
        b["text"] for b in blocks
        if isinstance(b, dict) and b.get("type") == "text" and isinstance(b.get("text"), str)
    ]
    return "\n".join(parts)


def _tool_result_text(block: dict) -> str:
    """The text of a tool_result, whose content is either a string or a block list."""
    content = block.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            item.get("text", "")
            for item in content
            if isinstance(item, dict) and item.get("type") == "text"
        ]
        return "\n".join(parts)
    return ""


def _openai_tools(tools: Any) -> list[dict] | None:
    if not isinstance(tools, list) or not tools:
        return None
    out: list[dict] = []
    for tool in tools:
        if not isinstance(tool, dict) or not tool.get("name"):
            continue
        out.append({
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool.get("description") or "",
                "parameters": tool.get("input_schema") or {"type": "object", "properties": {}},
            },
        })
    # Sorted by name, because the tool list is the very start of what the upstream
    # renders and a prefix cache only fires on a byte-identical prefix from token zero.
    # Any turn-to-turn jitter in tool order therefore costs the whole cache, not just
    # the reordered part. Sorting is free when the caller is already stable and is the
    # standard fix when it is not. Disable with PROXY_OPENAI_SORT_TOOLS=0 if a model
    # ever turns out to be sensitive to tool ordering.
    if _coerce_bool(os.getenv("PROXY_OPENAI_SORT_TOOLS"), default=True):
        out.sort(key=lambda t: t["function"]["name"])
    return out or None


def _openai_tool_choice(choice: Any) -> Any:
    if not isinstance(choice, dict):
        return None
    kind = choice.get("type")
    if kind == "tool" and choice.get("name"):
        return {"type": "function", "function": {"name": choice["name"]}}
    return _TOOL_CHOICE_MAP.get(kind)


def _assistant_message(blocks: list) -> dict | None:
    """One Anthropic assistant turn as an OpenAI assistant message.

    `thinking` becomes `reasoning_content` (see the note at the top of this section);
    `redacted_thinking` is dropped because its payload is an encrypted blob that only
    Anthropic can read, and forwarding it to another provider would be noise.
    """
    texts: list[str] = []
    reasoning: list[str] = []
    tool_calls: list[dict] = []
    carry_reasoning = _carry_reasoning()
    for b in blocks:
        if not isinstance(b, dict):
            continue
        btype = b.get("type")
        if btype == "text" and isinstance(b.get("text"), str):
            texts.append(b["text"])
        elif btype == "thinking" and carry_reasoning and isinstance(b.get("thinking"), str):
            reasoning.append(b["thinking"])
        elif btype == "tool_use" and b.get("id"):
            try:
                arguments = json.dumps(b.get("input") or {}, ensure_ascii=False)
            except (TypeError, ValueError):
                arguments = "{}"
            tool_calls.append({
                "id": b["id"],
                "type": "function",
                "function": {"name": b.get("name") or "", "arguments": arguments},
            })
    if not texts and not tool_calls:
        # An assistant turn carrying nothing but `thinking` has no legal OpenAI form:
        # `content` would be empty and there are no tool_calls, which upstreams reject
        # with "Invalid assistant message: content or tool_calls must be set". Reasoning
        # alone does not satisfy it. Such a turn is normally merged into the turn it
        # belongs to by _group_same_role; when it stands alone there is nowhere valid to
        # put it, so it is dropped rather than sent as an invalid message.
        if reasoning:
            print(
                f"{OPENAI_TRANSLATION_LOG_MARKER}dropped a thinking-only assistant turn "
                f"({len(''.join(reasoning))} chars of reasoning) - no legal OpenAI form",
                flush=True,
            )
        return None
    message: dict[str, Any] = {"role": "assistant", "content": "\n".join(texts)}
    if reasoning:
        message["reasoning_content"] = "\n".join(reasoning)
    if tool_calls:
        message["tool_calls"] = tool_calls
    return message


def _user_messages(content: Any) -> list[dict]:
    """One Anthropic user turn as the OpenAI messages it corresponds to.

    Can be more than one. Anthropic carries tool results inside the following user
    message, optionally next to ordinary text; OpenAI wants each result as its own
    `role: "tool"` message, and those must come before any new user text so they still
    sit directly behind the assistant turn that called them.
    """
    if isinstance(content, str):
        return [{"role": "user", "content": content}] if content else []
    if not isinstance(content, list):
        return []
    tool_messages: list[dict] = []
    texts: list[str] = []
    for b in content:
        if not isinstance(b, dict):
            continue
        btype = b.get("type")
        if btype == "tool_result" and b.get("tool_use_id"):
            tool_messages.append({
                "role": "tool",
                "tool_call_id": b["tool_use_id"],
                "content": _tool_result_text(b),
            })
        elif btype == "text" and isinstance(b.get("text"), str):
            texts.append(b["text"])
    out = tool_messages
    if texts:
        out = out + [{"role": "user", "content": "\n".join(texts)}]
    return out


def _group_same_role(messages: list) -> list[dict]:
    """Consecutive same-role messages merged into one, blocks concatenated.

    On the wire Anthropic already groups a turn's blocks into a single message, so this
    is usually a no-op. It matters because the mapping to OpenAI depends on that
    grouping: OpenAI wants one assistant message carrying all of a turn's `tool_calls`,
    with the matching `role: "tool"` messages directly behind it. Fed a split shape -
    one message per block, which is how the SDK's own trajectory records a turn - a
    naive conversion emits `assistant(call A), assistant(call B), tool(B), tool(A)`,
    where tool(B) no longer follows the message that made call B and the request is
    rejected. Merging first makes both shapes converge on the same output.
    """
    grouped: list[dict] = []
    for message in messages:
        if not isinstance(message, dict) or message.get("role") not in (
            "user", "assistant", "system"
        ):
            continue
        content = message.get("content")
        blocks = content if isinstance(content, list) else [{"type": "text", "text": content}] \
            if isinstance(content, str) and content else []
        if grouped and grouped[-1]["role"] == message["role"]:
            grouped[-1]["content"].extend(blocks)
        else:
            grouped.append({"role": message["role"], "content": list(blocks)})
    return grouped


def _anthropic_request_to_openai(payload: dict) -> dict:
    """An Anthropic Messages body as an OpenAI `chat/completions` body.

    `cache_control` markers are not carried over: they are Anthropic's explicit-write
    cache protocol, and every OpenAI-shaped backend we target caches automatically by
    prefix instead. Dropping them is the point of the exercise, not a loss.
    """
    messages: list[dict] = []

    system = payload.get("system")
    system_text = _text_of(system)
    if system_text:
        messages.append({"role": "system", "content": system_text})

    for message in _group_same_role(payload.get("messages") or []):
        role = message["role"]
        content = message["content"]
        if role == "assistant":
            converted = _assistant_message(content)
            if converted is not None:
                messages.append(converted)
        elif role == "user":
            messages.extend(_user_messages(content))
        elif role == "system":
            # Claude Code appends `role: "system"` notes mid-conversation - an environment
            # block, and a `<total_tokens>N tokens left</total_tokens>` counter after every
            # turn. Forwarding them with that role destroys the prompt cache: DeepSeek's
            # chat template hoists system messages to the FRONT of the rendered prompt, so
            # a note appended at the end of the array lands at the start of the prompt and
            # moves the whole prefix. Measured, growing conversation, everything else
            # equal: without them 50% -> 66% -> 75% -> 80% cached; with them 0% on every
            # turn. In a real run it showed up as a cache pinned at ~6.3k - just `tools`,
            # which render ahead of the system block.
            #
            # Emitting them as `user` keeps the text (the model still sees the note) and
            # keeps it where it belongs in the order, so the prefix stays append-only.
            text = _text_of(content)
            if text:
                messages.append({"role": "user", "content": text})

    out: dict[str, Any] = {"model": payload.get("model"), "messages": messages}
    for src, dst in (("max_tokens", "max_tokens"), ("temperature", "temperature"),
                     ("top_p", "top_p"), ("stream", "stream")):
        if payload.get(src) is not None:
            out[dst] = payload[src]
    if not _carry_reasoning():
        # OpenRouter's switch for the provider's thinking mode. Without it DeepSeek keeps
        # thinking, returns reasoning_content, and then demands it back on the next turn -
        # which is exactly the round trip this setting is turning off.
        out["reasoning"] = {"enabled": False}
    elif _reasoning_effort():
        out["reasoning"] = {"effort": _reasoning_effort()}
    if out.get("stream"):
        # Without this an OpenAI-shaped stream carries no usage at all, and the run's
        # token totals - the thing the benchmark actually scores - would come out zero.
        out["stream_options"] = {"include_usage": True}
    if payload.get("stop_sequences"):
        out["stop"] = payload["stop_sequences"]
    tools = _openai_tools(payload.get("tools"))
    if tools:
        out["tools"] = tools
        choice = _openai_tool_choice(payload.get("tool_choice"))
        if choice is not None:
            out["tool_choice"] = choice
    return out


# --- OpenAI -> Anthropic response translation -----------------------------------------
#
# The return leg of the translation above. Non-streaming only; the SSE stream is a
# separate transducer, and since Claude Code always streams, this pair is not yet enough
# to run an instance end to end.
#
# `finish_reason` -> `stop_reason`. `content_filter` maps to `end_turn` rather than
# inventing a reason Claude Code does not know: it ends the turn, which is what happened.
_STOP_REASON_MAP = {
    "stop": "end_turn",
    "tool_calls": "tool_use",
    "function_call": "tool_use",
    "length": "max_tokens",
    "content_filter": "end_turn",
}

# Anthropic marks a thinking block with a signature it later verifies. Nothing on this
# path can produce a real one - the upstream is not Anthropic - so reasoning that arrives
# without a signature gets this marker instead of an empty string, which makes it
# obvious in a trajectory where the block came from. The translator on the request side
# reads only the `thinking` text and ignores the signature, so the round trip is
# unaffected; what must never happen is a real Anthropic signature being altered, and on
# this path there is none to alter.
UNSIGNED_THINKING_MARKER = "unsigned:openai-translation"


def _reasoning_of(message: dict) -> tuple[str, str | None]:
    """(reasoning text, signature) from an OpenAI-shaped assistant message.

    Three shapes in the wild: DeepSeek's `reasoning_content`, OpenRouter's top-level
    `reasoning`, and OpenRouter's `reasoning_details` array. The array wins when present
    because it is the only one that can carry a real signature, and OpenRouter documents
    that the raw `reasoning` field is ignored when it is there.
    """
    details = message.get("reasoning_details")
    if isinstance(details, list) and details:
        texts: list[str] = []
        signature: str | None = None
        for item in details:
            if not isinstance(item, dict):
                continue
            text = item.get("text") or item.get("summary")
            if isinstance(text, str) and text:
                texts.append(text)
            if isinstance(item.get("signature"), str) and item["signature"]:
                signature = item["signature"]
        if texts:
            return "\n".join(texts), signature
    for key in ("reasoning_content", "reasoning"):
        value = message.get(key)
        if isinstance(value, str) and value:
            return value, None
    return "", None


def _anthropic_content_blocks(message: dict) -> list[dict]:
    """The assistant message as Anthropic content blocks, thinking first.

    Order matters: Anthropic requires a turn's thinking block to come before its text
    and tool_use blocks, and Claude Code relies on that ordering when it replays a turn.
    """
    blocks: list[dict] = []
    reasoning, signature = _reasoning_of(message) if _carry_reasoning() else ("", None)
    if reasoning:
        blocks.append({
            "type": "thinking",
            "thinking": reasoning,
            "signature": signature or UNSIGNED_THINKING_MARKER,
        })
    content = message.get("content")
    if isinstance(content, str) and content:
        blocks.append({"type": "text", "text": content})
    elif isinstance(content, list):
        # Some gateways return OpenAI "parts"; keep only the text ones.
        text = "\n".join(
            p["text"] for p in content
            if isinstance(p, dict) and p.get("type") == "text" and isinstance(p.get("text"), str)
        )
        if text:
            blocks.append({"type": "text", "text": text})
    for call in message.get("tool_calls") or []:
        if not isinstance(call, dict):
            continue
        function = call.get("function") if isinstance(call.get("function"), dict) else {}
        raw_arguments = function.get("arguments")
        try:
            parsed = json.loads(raw_arguments) if isinstance(raw_arguments, str) else {}
        except ValueError:
            # A truncated or malformed argument string would otherwise crash the whole
            # response. An empty input lets the agent see the call, fail it and retry,
            # which is recoverable; a 500 from the proxy is not.
            print(
                f"{OPENAI_TRANSLATION_LOG_MARKER}unparseable tool arguments for "
                f"{call.get('id')!r}, sending empty input",
                flush=True,
            )
            parsed = {}
        blocks.append({
            "type": "tool_use",
            "id": call.get("id") or f"toolu_{uuid.uuid4().hex[:24]}",
            "name": function.get("name") or "",
            "input": parsed if isinstance(parsed, dict) else {},
        })
    if not blocks:
        # Anthropic rejects an empty content array, and Claude Code treats a contentless
        # turn as a protocol error rather than as an empty answer.
        blocks.append({"type": "text", "text": ""})
    return blocks


def _openai_response_to_anthropic(payload: dict) -> dict:
    """An OpenAI `chat/completions` response as an Anthropic Messages response."""
    choices = payload.get("choices")
    choice = choices[0] if isinstance(choices, list) and choices else {}
    message = choice.get("message") if isinstance(choice.get("message"), dict) else {}
    blocks = _anthropic_content_blocks(message)

    finish = choice.get("finish_reason")
    stop_reason = _STOP_REASON_MAP.get(finish, "end_turn")
    # A turn that produced a tool call is `tool_use` whatever the upstream called it:
    # Claude Code drives its loop off this field, and a `tool_use` block under
    # `end_turn` would stop the agent with the call unanswered.
    if any(b["type"] == "tool_use" for b in blocks):
        stop_reason = "tool_use"

    usage = _sse_usage_fields(payload) or {
        "input_tokens": 0, "output_tokens": 0,
        "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0,
    }

    message_id = payload.get("id") or uuid.uuid4().hex
    if not str(message_id).startswith("msg_"):
        message_id = f"msg_{message_id}"
    return {
        "id": message_id,
        "type": "message",
        "role": "assistant",
        "model": payload.get("model") or "",
        "content": blocks,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": usage,
    }


# --- OpenAI SSE -> Anthropic SSE (buffering bridge) -----------------------------------
#
# Claude Code always streams, so the JSON pair above is not enough to run an instance.
# This closes the loop the short way: read the whole OpenAI stream, assemble it into one
# response, reuse the (tested) `_openai_response_to_anthropic`, and emit the Anthropic
# event sequence in one burst at the end.
#
# The honest cost is that it is NOT incremental - the agent sees nothing until the model
# has finished the turn. For a benchmark run that is acceptable (nobody is reading the
# output as it arrives) but it is not what a real client wants, and a true per-event
# transducer should replace it before this path is used for anything else.
#
# What it must not do is go quiet: Claude Code aborts a response silent for 300s, so a
# ping goes out while the upstream is still thinking.
TRANSLATION_SSE_PING_SECONDS = 20.0


def _assemble_openai_stream(chunks: list[dict]) -> dict:
    """The OpenAI streamed deltas folded back into one non-streamed response body."""
    text_parts: list[str] = []
    reasoning_parts: list[str] = []
    tool_calls: dict[int, dict] = {}
    finish_reason = None
    usage: dict | None = None
    response_id = ""
    model = ""
    provider = ""

    for chunk in chunks:
        response_id = response_id or chunk.get("id") or ""
        model = model or chunk.get("model") or ""
        provider = provider or chunk.get("provider") or ""
        if isinstance(chunk.get("usage"), dict):
            usage = chunk["usage"]
        choices = chunk.get("choices")
        if not isinstance(choices, list) or not choices:
            continue
        choice = choices[0]
        if choice.get("finish_reason"):
            finish_reason = choice["finish_reason"]
        delta = choice.get("delta")
        if not isinstance(delta, dict):
            continue
        if isinstance(delta.get("content"), str):
            text_parts.append(delta["content"])
        for key in ("reasoning_content", "reasoning"):
            if isinstance(delta.get(key), str):
                reasoning_parts.append(delta[key])
                break
        for call in delta.get("tool_calls") or []:
            if not isinstance(call, dict):
                continue
            # `index` is what ties a call's fragments together; id and name arrive once,
            # arguments arrive a character at a time.
            slot = tool_calls.setdefault(
                call.get("index", len(tool_calls)),
                {"id": "", "type": "function", "function": {"name": "", "arguments": ""}},
            )
            if call.get("id"):
                slot["id"] = call["id"]
            function = call.get("function")
            if isinstance(function, dict):
                if function.get("name"):
                    slot["function"]["name"] = function["name"]
                if isinstance(function.get("arguments"), str):
                    slot["function"]["arguments"] += function["arguments"]

    message: dict[str, Any] = {"role": "assistant", "content": "".join(text_parts)}
    if reasoning_parts:
        message["reasoning_content"] = "".join(reasoning_parts)
    if tool_calls:
        message["tool_calls"] = [tool_calls[i] for i in sorted(tool_calls)]
    body: dict[str, Any] = {
        "id": response_id,
        "model": model,
        "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
    }
    if provider:
        # Carried purely for diagnostics. A prefix cache is per provider, so a run whose
        # turns land on different providers cannot accumulate one - and that is invisible
        # unless the provider is recorded per request.
        body["provider"] = provider
    if usage:
        body["usage"] = usage
    return body


def _sse(event: str, payload: dict) -> bytes:
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n".encode("utf-8")


def _anthropic_sse_events(response: dict) -> list[bytes]:
    """An Anthropic Messages response as the SSE sequence a client expects."""
    usage = response["usage"]
    out = [_sse("message_start", {
        "type": "message_start",
        "message": {
            "id": response["id"], "type": "message", "role": "assistant",
            "model": response["model"], "content": [], "stop_reason": None,
            "stop_sequence": None,
            "usage": {
                "input_tokens": usage["input_tokens"], "output_tokens": 0,
                "cache_read_input_tokens": usage["cache_read_input_tokens"],
                "cache_creation_input_tokens": usage["cache_creation_input_tokens"],
            },
        },
    })]
    for index, block in enumerate(response["content"]):
        kind = block["type"]
        if kind == "thinking":
            out.append(_sse("content_block_start", {
                "type": "content_block_start", "index": index,
                "content_block": {"type": "thinking", "thinking": "", "signature": ""}}))
            out.append(_sse("content_block_delta", {
                "type": "content_block_delta", "index": index,
                "delta": {"type": "thinking_delta", "thinking": block["thinking"]}}))
            out.append(_sse("content_block_delta", {
                "type": "content_block_delta", "index": index,
                "delta": {"type": "signature_delta", "signature": block["signature"]}}))
        elif kind == "tool_use":
            out.append(_sse("content_block_start", {
                "type": "content_block_start", "index": index,
                "content_block": {"type": "tool_use", "id": block["id"],
                                  "name": block["name"], "input": {}}}))
            out.append(_sse("content_block_delta", {
                "type": "content_block_delta", "index": index,
                "delta": {"type": "input_json_delta",
                          "partial_json": json.dumps(block["input"], ensure_ascii=False)}}))
        else:
            out.append(_sse("content_block_start", {
                "type": "content_block_start", "index": index,
                "content_block": {"type": "text", "text": ""}}))
            out.append(_sse("content_block_delta", {
                "type": "content_block_delta", "index": index,
                "delta": {"type": "text_delta", "text": block.get("text", "")}}))
        out.append(_sse("content_block_stop", {"type": "content_block_stop", "index": index}))

    out.append(_sse("message_delta", {
        "type": "message_delta",
        "delta": {"stop_reason": response["stop_reason"], "stop_sequence": None},
        "usage": {"output_tokens": usage["output_tokens"]}}))
    out.append(_sse("message_stop", {"type": "message_stop"}))
    return out


async def _translate_event_stream(upstream_response) -> AsyncIterator[bytes]:
    """Read an OpenAI SSE body, emit it as an Anthropic one."""
    chunks: list[dict] = []
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    pending = ""
    last_ping = anyio.current_time()
    try:
        while True:
            chunk = await anyio.to_thread.run_sync(
                upstream_response.read, SSE_RELAY_CHUNK_BYTES
            )
            if not chunk:
                break
            pending += decoder.decode(chunk)
            while "\n" in pending:
                line, pending = pending.split("\n", 1)
                line = line.strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if not data or data == "[DONE]":
                    continue
                try:
                    chunks.append(json.loads(data))
                except ValueError:
                    continue
            now = anyio.current_time()
            if now - last_ping >= TRANSLATION_SSE_PING_SECONDS:
                last_ping = now
                yield _sse("ping", {"type": "ping"})
    finally:
        upstream_response.close()

    assembled = _assemble_openai_stream(chunks)
    translated = _openai_response_to_anthropic(assembled)
    print(
        f"{OPENAI_TRANSLATION_LOG_MARKER}"
        f"{json.dumps({'stream_chunks': len(chunks), 'provider': assembled.get('provider',''), 'blocks': [b['type'] for b in translated['content']], 'stop_reason': translated['stop_reason'], 'usage': translated['usage']})}",
        flush=True,
    )
    for event in _anthropic_sse_events(translated):
        yield event

    usage = assembled.get("usage")
    if isinstance(usage, dict):
        await _accumulate_token_usage(usage)


@app.get("/health")
def health() -> JSONResponse:
    return JSONResponse(
        {
            "status": "ok",
            "upstream": _UPSTREAM_BASE_URL,
            "compression_base_url": _resolve_compression_base_url(),
            "compression_enabled": _resolve_compression_enabled(),
            "api_key_override_enabled": bool(_resolve_provider_api_key_override()),
            "run_id_header_enabled": bool(_resolve_run_id_header_value()),
        }
    )


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"])
async def proxy_passthrough(path: str, request: Request) -> Response:
    request_id = uuid.uuid4().hex
    upstream_url = _build_upstream_url(base_url=_UPSTREAM_BASE_URL, path=path, query=request.url.query)
    if urlsplit(upstream_url).netloc != _UPSTREAM_NETLOC:
        raise HTTPException(status_code=400, detail="Upstream host mismatch — only configured host is allowed")

    method = request.method.upper()
    body = await request.body()
    content_type = request.headers.get("content-type", "")

    forwarded_body = body
    # Whether the request went upstream translated, which decides whether the response
    # has to be translated back.
    translated_request = False
    override_api_key = _resolve_provider_api_key_override()
    run_id_for_routing = _resolve_run_id_for_routing(request, override_api_key=override_api_key)
    compression_base_url = _resolve_compression_base_url_for_run(run_id=run_id_for_routing)
    parsed_payload = None
    if body and "application/json" in content_type.lower():
        try:
            parsed_payload = json.loads(body)
        except json.JSONDecodeError:
            pass

    if isinstance(parsed_payload, dict):
        _emit_message_event(
            stage="in", request_id=request_id, path=path, query=request.url.query,
            run_id=run_id_for_routing, payload=parsed_payload,
        )

        if _resolve_compression_enabled():
            # Last of the body rewrites on purpose: everything above works on Anthropic
            # content blocks, so the shape has to survive until they are done.
            if _openai_translation_enabled() and _is_anthropic_messages_request(
                path, parsed_payload
            ):
                translated = _anthropic_request_to_openai(parsed_payload)
                print(
                    f"{OPENAI_TRANSLATION_LOG_MARKER}"
                    f"{json.dumps({'messages_in': len(parsed_payload.get('messages') or []), 'messages_out': len(translated['messages']), 'tools': len(translated.get('tools') or []), 'reasoning_carried': sum(1 for m in translated['messages'] if m.get('reasoning_content')), 'reasoning': translated.get('reasoning')})}",
                    flush=True,
                )
                parsed_payload = translated
                translated_request = True
                # The body is not the only Anthropic thing about this request: it was
                # addressed to `/v1/messages`, and an OpenAI body posted there is a 400.
                # The query goes too - `?beta=true` is an Anthropic parameter.
                upstream_url = _build_upstream_url(
                    base_url=_UPSTREAM_BASE_URL,
                    path=OPENAI_CHAT_COMPLETIONS_PATH,
                    query="",
                )
                if urlsplit(upstream_url).netloc != _UPSTREAM_NETLOC:
                    raise HTTPException(
                        status_code=400,
                        detail="Upstream host mismatch — only configured host is allowed",
                    )
            stripped_payload, protected_prompts = _strip_protected_prompts(parsed_payload)
            transformed_payload = _transform_payload_via_compression_service(
                path=path,
                query=request.url.query,
                payload=stripped_payload,
                request_id=request_id,
                compression_base_url=compression_base_url,
                context=_task_context(protected_prompts),
                service_headers=_service_auth_headers(request, override_api_key=override_api_key),
            )
            transformed_payload = _restore_protected_prompts(transformed_payload, protected_prompts)
            forwarded_body = json.dumps(transformed_payload, ensure_ascii=False).encode("utf-8")
        forwarded_payload = json.loads(forwarded_body)
        if isinstance(forwarded_payload, dict):
            _emit_message_event(
                stage="out", request_id=request_id, path=path, query=request.url.query,
                run_id=run_id_for_routing, payload=forwarded_payload,
            )
    forwarded_headers = _forwardable_headers(request)
    run_id_header_value = _resolve_run_id_header_value()
    if run_id_header_value:
        forwarded_headers["X-Run-Id"] = run_id_header_value
    if override_api_key:
        forwarded_headers["Authorization"] = f"Bearer {override_api_key}"

    upstream_request = UrlRequest(
        upstream_url,
        data=forwarded_body if method in {"POST", "PUT", "PATCH", "DELETE"} else None,
        headers=forwarded_headers,
        method=method,
    )

    try:
        upstream_response = urlopen(upstream_request, timeout=_resolve_upstream_timeout_seconds())
    except HTTPError as exc:
        response_body = exc.read()
        response_headers = _response_headers_from_upstream(list(exc.headers.items()))
        return Response(content=response_body, status_code=exc.code, headers=response_headers)
    except URLError as exc:
        raise HTTPException(status_code=502, detail=f"Upstream request failed: {exc}") from exc

    response_headers = _response_headers_from_upstream(list(upstream_response.headers.items()))
    upstream_content_type = upstream_response.headers.get("content-type", "")

    if SSE_CONTENT_TYPE in upstream_content_type.lower():
        # Relayed incrementally rather than buffered; _relay_event_stream explains why, and
        # tallies token usage off the events as they pass. A translated request is the
        # exception: its upstream stream is OpenAI-shaped and has to be rebuilt as an
        # Anthropic one before the client can read a single event of it.
        stream = (
            _translate_event_stream(upstream_response)
            if translated_request
            else _relay_event_stream(upstream_response)
        )
        return StreamingResponse(
            stream,
            status_code=upstream_response.status,
            headers=response_headers,
            media_type=upstream_content_type or SSE_CONTENT_TYPE,
        )

    try:
        response_body = upstream_response.read()
        status_code = upstream_response.status
        content_encoding = upstream_response.headers.get("content-encoding", "")
    finally:
        upstream_response.close()

    if "application/json" in upstream_content_type:
        usage = _extract_usage_from_response(response_body, content_encoding)
        if usage:
            await _accumulate_token_usage(usage)
        # Paired with the request translation: having sent an OpenAI body upstream, the
        # OpenAI body that comes back has to be turned into the Anthropic response the
        # client is waiting for. Only the non-streaming case; an SSE response is relayed
        # above and still needs its own transducer, which is why this whole path stays
        # behind a flag.
        if translated_request and status_code < 400:
            try:
                parsed_response = json.loads(
                    _decompress_for_parsing(response_body, content_encoding)
                )
            except (ValueError, UnicodeDecodeError):
                parsed_response = None
            if isinstance(parsed_response, dict) and "choices" in parsed_response:
                response_body = json.dumps(
                    _openai_response_to_anthropic(parsed_response), ensure_ascii=False
                ).encode("utf-8")
                # The body was re-encoded, so any length/encoding headers describing the
                # old one are now lies.
                response_headers = {
                    k: v for k, v in response_headers.items()
                    if k.lower() not in ("content-length", "content-encoding")
                }
                response_headers["content-type"] = "application/json"
    return Response(content=response_body, status_code=status_code, headers=response_headers)
