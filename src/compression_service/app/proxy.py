from __future__ import annotations

import asyncio
import gzip
import json
import os
import uuid
import zlib
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit, urlunsplit
from urllib.request import Request as UrlRequest
from urllib.request import urlopen

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response

from app.contracts import TransformResponseContract, apply_edit_plan
from app.token_usage import add_usage_totals, empty_usage_totals, normalize_provider_usage

DEFAULT_UPSTREAM_TIMEOUT_SECONDS = 120.0
DEFAULT_COMPRESSION_TIMEOUT_SECONDS = 30.0
DEFAULT_COMPRESSION_BASE_URL = "http://compression-service:8000/"
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
_token_totals: dict[str, int] = empty_usage_totals()

TOKEN_USAGE_LOG_MARKER = "[proxy][token-usage] "


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
    normalized = normalize_provider_usage(usage)
    await _accumulate_usage_totals(normalized)


async def _accumulate_usage_totals(usage: dict[str, int]) -> None:
    async with _token_lock:
        snapshot = add_usage_totals(_token_totals, usage)
        _token_totals.update(snapshot)
    print(f"{TOKEN_USAGE_LOG_MARKER}{json.dumps(snapshot)}", flush=True)


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


def _transform_payload_via_compression_service(
    *,
    path: str,
    query: str,
    payload: dict[str, Any],
    request_id: str,
    compression_base_url: str,
) -> tuple[dict[str, Any], dict[str, int]]:
    transform_url = _build_upstream_url(base_url=compression_base_url, path="transform", query="")
    request_body = json.dumps(
        {
            "path": f"/{path}" if path else "/",
            "query": query,
            "payload": payload,
            "request_id": request_id,
        },
        ensure_ascii=False,
    ).encode("utf-8")
    compression_request = UrlRequest(
        transform_url,
        data=request_body,
        headers={"content-type": "application/json"},
        method="POST",
    )
    with urlopen(compression_request, timeout=_resolve_compression_timeout_seconds()) as response:
        raw = response.read()
    parsed = json.loads(raw)
    response_contract = TransformResponseContract.model_validate(parsed)
    return apply_edit_plan(payload, response_contract), response_contract.compression_usage.model_dump(mode="json")


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
    override_api_key = _resolve_provider_api_key_override()
    run_id_for_routing = _resolve_run_id_for_routing(request, override_api_key=override_api_key)
    compression_base_url = _resolve_compression_base_url_for_run(run_id=run_id_for_routing)
    if _resolve_compression_enabled() and body and "application/json" in content_type.lower():
        try:
            parsed_payload = json.loads(body)
        except json.JSONDecodeError:
            parsed_payload = None

        if isinstance(parsed_payload, dict):
            stripped_payload, protected_prompts = _strip_protected_prompts(parsed_payload)
            transformed_payload, compression_usage = _transform_payload_via_compression_service(
                path=path,
                query=request.url.query,
                payload=stripped_payload,
                request_id=request_id,
                compression_base_url=compression_base_url,
            )
            await _accumulate_usage_totals(compression_usage)
            transformed_payload = _restore_protected_prompts(transformed_payload, protected_prompts)
            forwarded_body = json.dumps(transformed_payload, ensure_ascii=False).encode("utf-8")

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
        with urlopen(upstream_request, timeout=_resolve_upstream_timeout_seconds()) as upstream_response:
            response_body = upstream_response.read()
            response_headers = _response_headers_from_upstream(list(upstream_response.headers.items()))
            content_type = upstream_response.headers.get("content-type", "")
            if "application/json" in content_type:
                content_encoding = upstream_response.headers.get("content-encoding", "")
                usage = _extract_usage_from_response(response_body, content_encoding)
                if usage:
                    await _accumulate_token_usage(usage)
            return Response(
                content=response_body,
                status_code=upstream_response.status,
                headers=response_headers,
            )
    except HTTPError as exc:
        response_body = exc.read()
        response_headers = _response_headers_from_upstream(list(exc.headers.items()))
        return Response(content=response_body, status_code=exc.code, headers=response_headers)
    except URLError as exc:
        raise HTTPException(status_code=502, detail=f"Upstream request failed: {exc}") from exc
