from __future__ import annotations

import asyncio
import fnmatch
import gzip
import json
import os
import time
import uuid
import zlib
from concurrent.futures import ThreadPoolExecutor
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit, urlunsplit
from urllib.request import Request as UrlRequest
from urllib.request import urlopen

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response

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
    "models": ["jev-latest", "jev-1.13", "typesafe/*", "~typesafe/*"],
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
        raw_input = usage["input_tokens"]
        cache_read = usage.get("cache_read_input_tokens", 0)
        cache_write = usage.get("cache_creation_input_tokens", 0)
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
        if not isinstance(body["model"], str) or not any(fnmatch.fnmatchcase(body["model"], m) for m in offer["models"]):
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
        print(f"{SERVICE_CALL_LOG_MARKER}{json.dumps(entry, ensure_ascii=False)}", flush=True)
    if runnable:
        print(f"{SERVICE_USAGE_LOG_MARKER}{json.dumps(_service_totals)}", flush=True)
    for cid, result in results.items():
        if not result.get("ok") and cid not in dict(runnable):
            print(f"{SERVICE_CALL_LOG_MARKER}{json.dumps({'request_id': request_id, 'session': pending.get('session'), 'round': pending.get('round'), 'id': cid, 'ok': False, 'error': result['error']}, ensure_ascii=False)}", flush=True)
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
