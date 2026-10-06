from __future__ import annotations

import contextlib
import importlib.util
import inspect
import io
import json
import os
import sys
import threading
import time
import traceback
from pathlib import Path
from types import ModuleType
from typing import Any, Callable
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from app import services as _services

COMPRESSOR_CANDIDATE_NAMES = (
    "compress_messages",
    "compress_payload",
    "process_request",
    "transform_payload",
)

app = FastAPI(title="SOMA Compression Service", version="0.3.0")


class TransformRequest(BaseModel):
    path: str = "/"
    query: str = ""
    request_id: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)
    # Read-only facts about the request that are not in the payload, e.g. the
    # protected task prompt ({"task": ...}). Offered to a compressor that
    # takes `metadata`; never forwarded upstream.
    context: dict[str, Any] = Field(default_factory=dict)
    # Services the proxy offers on this turn (soma-compressor-services/1). Absent: none.
    services: dict[str, Any] | None = None


class ResumeRequest(BaseModel):
    session: str
    results: list[dict[str, Any]] = Field(default_factory=list)


class TransformResponse(BaseModel):
    payload: dict[str, Any]


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


def _extract_messages(payload: dict[str, Any]) -> list[Any]:
    messages = payload.get("messages")
    if isinstance(messages, list):
        return messages
    return []


def _emit_message_event(*, request_id: str, stage: str, path: str, query: str, payload: dict[str, Any]) -> None:
    marker = f"[compression-service][messages.{stage}]"
    entry = {
        "request_id": request_id,
        "path": path,
        "query": query,
        "model": payload.get("model"),
        "messages": _extract_messages(payload),
    }
    print(f"{marker} {json.dumps(entry)}", flush=True)


_COMPRESSOR_EXEC_MARKER = "[compression-service][compressor.exec]"
# Serializes compressor invocations so per-invocation stdout/stderr capture does not
# interleave between concurrent /transform requests (handler runs in a threadpool).
_COMPRESSOR_EXEC_LOCK = threading.Lock()


def _get_compressor_output_capture_limit() -> int:
    raw = os.getenv("COMPRESSOR_EXEC_OUTPUT_CAPTURE_LIMIT_CHARS", "20000").strip()
    try:
        limit = int(raw)
    except ValueError:
        limit = 20000
    return max(0, limit)


def _truncate_captured_output(value: str) -> str:
    limit = _get_compressor_output_capture_limit()
    if len(value) <= limit:
        return value
    return f"{value[:limit]}\n... [truncated {len(value) - limit} chars]"


def _emit_compressor_exec_event(entry: dict[str, Any]) -> None:
    """Emit one JSON line per compressor invocation to container stdout.

    These lines are picked out of the collected container log by the sandbox
    service and uploaded to S3 as the run's compressor execution log.
    """
    print(f"{_COMPRESSOR_EXEC_MARKER} {json.dumps(entry)}", flush=True)


def _load_compressor_module() -> ModuleType | None:
    module_path_raw = os.getenv("MINER_MODULE_PATH", "").strip()
    if not module_path_raw:
        return None
    module_path = Path(module_path_raw).expanduser().resolve()
    if not module_path.is_file():
        return None

    spec = importlib.util.spec_from_file_location("soma_compressor_miner", str(module_path))
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    # Ensure decorators/type resolution that rely on sys.modules can find the module during import.
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(spec.name, None)
        raise
    return module

def _resolve_compressor_callable(module: ModuleType | None) -> Callable[..., Any] | None:
    if module is None:
        return None
    for name in COMPRESSOR_CANDIDATE_NAMES:
        candidate = getattr(module, name, None)
        if callable(candidate):
            return candidate
    return None


_COMPRESSOR_MODULE = _load_compressor_module()
_COMPRESSOR_FN = _resolve_compressor_callable(_COMPRESSOR_MODULE)


def _invoke_compressor(
    payload: dict[str, Any],
    *,
    path: str,
    context: dict[str, Any] | None = None,
    services: _services.Services | None = None,
) -> dict[str, Any]:
    if _COMPRESSOR_FN is None:
        return payload

    mutate = _coerce_bool(os.getenv("COMPRESSION_MUTATE_REQUEST", "true"), default=True)
    input_messages = _extract_messages(payload)
    try:
        signature = inspect.signature(_COMPRESSOR_FN)
    except (TypeError, ValueError):
        signature = None

    kwargs: dict[str, Any] = {}
    if signature is not None:
        parameters = signature.parameters
        if "messages" in parameters:
            kwargs["messages"] = input_messages
        if "path" in parameters:
            kwargs["path"] = path
        if "metadata" in parameters:
            kwargs["metadata"] = {**(context or {}), "path": path}
        if "services" in parameters:
            kwargs["services"] = services if services is not None else _services.Services({}, None)

    result: Any
    if kwargs:
        result = _COMPRESSOR_FN(**kwargs)
    else:
        result = _COMPRESSOR_FN(input_messages)

    if mutate:
        if isinstance(result, list):
            mutated_payload = dict(payload)
            mutated_payload["messages"] = result
            return mutated_payload
        if isinstance(result, dict):
            if isinstance(result.get("messages"), list):
                mutated_payload = dict(payload)
                mutated_payload["messages"] = result["messages"]
                return mutated_payload
            return result
    return payload


def _invoke_compressor_logged(
    payload: dict[str, Any],
    *,
    path: str,
    request_id: str,
    context: dict[str, Any] | None = None,
    services: _services.Services | None = None,
) -> dict[str, Any]:
    """Invoke the miner compressor and emit a per-invocation execution log event.

    Captures everything the miner module writes to stdout/stderr during the call,
    together with timing and failure details, and emits it as a single marked JSON
    line so the sandbox service can extract and persist it per run.
    """
    input_messages = _extract_messages(payload)
    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()
    started = time.monotonic()
    error: Exception | None = None
    error_traceback: str | None = None
    result: dict[str, Any] | None = None
    with _COMPRESSOR_EXEC_LOCK:
        try:
            with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(stderr_buffer):
                result = _invoke_compressor(payload, path=path, context=context, services=services)
        except Exception as exc:  # noqa: BLE001
            error = exc
            error_traceback = traceback.format_exc()
    duration_ms = round((time.monotonic() - started) * 1000.0, 3)

    _emit_compressor_exec_event(
        {
            "request_id": request_id,
            "path": path,
            "ok": error is None,
            "compressor_loaded": _COMPRESSOR_FN is not None,
            "duration_ms": duration_ms,
            "input_messages": len(input_messages),
            "output_messages": len(_extract_messages(result)) if isinstance(result, dict) else None,
            "stdout": _truncate_captured_output(stdout_buffer.getvalue()),
            "stderr": _truncate_captured_output(stderr_buffer.getvalue()),
            "error": f"{type(error).__name__}: {error}" if error is not None else None,
            "traceback": error_traceback,
        }
    )

    if error is not None:
        raise HTTPException(status_code=500, detail=f"compressor error: {error}") from error
    return result


@app.get("/health")
def health() -> JSONResponse:
    return JSONResponse(
        {
            "status": "ok",
            "compressor_loaded": _COMPRESSOR_FN is not None,
            "mutate_enabled": _coerce_bool(os.getenv("COMPRESSION_MUTATE_REQUEST", "true"), default=True),
            "mode": "transform-only",
        }
    )


def _miner_takes_services() -> bool:
    try:
        return _COMPRESSOR_FN is not None and "services" in inspect.signature(_COMPRESSOR_FN).parameters
    except (TypeError, ValueError):
        return False


_SESSIONS = _services.SessionStore()


def _session_answer(session: _services.Session, request: TransformRequest) -> JSONResponse:
    """Wait for the suspended miner's next move and turn it into the HTTP answer."""
    kind, value = session.next_event()
    if kind == "pending":
        return JSONResponse({"pending": value})
    _SESSIONS.drop(session.id)
    if kind == "error":
        if isinstance(value, HTTPException):
            raise value
        raise HTTPException(status_code=500, detail=f"compressor error: {value}")
    return JSONResponse({"payload": _finish_transform(request, value)})


def _finish_transform(request: TransformRequest, transformed: Any) -> dict[str, Any]:
    if not isinstance(transformed, dict):
        raise HTTPException(status_code=500, detail="compressor returned unsupported payload type")
    _emit_message_event(
        request_id=(request.request_id or "").strip() or "no-request-id",
        stage="out",
        path=request.path,
        query=request.query,
        payload=transformed,
    )
    return transformed


@app.post("/transform/resume")
def transform_resume(request: ResumeRequest) -> JSONResponse:
    session = _SESSIONS.get(request.session)
    if session is None:
        raise HTTPException(status_code=404, detail="unknown or expired session")
    try:
        session.deliver(request.results)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    # messages.out is emitted against the /transform request that started the session.
    return _session_answer(session, session.request)


@app.post("/transform")
def transform_payload(request: TransformRequest) -> JSONResponse:
    if not isinstance(request.payload, dict):
        raise HTTPException(status_code=400, detail="payload must be an object")

    request_id = (request.request_id or "").strip() or "no-request-id"
    _emit_message_event(
        request_id=request_id,
        stage="in",
        path=request.path,
        query=request.query,
        payload=request.payload,
    )

    offer = _services.parse_offer(request.services)
    if offer is None or not _miner_takes_services():
        transformed = _invoke_compressor_logged(
            request.payload,
            path=request.path,
            request_id=request_id,
            context=request.context,
        )
        return JSONResponse({"payload": _finish_transform(request, transformed)})

    # The miner may call services: run it in a worker so it can be suspended between
    # rounds, and answer this request with its first move (see app/services.py).
    session = _services.Session(offer)
    session.request = request
    _SESSIONS.add(session)

    def work() -> None:
        try:
            session.finish(_invoke_compressor_logged(
                request.payload,
                path=request.path,
                request_id=request_id,
                context=request.context,
                services=session.services,
            ))
        except BaseException as exc:  # noqa: BLE001 - delivered to the waiting handler
            session.finish(error=exc)

    threading.Thread(target=work, name=f"miner-{session.id[:8]}", daemon=True).start()
    return _session_answer(session, request)
