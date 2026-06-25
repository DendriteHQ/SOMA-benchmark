from __future__ import annotations

import importlib.util
import inspect
import json
import os
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Callable
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

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
    print(f"{marker} {json.dumps(entry, ensure_ascii=False)}", flush=True)


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


def _invoke_compressor(payload: dict[str, Any], *, path: str) -> dict[str, Any]:
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
            kwargs["metadata"] = {"path": path}

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


@app.post("/transform", response_model=TransformResponse)
def transform_payload(request: TransformRequest) -> TransformResponse:
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

    try:
        transformed = _invoke_compressor(request.payload, path=request.path)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"compressor error: {exc}") from exc

    if not isinstance(transformed, dict):
        raise HTTPException(status_code=500, detail="compressor returned unsupported payload type")

    _emit_message_event(
        request_id=request_id,
        stage="out",
        path=request.path,
        query=request.query,
        payload=transformed,
    )
    return TransformResponse(payload=transformed)
