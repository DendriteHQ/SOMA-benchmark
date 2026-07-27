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

from app.contracts import (
    CONTRACT_VERSION,
    ReplaceSpanWithArtifactEdit,
    TransformRequestContract,
    TransformResponseContract,
    normalize_edit_plan_output,
)
from app.miner_tools import (
    activate_request_tool_context,
    clear_request_tool_context,
    set_candidate_rewrite_artifacts,
)
from app.prompt_catalog import list_prompt_ids
from app.rewrite_cache import get_rewrite_artifact
from app.token_usage import add_usage_totals

COMPRESSOR_CANDIDATE_NAMES = (
    "plan_prompt_edits",
)

app = FastAPI(title="SOMA Compression Service", version="0.3.0")


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


def _emit_edit_plan_event(*, request_id: str, path: str, edit_plan: TransformResponseContract) -> None:
    marker = "[compression-service][edit-plan]"
    entry = {
        "request_id": request_id,
        "path": path,
        "contract_version": edit_plan.contract_version,
        "edit_count": len(edit_plan.edits),
        "artifacts": [
            {
                "artifact_key": artifact.artifact_key,
                "prompt_id": artifact.prompt_id,
                "text_length": len(artifact.text),
                "source_text_length": len(artifact.source_text),
                "cache_hit": artifact.cache_hit,
            }
            for artifact in edit_plan.artifacts
        ],
        "compression_usage": edit_plan.compression_usage.model_dump(mode="json"),
        "edits": [edit.model_dump(mode="json") for edit in edit_plan.edits],
    }
    print(f"{marker} {json.dumps(entry, ensure_ascii=False)}", flush=True)


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
    print(f"{_COMPRESSOR_EXEC_MARKER} {json.dumps(entry, ensure_ascii=False)}", flush=True)


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


def _invoke_compressor(payload: dict[str, Any], *, path: str, query: str, request_id: str) -> TransformResponseContract:
    if _COMPRESSOR_FN is None:
        return TransformResponseContract(contract_version=CONTRACT_VERSION, edits=[])

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
        if "payload" in parameters:
            kwargs["payload"] = payload
        if "path" in parameters:
            kwargs["path"] = path
        if "query" in parameters:
            kwargs["query"] = query
        if "request_id" in parameters:
            kwargs["request_id"] = request_id
        metadata_enabled = "metadata" in parameters
    else:
        metadata_enabled = False

    result: Any
    tool_context = activate_request_tool_context(
        request_id=request_id,
        path=path,
        query=query,
        payload=payload,
    )
    try:
        set_candidate_rewrite_artifacts(
            payload=payload,
            limit=int(os.getenv("COMPRESSION_ARTIFACT_CANDIDATE_LIMIT", "40").strip() or "40"),
        )
        if metadata_enabled:
            kwargs["metadata"] = {
                "path": path,
                "query": query,
                "request_id": request_id,
                "contract_version": CONTRACT_VERSION,
                "artifacts": [
                    {
                        "artifact_key": artifact.artifact_key,
                        "prompt_id": artifact.prompt_id,
                        "request_url": artifact.request_url,
                        "model": artifact.model,
                        "source_text": artifact.source_text,
                        "rewritten_text": artifact.rewritten_text,
                        "cache_hit": artifact.cache_hit,
                    }
                    for artifact in tool_context.candidate_artifacts
                ],
            }
        if kwargs:
            result = _COMPRESSOR_FN(**kwargs)
        else:
            result = _COMPRESSOR_FN()
    finally:
        clear_request_tool_context()

    normalized = normalize_edit_plan_output(result)
    normalized.artifacts.extend(
        [
            {
                "artifact_key": entry.artifact_key,
                "source_text": entry.source_text,
                "text": entry.rewritten_text,
                "prompt_id": entry.prompt_id,
                "request_url": entry.request_url,
                "model": entry.model,
                "cache_hit": entry.cache_hit,
            }
            for entry in tool_context.rewrite_history
        ]
    )
    known_keys = {artifact.artifact_key for artifact in normalized.artifacts}
    for edit in normalized.edits:
        if not isinstance(edit, ReplaceSpanWithArtifactEdit):
            continue
        if edit.artifact_key in known_keys:
            continue
        artifact = get_rewrite_artifact(edit.artifact_key)
        if artifact is None:
            raise RuntimeError(f"unknown artifact_key referenced by edit plan: {edit.artifact_key}")
        normalized.artifacts.append(
            {
                "artifact_key": artifact.artifact_key,
                "source_text": artifact.source_text,
                "text": artifact.rewritten_text,
                "prompt_id": artifact.prompt_id,
                "request_url": artifact.request_url,
                "model": artifact.model,
                "cache_hit": True,
            }
        )
        known_keys.add(artifact.artifact_key)
    normalized.compression_usage = add_usage_totals(
        normalized.compression_usage.model_dump(mode="json"),
        tool_context.usage_totals,
    )
    return TransformResponseContract.model_validate(normalized.model_dump(mode="json"))


def _invoke_compressor_logged(
    payload: dict[str, Any],
    *,
    path: str,
    query: str,
    request_id: str,
) -> TransformResponseContract:
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
    result: TransformResponseContract | None = None
    with _COMPRESSOR_EXEC_LOCK:
        try:
            with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(stderr_buffer):
                result = _invoke_compressor(payload, path=path, query=query, request_id=request_id)
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
            "edit_count": len(result.edits) if isinstance(result, TransformResponseContract) else None,
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
            "contract_version": CONTRACT_VERSION,
            "mode": "edit-plan",
            "available_prompts": list_prompt_ids(),
        }
    )


@app.post("/transform", response_model=TransformResponseContract)
def transform_payload(request: TransformRequestContract) -> TransformResponseContract:
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

    transformed = _invoke_compressor_logged(
        request.payload,
        path=request.path,
        query=request.query,
        request_id=request_id,
    )
    _emit_edit_plan_event(request_id=request_id, path=request.path, edit_plan=transformed)
    return transformed
