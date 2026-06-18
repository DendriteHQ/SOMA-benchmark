from __future__ import annotations

import importlib.util
import inspect
import json
import os
import uuid
from pathlib import Path
from types import ModuleType
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit, urlunsplit
from urllib.request import Request as UrlRequest
from urllib.request import urlopen

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response

DEFAULT_UPSTREAM_BASE_URL = "http://proxy:8080/"
DEFAULT_TIMEOUT_SECONDS = 120.0
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

COMPRESSOR_CANDIDATE_NAMES = (
	"compress_messages",
	"compress_payload",
	"process_request",
	"transform_payload",
)

app = FastAPI(title="SOMA Compression Service", version="0.2.0")


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


def _resolve_timeout_seconds() -> float:
	raw = os.getenv("COMPRESSION_UPSTREAM_TIMEOUT_SECONDS", "").strip()
	if not raw:
		return DEFAULT_TIMEOUT_SECONDS
	try:
		value = float(raw)
	except ValueError:
		return DEFAULT_TIMEOUT_SECONDS
	return value if value > 0 else DEFAULT_TIMEOUT_SECONDS


def _normalize_base_url(raw_url: str) -> str:
	parsed = urlsplit(raw_url)
	if not parsed.scheme or not parsed.netloc:
		raise RuntimeError(
			"Compression upstream URL must be absolute http(s). "
			f"Received: {raw_url!r}"
		)
	path = parsed.path or "/"
	if not path.endswith("/"):
		path = f"{path}/"
	return urlunsplit((parsed.scheme, parsed.netloc, path, "", ""))


def _resolve_upstream_base_url() -> str:
	for env_name in (
		"COMPRESSION_UPSTREAM_BASE_URL",
		"SOMA_COMPRESSION_SERVICE_UPSTREAM_BASE_URL",
		"COMPACT_BENCH_LLM_BASE_URL",
	):
		candidate = os.getenv(env_name, "").strip()
		if candidate:
			return _normalize_base_url(candidate)
	return _normalize_base_url(DEFAULT_UPSTREAM_BASE_URL)


def _build_upstream_url(*, base_url: str, path: str, query: str) -> str:
	base = urlsplit(base_url)
	base_path = base.path if base.path else "/"
	if not base_path.endswith("/"):
		base_path = f"{base_path}/"
	normalized_path = path.lstrip("/")
	full_path = f"{base_path}{normalized_path}" if normalized_path else base_path
	return urlunsplit((base.scheme, base.netloc, full_path, query, ""))


def _extract_message_preview(body: bytes, content_type: str) -> str:
	if not body or "application/json" not in content_type.lower():
		return ""
	try:
		payload = json.loads(body)
	except json.JSONDecodeError:
		return ""

	if not isinstance(payload, dict):
		return ""
	messages = payload.get("messages")
	if not isinstance(messages, list) or not messages:
		return ""

	last = messages[-1]
	if not isinstance(last, dict):
		return ""

	content = last.get("content")
	if isinstance(content, str):
		return content.strip()[:500]
	if isinstance(content, list):
		parts: list[str] = []
		for item in content:
			if isinstance(item, dict) and isinstance(item.get("text"), str):
				parts.append(item["text"].strip())
		return " ".join(part for part in parts if part)[:500]
	return ""


def _extract_messages(payload: dict[str, Any]) -> list[Any]:
	messages = payload.get("messages")
	if isinstance(messages, list):
		return messages
	return []


def _emit_message_event(
	*,
	request_id: str,
	stage: str,
	method: str,
	path: str,
	query: str,
	payload: dict[str, Any],
) -> None:
	marker = f"[compression-service][messages.{stage}]"
	entry = {
		"request_id": request_id,
		"method": method,
		"path": path,
		"query": query,
		"model": payload.get("model"),
		"messages": _extract_messages(payload),
	}
	print(f"{marker} {json.dumps(entry, ensure_ascii=False)}", flush=True)


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
	spec.loader.exec_module(module)
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

	mutate = _coerce_bool(os.getenv("COMPRESSION_MUTATE_REQUEST", "false"), default=False)
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
			"upstream": _resolve_upstream_base_url(),
			"compressor_loaded": _COMPRESSOR_FN is not None,
			"mutate_enabled": _coerce_bool(os.getenv("COMPRESSION_MUTATE_REQUEST", "false"), default=False),
		}
	)


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"])
async def proxy_passthrough(path: str, request: Request) -> Response:
	upstream_base_url = _resolve_upstream_base_url()
	upstream_url = _build_upstream_url(base_url=upstream_base_url, path=path, query=request.url.query)

	method = request.method.upper()
	request_id = uuid.uuid4().hex
	body = await request.body()
	content_type = request.headers.get("content-type", "")
	preview = _extract_message_preview(body, content_type)
	if preview:
		print(f"[compression-service] message preview: {preview}", flush=True)

	forwarded_body = body
	upstream_payload_for_logging: dict[str, Any] | None = None
	if body and "application/json" in content_type.lower():
		try:
			parsed_payload = json.loads(body)
		except json.JSONDecodeError:
			parsed_payload = None

		if isinstance(parsed_payload, dict):
			_emit_message_event(
				request_id=request_id,
				stage="in",
				method=method,
				path=f"/{path}" if path else "/",
				query=request.url.query,
				payload=parsed_payload,
			)
			try:
				transformed = _invoke_compressor(parsed_payload, path=path)
				if isinstance(transformed, dict):
					upstream_payload_for_logging = transformed
				else:
					upstream_payload_for_logging = parsed_payload
				# By default no mutation is applied; encode only if payload changed.
				if transformed is not parsed_payload:
					forwarded_body = json.dumps(transformed, ensure_ascii=False).encode("utf-8")
			except Exception as exc:  # noqa: BLE001
				print(f"[compression-service] compressor error: {exc}", flush=True)
				upstream_payload_for_logging = parsed_payload

	if upstream_payload_for_logging is not None:
		_emit_message_event(
			request_id=request_id,
			stage="out",
			method=method,
			path=f"/{path}" if path else "/",
			query=request.url.query,
			payload=upstream_payload_for_logging,
		)

	upstream_request = UrlRequest(
		upstream_url,
		data=forwarded_body if method in {"POST", "PUT", "PATCH", "DELETE"} else None,
		headers=_forwardable_headers(request),
		method=method,
	)

	try:
		with urlopen(upstream_request, timeout=_resolve_timeout_seconds()) as upstream_response:
			response_body = upstream_response.read()
			response_headers = _response_headers_from_upstream(list(upstream_response.headers.items()))
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
