from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any
from urllib import error, parse, request

from .cache_paths import benchmark_cache_root
from .config import DEFAULTS


class BenchmarkRunnerSettingsError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class RunnerAgentConfig:
    agent_name: str
    runtime_backend: str


@dataclass(frozen=True, slots=True)
class RunnerBenchmarkConfig:
    benchmark_name: str
    dataset_config: str
    split: str
    runtime_setup_entry: dict[str, Any]
    output_dir: Path
    workspace: str
    max_iterations: int
    concurrency: int
    execute: bool


@dataclass(frozen=True, slots=True)
class ResolvedRunnerConfig:
    agent: RunnerAgentConfig
    benchmark: RunnerBenchmarkConfig
    llm_config: dict[str, Any]


def resolve_runner_config(
    *,
    repo_root: Path,
    agent_name: str,
    model: str | None,
    benchmark_name: str,
    selection_id: str,
    ignore_api_key: bool = False,
) -> ResolvedRunnerConfig:
    load_repo_env(repo_root)
    agent_config = RunnerAgentConfig(
        agent_name=agent_name,
        runtime_backend=agent_name,
    )

    dataset_config, split, runtime_setup_entry = resolve_benchmark_runtime_setup(
        benchmark_name=benchmark_name,
        selection_id=selection_id,
    )

    output_root = Path(_first_non_empty(os.getenv("SOMA_BENCHMARK_OUTPUT_DIR"), DEFAULTS.output_dir))
    output_dir = output_root / _slug(benchmark_name) / _slug(selection_id)

    benchmark_config = RunnerBenchmarkConfig(
        benchmark_name=benchmark_name,
        dataset_config=dataset_config,
        split=split,
        runtime_setup_entry=runtime_setup_entry,
        output_dir=output_dir,
        workspace=_first_non_empty(os.getenv("SOMA_BENCHMARK_WORKSPACE"), DEFAULTS.workspace),
        max_iterations=_coerce_positive_int(os.getenv("SOMA_BENCHMARK_MAX_ITERATIONS"), DEFAULTS.max_iterations),
        concurrency=resolve_benchmark_concurrency(),
        execute=_coerce_bool(os.getenv("SOMA_BENCHMARK_EXECUTE"), False),
    )

    return ResolvedRunnerConfig(
        agent=agent_config,
        benchmark=benchmark_config,
        llm_config=build_llm_config(model, ignore_api_key=ignore_api_key),
    )


def load_repo_env(repo_root: Path) -> None:
    env_path = repo_root / ".env"
    if not env_path.is_file():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key:
            continue
        parsed_value = value.strip()
        if len(parsed_value) >= 2 and parsed_value[0] == parsed_value[-1] and parsed_value[0] in {'"', "'"}:
            parsed_value = parsed_value[1:-1]
        os.environ.setdefault(key, parsed_value)


def build_llm_config(model: str | None, *, ignore_api_key: bool = False) -> dict[str, Any]:
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
    api_key = _first_non_empty(
        os.getenv("LLM_API_KEY"),
        os.getenv("OPENAI_API_KEY"),
        openrouter_api_key,
    )
    if ignore_api_key:
        api_key = ""
    elif not api_key:
        raise BenchmarkRunnerSettingsError(
            "LLM API key is not configured. Set LLM_API_KEY, OPENAI_API_KEY, or OPENROUTER_API_KEY in .env."
        )

    base_url = _first_non_empty(
        os.getenv("LLM_BASE_URL"),
        os.getenv("OPENAI_BASE_URL"),
        os.getenv("OPENROUTER_BASE_URL"),
    )
    is_openrouter_config = (
        "openrouter.ai" in (base_url or "").lower()
        or bool(openrouter_api_key and api_key == openrouter_api_key and not os.getenv("LLM_API_KEY") and not os.getenv("OPENAI_API_KEY"))
    )
    normalized_model = _resolve_llm_model(model)
    if is_openrouter_config:
        normalized_model = _normalize_openrouter_model(normalized_model)

    payload = {
        "model": normalized_model,
    }
    if api_key:
        payload["api_key"] = api_key
    if base_url:
        payload["base_url"] = base_url

    api_version = _first_non_empty(os.getenv("LLM_API_VERSION"))
    if api_version:
        payload["api_version"] = api_version
    return payload


def resolve_benchmark_concurrency(explicit_value: int | None = None) -> int:
    if explicit_value is not None:
        if explicit_value <= 0:
            raise BenchmarkRunnerSettingsError("Benchmark concurrency must be a positive integer.")
        return explicit_value
    return _coerce_positive_int(os.getenv("SOMA_BENCHMARK_CONCURRENCY"), DEFAULTS.concurrency)


def _resolve_llm_model(model: str | None) -> str:
    normalized_model = _first_non_empty(
        model,
        os.getenv("LLM_MODEL"),
        os.getenv("OPENAI_MODEL"),
        os.getenv("OPENROUTER_MODEL"),
    ).strip()
    if not normalized_model:
        raise BenchmarkRunnerSettingsError(
            "LLM model is not configured. Set LLM_MODEL, OPENAI_MODEL, or OPENROUTER_MODEL in .env, or pass --model."
        )
    return normalized_model


def _normalize_openrouter_model(model: str) -> str:
    normalized_model = model.strip()
    if not normalized_model:
        return normalized_model

    if normalized_model.startswith("openrouter/"):
        remainder = normalized_model[len("openrouter/") :]
        if remainder in {"auto", "free", "bodybuilder"}:
            return normalized_model
        if "/" in remainder:
            return remainder
        normalized_model = remainder

    if "/" not in normalized_model and normalized_model.startswith(("gpt-", "o1", "o3", "o4")):
        return f"openai/{normalized_model}"

    return normalized_model


def resolve_benchmark_runtime_setup(
    *,
    benchmark_name: str,
    selection_id: str,
) -> tuple[str, str, dict[str, Any]]:
    dataset_info = _load_cached_dataset_info(benchmark_name)
    if dataset_info is None:
        dataset_info = _load_dataset_info_from_local_hf_cache(benchmark_name)
        if dataset_info is None:
            dataset_info = _request_dataset_viewer_json("info", {"dataset": benchmark_name})
        _write_cached_dataset_info(benchmark_name, dataset_info)
    dataset_configs = dataset_info.get("dataset_info")
    if not isinstance(dataset_configs, dict) or not dataset_configs:
        raise BenchmarkRunnerSettingsError(f"Could not resolve dataset info for benchmark: {benchmark_name}")

    dataset_config = _first_non_empty(os.getenv("SOMA_BENCHMARK_CONFIG"))
    if not dataset_config:
        dataset_config = "default" if "default" in dataset_configs else next(iter(dataset_configs))
    config_payload = dataset_configs.get(dataset_config)
    if not isinstance(config_payload, dict):
        available_configs = ", ".join(sorted(dataset_configs))
        raise BenchmarkRunnerSettingsError(
            f"Benchmark config '{dataset_config}' was not found for {benchmark_name}. Available configs: {available_configs}"
        )

    split = _first_non_empty(os.getenv("SOMA_BENCHMARK_SPLIT"))
    split_payload = config_payload.get("splits") if isinstance(config_payload.get("splits"), dict) else {}
    if not split:
        split = "test" if "test" in split_payload else next(iter(split_payload), "")
    if not split:
        raise BenchmarkRunnerSettingsError(f"No split metadata available for benchmark: {benchmark_name}")

    features = config_payload.get("features") if isinstance(config_payload.get("features"), dict) else {}
    lookup_field = "instance_id"
    if lookup_field not in features:
        raise BenchmarkRunnerSettingsError(
            f"Benchmark dataset {benchmark_name} does not expose an '{lookup_field}' field needed by the runner"
        )
    where = f'"{lookup_field}"={_sql_string_literal(selection_id)}'
    split_row_count = _extract_split_row_count(split_payload.get(split))

    # Resolving a single instance only needs the one targeted lookup below.
    # Prefetching the whole split costs one datasets-server request per 100 rows,
    # which trips the viewer rate limit on large datasets (SWE-rebench-V2 has
    # ~32k rows -> ~321 requests -> HTTP 429). Opt in explicitly when the full
    # offline split cache is actually wanted.
    if _coerce_bool(os.getenv("SOMA_BENCHMARK_PREFETCH_SPLIT"), False):
        _ensure_cached_dataset_split(
            benchmark_name=benchmark_name,
            dataset_config=dataset_config,
            split=split,
            split_row_count=split_row_count,
        )

    row_wrapper = _resolve_dataset_row(
        benchmark_name=benchmark_name,
        dataset_config=dataset_config,
        split=split,
        lookup_field=lookup_field,
        selection_id=selection_id,
        split_row_count=split_row_count,
        filter_where=where,
    )

    # Persist just the resolved row into the split cache so consumers that scan
    # rows.jsonl (e.g. the SWE-rebench eval harness) find it without a full
    # prefetch. Idempotent; no-op when the row is already cached.
    _cache_resolved_dataset_row(
        benchmark_name=benchmark_name,
        dataset_config=dataset_config,
        split=split,
        row_wrapper=row_wrapper,
    )

    return dataset_config, split, normalize_runtime_setup_entry(row_wrapper["row"], benchmark_name=benchmark_name)


def _resolve_dataset_row(
    *,
    benchmark_name: str,
    dataset_config: str,
    split: str,
    lookup_field: str,
    selection_id: str,
    split_row_count: int,
    filter_where: str,
) -> dict[str, Any]:
    cached_row_wrapper = _resolve_cached_dataset_row(
        benchmark_name=benchmark_name,
        dataset_config=dataset_config,
        split=split,
        lookup_field=lookup_field,
        selection_id=selection_id,
    )
    if cached_row_wrapper is not None:
        return cached_row_wrapper

    try:
        response = _request_dataset_viewer_json(
            "filter",
            {
                "dataset": benchmark_name,
                "config": dataset_config,
                "split": split,
                "where": filter_where,
                "offset": 0,
                "length": 1,
            },
        )
        rows = response.get("rows")
        if isinstance(rows, list) and rows:
            row_wrapper = rows[0]
            if isinstance(row_wrapper, dict) and isinstance(row_wrapper.get("row"), dict):
                return row_wrapper
    except BenchmarkRunnerSettingsError as exc:
        if "dataset index is loading" not in str(exc).lower():
            raise

    return _scan_dataset_rows_for_selection(
        benchmark_name=benchmark_name,
        dataset_config=dataset_config,
        split=split,
        lookup_field=lookup_field,
        selection_id=selection_id,
        split_row_count=split_row_count,
    )


def _scan_dataset_rows_for_selection(
    *,
    benchmark_name: str,
    dataset_config: str,
    split: str,
    lookup_field: str,
    selection_id: str,
    split_row_count: int,
) -> dict[str, Any]:
    cached_row_wrapper = _resolve_cached_dataset_row(
        benchmark_name=benchmark_name,
        dataset_config=dataset_config,
        split=split,
        lookup_field=lookup_field,
        selection_id=selection_id,
    )
    if cached_row_wrapper is not None:
        return cached_row_wrapper

    page_size = min(_coerce_positive_int(os.getenv("SOMA_BENCHMARK_ROW_SCAN_PAGE_SIZE"), 100), 100)
    if split_row_count <= 0:
        raise BenchmarkRunnerSettingsError(
            f"No split row count metadata available to scan benchmark rows for {benchmark_name}:{dataset_config}/{split}"
        )

    for offset in range(0, split_row_count, page_size):
        response = _request_dataset_viewer_json(
            "rows",
            {
                "dataset": benchmark_name,
                "config": dataset_config,
                "split": split,
                "offset": offset,
                "length": page_size,
            },
        )
        rows = response.get("rows")
        if not isinstance(rows, list):
            continue
        for row_wrapper in rows:
            if not isinstance(row_wrapper, dict):
                continue
            row = row_wrapper.get("row")
            if not isinstance(row, dict):
                continue
            candidate = row.get(lookup_field)
            if not isinstance(candidate, str) and lookup_field != "instance_id":
                candidate = row.get("instance_id")
            if isinstance(candidate, str) and candidate.strip() == selection_id:
                return row_wrapper

    raise BenchmarkRunnerSettingsError(
        f"No benchmark rows matched {lookup_field}={selection_id!r} in {benchmark_name}:{dataset_config}/{split}"
    )


def _extract_split_row_count(split_payload: Any) -> int:
    if isinstance(split_payload, dict):
        return _coerce_positive_int(split_payload.get("num_examples"), 1)
    return 0


def normalize_runtime_setup_entry(row: dict[str, Any], *, benchmark_name: str) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "benchmark_id": _optional_string(row.get("benchmark_id")),
        "task_id": _optional_string(row.get("task_id")),
        "issue_id": _optional_string(row.get("issue_id")),
        "instance_id": _first_non_empty(row.get("instance_id")),
        "repo": _first_non_empty(row.get("repo")),
        "base_commit": _first_non_empty(row.get("base_commit")),
        "environment_setup_commit": _first_non_empty(row.get("environment_setup_commit")),
        "install_config": _normalize_mapping(row.get("install_config")),
        "requirements": _normalize_sequence(row.get("requirements")),
        "environment": _normalize_mapping(row.get("environment")),
        "fail_to_pass": _normalize_sequence(row.get("FAIL_TO_PASS")),
        "pass_to_pass": _normalize_sequence(row.get("PASS_TO_PASS")),
    }
    hidden_eval = {
        "benchmark": benchmark_name,
    }
    for key in (
        "patch",
        "test_patch",
        "problem_statement",
        "hints_text",
        "created_at",
        "version",
        "FAIL_TO_FAIL",
        "PASS_TO_FAIL",
        "difficulty",
        "meta",
        "docker_image",
        "image_name",
        "license_name",
        # SWE-rebench-V2 content fields (absent in V1); kept verbatim so V2 rows
        # are fully represented instead of silently dropped.
        "pr_description",
        "interface",
        "language",
        "ground_truth",
    ):
        value = row.get(key)
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        hidden_eval[key] = value
    entry["hidden_eval"] = hidden_eval
    return entry


def _request_dataset_viewer_json(endpoint: str, params: dict[str, Any]) -> dict[str, Any]:
    query = parse.urlencode({key: value for key, value in params.items() if value is not None})
    url = f"https://datasets-server.huggingface.co/{endpoint}?{query}"
    headers = {
        "Accept": "application/json",
    }
    timeout_seconds = _coerce_positive_int(os.getenv("SOMA_BENCHMARK_HF_TIMEOUT_SECONDS"), 60)
    max_attempts = _coerce_positive_int(os.getenv("SOMA_BENCHMARK_HF_MAX_ATTEMPTS"), 3)
    hf_token = _first_non_empty(os.getenv("HF_TOKEN"), os.getenv("HUGGINGFACE_TOKEN"))
    if hf_token:
        headers["Authorization"] = f"Bearer {hf_token}"

    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            with request.urlopen(request.Request(url, headers=headers), timeout=timeout_seconds) as response:
                payload = json.load(response)
            break
        except error.HTTPError as exc:
            details = exc.read().decode("utf-8", errors="replace")
            is_loading_error = exc.code >= 500 and "dataset index is loading" in details.lower()
            if is_loading_error and attempt < max_attempts:
                time.sleep(min(attempt, 3))
                last_error = exc
                continue
            raise BenchmarkRunnerSettingsError(
                f"Hugging Face datasets server request failed with HTTP {exc.code}: {details}"
            ) from exc
        except TimeoutError as exc:
            if attempt < max_attempts:
                time.sleep(min(attempt, 3))
                last_error = exc
                continue
            raise BenchmarkRunnerSettingsError(
                f"Hugging Face datasets server request timed out after {timeout_seconds} seconds"
            ) from exc
        except error.URLError as exc:
            if attempt < max_attempts:
                time.sleep(min(attempt, 3))
                last_error = exc
                continue
            raise BenchmarkRunnerSettingsError(
                f"Hugging Face datasets server request failed: {exc.reason}"
            ) from exc
        except json.JSONDecodeError as exc:
            raise BenchmarkRunnerSettingsError(
                f"Invalid JSON returned by the Hugging Face datasets server: {exc.msg}"
            ) from exc
    else:
        raise BenchmarkRunnerSettingsError(
            f"Hugging Face datasets server request failed after {max_attempts} attempts: {last_error}"
        )

    if not isinstance(payload, dict):
        raise BenchmarkRunnerSettingsError("Hugging Face datasets server response must be a JSON object")
    return payload


def _benchmark_cache_root() -> Path:
    return benchmark_cache_root()


def _cache_slug(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-._")
    return normalized or "default"


def _dataset_cache_dir(benchmark_name: str) -> Path:
    return _benchmark_cache_root() / _cache_slug(benchmark_name)


def _dataset_info_cache_path(benchmark_name: str) -> Path:
    return _dataset_cache_dir(benchmark_name) / "dataset-info.json"


def _dataset_split_cache_dir(benchmark_name: str, dataset_config: str, split: str) -> Path:
    return _dataset_cache_dir(benchmark_name) / "splits" / _cache_slug(dataset_config) / _cache_slug(split)


def _dataset_split_rows_cache_path(benchmark_name: str, dataset_config: str, split: str) -> Path:
    return _dataset_split_cache_dir(benchmark_name, dataset_config, split) / "rows.jsonl"


def _dataset_split_meta_cache_path(benchmark_name: str, dataset_config: str, split: str) -> Path:
    return _dataset_split_cache_dir(benchmark_name, dataset_config, split) / "meta.json"


def _load_cached_dataset_info(benchmark_name: str) -> dict[str, Any] | None:
    if _cache_refresh_enabled():
        return None
    cache_path = _dataset_info_cache_path(benchmark_name)
    return _read_cached_json_object(cache_path)


def _write_cached_dataset_info(benchmark_name: str, payload: dict[str, Any]) -> None:
    cache_path = _dataset_info_cache_path(benchmark_name)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json_object_atomic(cache_path, payload)


def _write_json_object_atomic(path: Path, payload: dict[str, Any]) -> None:
    temp_path = path.with_suffix(f"{path.suffix}.{os.getpid()}.tmp")
    temp_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temp_path.replace(path)


def _cache_refresh_enabled() -> bool:
    return _coerce_bool(os.getenv("SOMA_BENCHMARK_REFRESH_CACHE"), False)


def _read_cached_json_object(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _is_dataset_split_cache_ready(
    *,
    benchmark_name: str,
    dataset_config: str,
    split: str,
    split_row_count: int,
) -> bool:
    if _cache_refresh_enabled():
        return False
    meta = _read_cached_json_object(_dataset_split_meta_cache_path(benchmark_name, dataset_config, split))
    rows_path = _dataset_split_rows_cache_path(benchmark_name, dataset_config, split)
    if meta is None or not rows_path.is_file():
        return False
    if meta.get("complete") is not True:
        return False
    cached_count = meta.get("row_count")
    if not isinstance(cached_count, int) or cached_count < 0:
        return False
    if split_row_count > 0 and cached_count != split_row_count:
        return False
    return True


def _ensure_cached_dataset_split(
    *,
    benchmark_name: str,
    dataset_config: str,
    split: str,
    split_row_count: int,
) -> None:
    if _is_dataset_split_cache_ready(
        benchmark_name=benchmark_name,
        dataset_config=dataset_config,
        split=split,
        split_row_count=split_row_count,
    ):
        return

    if split_row_count <= 0:
        raise BenchmarkRunnerSettingsError(
            f"No split row count metadata available to cache benchmark rows for {benchmark_name}:{dataset_config}/{split}"
        )

    page_size = min(_coerce_positive_int(os.getenv("SOMA_BENCHMARK_ROW_SCAN_PAGE_SIZE"), 100), 100)
    cache_dir = _dataset_split_cache_dir(benchmark_name, dataset_config, split)
    rows_path = _dataset_split_rows_cache_path(benchmark_name, dataset_config, split)
    meta_path = _dataset_split_meta_cache_path(benchmark_name, dataset_config, split)
    temp_rows_path = rows_path.with_suffix(f".jsonl.{os.getpid()}.tmp")
    cache_dir.mkdir(parents=True, exist_ok=True)

    try:
        row_count = 0
        with temp_rows_path.open("w", encoding="utf-8") as handle:
            for offset in range(0, split_row_count, page_size):
                response = _request_dataset_viewer_json(
                    "rows",
                    {
                        "dataset": benchmark_name,
                        "config": dataset_config,
                        "split": split,
                        "offset": offset,
                        "length": min(page_size, split_row_count - offset),
                    },
                )
                rows = response.get("rows")
                if not isinstance(rows, list):
                    raise BenchmarkRunnerSettingsError(
                        f"Unexpected rows payload while caching {benchmark_name}:{dataset_config}/{split} at offset {offset}"
                    )
                for row_wrapper in rows:
                    if not isinstance(row_wrapper, dict):
                        continue
                    handle.write(json.dumps(row_wrapper, ensure_ascii=True) + "\n")
                    row_count += 1
    except BenchmarkRunnerSettingsError:
        if temp_rows_path.exists():
            temp_rows_path.unlink()
        if _write_dataset_split_cache_from_local_hf_cache(
            benchmark_name=benchmark_name,
            dataset_config=dataset_config,
            split=split,
            rows_path=rows_path,
            meta_path=meta_path,
            expected_row_count=split_row_count,
        ):
            return
        raise

    meta_payload = {
        "benchmark_name": benchmark_name,
        "dataset_config": dataset_config,
        "split": split,
        "row_count": row_count,
        "expected_row_count": split_row_count,
        "complete": True,
        "cached_at": int(time.time()),
    }
    temp_rows_path.replace(rows_path)
    _write_json_object_atomic(meta_path, meta_payload)


def _cache_resolved_dataset_row(
    *,
    benchmark_name: str,
    dataset_config: str,
    split: str,
    row_wrapper: dict[str, Any],
) -> None:
    """Append a single resolved row to the split rows cache.

    Lets consumers that scan ``rows.jsonl`` (e.g. the SWE-rebench eval harness)
    find the instance without prefetching the whole split. Idempotent: skips
    when a row with the same instance_id is already cached.
    """
    if not isinstance(row_wrapper, dict):
        return
    row = row_wrapper.get("row")
    if not isinstance(row, dict):
        return
    instance_id = row.get("instance_id")
    if not isinstance(instance_id, str) or not instance_id.strip():
        return

    rows_path = _dataset_split_rows_cache_path(benchmark_name, dataset_config, split)
    if rows_path.is_file():
        try:
            with rows_path.open("r", encoding="utf-8") as handle:
                for raw_line in handle:
                    line = raw_line.strip()
                    if not line:
                        continue
                    try:
                        existing = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    existing_row = existing.get("row") if isinstance(existing, dict) else None
                    if isinstance(existing_row, dict) and existing_row.get("instance_id") == instance_id:
                        return
        except OSError:
            pass

    try:
        rows_path.parent.mkdir(parents=True, exist_ok=True)
        with rows_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"row": row}, ensure_ascii=True) + "\n")
    except OSError:
        pass


def _resolve_cached_dataset_row(
    *,
    benchmark_name: str,
    dataset_config: str,
    split: str,
    lookup_field: str,
    selection_id: str,
) -> dict[str, Any] | None:
    rows_path = _dataset_split_rows_cache_path(benchmark_name, dataset_config, split)
    if not rows_path.is_file():
        return None

    with rows_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            try:
                row_wrapper = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(row_wrapper, dict):
                continue
            row = row_wrapper.get("row")
            if not isinstance(row, dict):
                continue
            candidate = row.get(lookup_field)
            if not isinstance(candidate, str) and lookup_field != "instance_id":
                candidate = row.get("instance_id")
            if isinstance(candidate, str) and candidate.strip() == selection_id:
                return row_wrapper
    return None


def _hf_datasets_cache_root() -> Path:
    configured = _first_non_empty(os.getenv("HF_DATASETS_CACHE"))
    if configured:
        return Path(configured).expanduser().resolve()
    return (Path.home() / ".cache" / "huggingface" / "datasets").resolve()


def _hf_datasets_cache_dataset_dir(benchmark_name: str) -> Path:
    root = _hf_datasets_cache_root()
    candidate_name = benchmark_name.replace("/", "___")
    direct = root / candidate_name
    if direct.is_dir():
        return direct

    lower_name = candidate_name.lower()
    lower_direct = root / lower_name
    if lower_direct.is_dir():
        return lower_direct

    try:
        for child in root.iterdir():
            if child.is_dir() and child.name.lower() == lower_name:
                return child
    except FileNotFoundError:
        return direct
    return direct


def _load_dataset_info_from_local_hf_cache(benchmark_name: str) -> dict[str, Any] | None:
    dataset_dir = _hf_datasets_cache_dataset_dir(benchmark_name)
    if not dataset_dir.is_dir():
        return None

    dataset_configs: dict[str, Any] = {}
    for config_dir in sorted(dataset_dir.iterdir()):
        if not config_dir.is_dir():
            continue
        candidate = next(config_dir.glob("**/dataset_info.json"), None)
        if candidate is None or not candidate.is_file():
            continue
        payload = _read_cached_json_object(candidate)
        if payload is None:
            continue
        dataset_configs[config_dir.name] = payload

    if not dataset_configs:
        return None
    return {"dataset_info": dataset_configs}


def _write_dataset_split_cache_from_local_hf_cache(
    *,
    benchmark_name: str,
    dataset_config: str,
    split: str,
    rows_path: Path,
    meta_path: Path,
    expected_row_count: int,
) -> bool:
    try:
        from datasets import DownloadConfig, load_dataset
    except ImportError:
        return False

    try:
        dataset = load_dataset(
            benchmark_name,
            dataset_config,
            split=split,
            download_config=DownloadConfig(local_files_only=True),
        )
    except Exception:
        return False

    temp_rows_path = rows_path.with_suffix(f".jsonl.{os.getpid()}.tmp")
    row_count = 0
    with temp_rows_path.open("w", encoding="utf-8") as handle:
        for row in dataset:
            if not isinstance(row, dict):
                continue
            handle.write(json.dumps({"row": row}, ensure_ascii=True) + "\n")
            row_count += 1

    meta_payload = {
        "benchmark_name": benchmark_name,
        "dataset_config": dataset_config,
        "split": split,
        "row_count": row_count,
        "expected_row_count": expected_row_count,
        "complete": True,
        "cached_at": int(time.time()),
        "source": "local-hf-datasets-cache",
    }
    temp_rows_path.replace(rows_path)
    _write_json_object_atomic(meta_path, meta_payload)
    return True


def _sql_string_literal(value: str) -> str:
    escaped = value.replace("'", "''")
    return f"'{escaped}'"


def _normalize_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    parsed = _parse_json_string(value)
    if isinstance(parsed, dict):
        return parsed
    return {}


def _normalize_sequence(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    parsed = _parse_json_string(value)
    if isinstance(parsed, list):
        return parsed
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def _parse_json_string(value: Any) -> Any:
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text or text[0] not in "[{":
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _optional_string(value: Any) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _first_non_empty(*values: Any) -> str:
    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _coerce_positive_int(value: Any, default: int) -> int:
    if value is None:
        return default
    if isinstance(value, bool):
        raise BenchmarkRunnerSettingsError("Boolean value cannot be used where an integer is required")
    try:
        coerced = int(value)
    except (TypeError, ValueError) as exc:
        raise BenchmarkRunnerSettingsError(f"Expected an integer value, got {value!r}") from exc
    if coerced <= 0:
        raise BenchmarkRunnerSettingsError(f"Expected a positive integer value, got {coerced}")
    return coerced


def _coerce_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    raise BenchmarkRunnerSettingsError(f"Expected a boolean value, got {value!r}")


def _slug(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-._")
    return normalized or "default"
