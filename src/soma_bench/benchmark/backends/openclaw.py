from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .base import RuntimeBackend, RuntimeExecutionContext, RuntimeExecutionResult
from ..progress import emit_progress
from ..swebench_images import resolve_benchmark_runtime_image
from ..swerebench_eval import capture_repo_patch, maybe_run_swerebench_evaluation
from ..swebench_images import resolve_benchmark_runtime_image

OPENCLAW_WORKSPACE_ERROR = (
    "OpenClaw backend currently supports only docker workspace execution. "
    "Received workspace={workspace!r}."
)

OPENCLAW_GATEWAY_IMAGE = "alpine/openclaw:latest"
OPENCLAW_GATEWAY_PORT = 18789
OPENCLAW_GATEWAY_START_RETRIES = 20
OPENCLAW_GATEWAY_START_SLEEP_SECONDS = 0.5
OPENCLAW_DIND_IMAGE = "docker:dind"
OPENCLAW_DIND_PORT = 2375
OPENCLAW_DIND_START_RETRIES = 60
OPENCLAW_DIND_START_SLEEP_SECONDS = 0.5
OPENCLAW_GATEWAY_STATE_DIRNAME = "openclaw-gateway-state"
OPENCLAW_PROBLEMS_DIRNAME = "openclaw-problems"
OPENCLAW_CONTAINER_ARTIFACTS_PATH = "/artifacts"
OPENCLAW_CONTAINER_STATE_DIR = "/home/node/.openclaw"
OPENCLAW_CONTAINER_PROBLEMS_ROOT = "/workspace/openclaw-problems"
OPENCLAW_CONTAINER_PLUGIN_ROOT = "/workspace/openclaw-plugins"
OPENCLAW_SESSION_SALT_ENV = "SOMA_OPENCLAW_SESSION_SALT"
OPENCLAW_RUNTIME_ENV_SCRIPT_NAME = "OPENCLAW_BENCHMARK_ENV.sh"
SOMA_PLUGIN_ID = "soma-miner"
PLUGIN_MANIFEST_NAME = "openclaw.plugin.json"
PLUGIN_ENTRYPOINT = "base_miner.py"
PLUGIN_VENV_DIRNAME = ".soma-openclaw-venv"
PLUGIN_REQUIREMENTS_MARKER = ".soma-openclaw-requirements.sha256"
OPENCLAW_WS_WATCHDOG_DEFAULT_RETRIES = 1
OPENCLAW_WS_WATCHDOG_MAX_RETRIES = 5
OPENCLAW_WS_WATCHDOG_DEFAULT_BACKOFF_SECONDS = 1.0
OPENCLAW_WS_WATCHDOG_MAX_BACKOFF_SECONDS = 10.0
OPENCLAW_HEARTBEAT_DISABLED_INTERVAL = "0m"
OPENCLAW_GATEWAY_PIDS_LIMIT = 4096
OPENCLAW_GATEWAY_ULIMIT_NOFILE = "4096"
_PLUGIN_REINSTALLED_FOR_RUNS: set[str] = set()
_GATEWAY_RECOVERY_LOCK = threading.Lock()
_GATEWAY_RECOVERY_LOCKS: dict[str, threading.Lock] = {}
_WS_CLOSE_RE = re.compile(
    r"gateway closed \((?P<code>\d{3,4})\s+(?P<reason>[^)]*)\)",
    re.IGNORECASE,
)
_DOCKER_NUMERIC_USER_RE = re.compile(r"^\d+(?::\d+)?$")


def _run_command(
    args: list[str],
    *,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, text=True, capture_output=True, check=False, env=env)


def _host_docker_binary() -> str | None:
    docker_binary = shutil.which("docker")
    if docker_binary and os.path.isabs(docker_binary):
        return docker_binary
    return None


def _docker_user_is_root(user_spec: str) -> bool:
    normalized = user_spec.strip().lower()
    return normalized in {"root", "0", "0:0"}


def _docker_socket_mount_args(*, user_spec: str) -> list[str]:
    docker_sock = "/var/run/docker.sock"
    if not os.path.exists(docker_sock):
        return []

    args = ["-v", f"{docker_sock}:{docker_sock}"]
    if _docker_user_is_root(user_spec):
        return args

    try:
        docker_sock_gid = os.stat(docker_sock).st_gid
    except OSError:
        return args

    return ["--group-add", str(docker_sock_gid), *args]


def _runtime_option_name_candidates(name: str) -> tuple[str, ...]:
    return (name, name.replace("gateway_", ""))


def _resolve_runtime_options(context: RuntimeExecutionContext) -> dict[str, Any]:
    value = context.run_payload.get("runtime_options")
    if isinstance(value, dict):
        return value
    return {}


def _resolve_runtime_option(context: RuntimeExecutionContext, name: str) -> Any:
    runtime_options = _resolve_runtime_options(context)
    for candidate in _runtime_option_name_candidates(name):
        if candidate in runtime_options:
            return runtime_options[candidate]
    return None


def _coerce_bool_option(raw_value: Any) -> bool | None:
    if isinstance(raw_value, bool):
        return raw_value
    if isinstance(raw_value, str):
        normalized = raw_value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    return None


def _coerce_positive_int(raw_value: Any, default: int) -> int:
    if raw_value is None:
        return default
    try:
        value = int(raw_value)
    except (TypeError, ValueError):
        return default
    return value if value > 0 else default


def _coerce_float_range(raw_value: Any, default: float, *, minimum: float, maximum: float) -> float:
    if raw_value is None:
        return default
    try:
        value = float(raw_value)
    except (TypeError, ValueError):
        return default
    return value if minimum <= value <= maximum else default


def _slug(value: str, *, prefix: str = "", limit: int = 48) -> str:
    normalized = re.sub(r"[^a-z0-9_.-]+", "-", value.lower()).strip("-._")
    normalized = normalized or "default"
    if prefix:
        normalized = f"{prefix}{normalized}"
    if len(normalized) <= limit:
        return normalized
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:8]
    head = normalized[: max(1, limit - len(digest) - 1)].rstrip("-._")
    return f"{head}-{digest}"


def _benchmark_key(context: RuntimeExecutionContext) -> str:
    return str(context.instance.benchmark_id or context.instance.instance_id)


def _problem_root(context: RuntimeExecutionContext) -> Path:
    override = _resolve_runtime_option(context, "openclaw_sandbox_root")
    if not isinstance(override, str) or not override.strip():
        override = os.getenv("SOMA_OPENCLAW_SANDBOX_ROOT")

    if isinstance(override, str) and override.strip():
        base_dir = Path(override.strip()).expanduser().resolve()
    else:
        base_dir = (Path(tempfile.gettempdir()) / "soma-openclaw-problems").resolve()

    return base_dir / _slug(str(context.output_dir.resolve()), prefix="run-") / _slug(_benchmark_key(context))


def _agent_workspace_dir(context: RuntimeExecutionContext) -> Path:
    return _problem_root(context)


def _gateway_workspace_root(context: RuntimeExecutionContext) -> Path:
    return _problem_root(context).parent


def _container_problem_root(context: RuntimeExecutionContext) -> str:
    return f"{OPENCLAW_CONTAINER_PROBLEMS_ROOT}/{_slug(_benchmark_key(context))}"


def _container_agent_workspace_dir(context: RuntimeExecutionContext) -> str:
    return _container_repo_path(context)


def _container_repo_path(context: RuntimeExecutionContext) -> str:
    return f"{_container_problem_root(context)}/repo"


def _container_plugin_path() -> str:
    return f"{OPENCLAW_CONTAINER_PLUGIN_ROOT}/{SOMA_PLUGIN_ID}"


def _repo_dir(context: RuntimeExecutionContext) -> Path:
    return _problem_root(context) / "repo"


def _artifacts_dir(context: RuntimeExecutionContext) -> Path:
    return _problem_root(context) / "artifacts"


def _runtime_env_script_path(context: RuntimeExecutionContext) -> Path:
    return _repo_dir(context) / OPENCLAW_RUNTIME_ENV_SCRIPT_NAME


def _container_runtime_env_script_path(context: RuntimeExecutionContext) -> str:
    return f"{_container_repo_path(context)}/{OPENCLAW_RUNTIME_ENV_SCRIPT_NAME}"


def _state_dir(context: RuntimeExecutionContext) -> Path:
    return (context.output_dir / OPENCLAW_GATEWAY_STATE_DIRNAME).resolve()


def _config_path(context: RuntimeExecutionContext) -> Path:
    return _state_dir(context) / "openclaw.json"


def _default_plugin_path() -> Path:
    repo_root = Path(__file__).resolve().parents[4]
    return repo_root.parent / "SOMA-plugin"


def _resolve_plugin_enabled(context: RuntimeExecutionContext) -> bool:
    explicit = _coerce_bool_option(_resolve_runtime_option(context, "openclaw_plugin_enabled"))
    if explicit is not None:
        return explicit

    env_value = _coerce_bool_option(os.getenv("SOMA_OPENCLAW_PLUGIN_ENABLED"))
    if env_value is not None:
        return env_value

    explicit_path = _resolve_runtime_option(context, "openclaw_plugin_path")
    if isinstance(explicit_path, str) and explicit_path.strip():
        return True

    env_path = os.getenv("SOMA_OPENCLAW_PLUGIN_PATH")
    if isinstance(env_path, str) and env_path.strip():
        return True

    return _default_plugin_path().is_dir()


def _resolve_plugin_path(context: RuntimeExecutionContext) -> Path | None:
    if not _resolve_plugin_enabled(context):
        return None

    for value in (
        _resolve_runtime_option(context, "openclaw_plugin_path"),
        os.getenv("SOMA_OPENCLAW_PLUGIN_PATH"),
    ):
        if isinstance(value, str) and value.strip():
            path = Path(value.strip()).expanduser().resolve()
            if not path.is_dir():
                raise RuntimeError(f"OpenClaw SOMA plugin path does not exist: {path}")
            return path

    path = _default_plugin_path().resolve()
    if path.is_dir():
        return path
    explicit_enabled = _coerce_bool_option(_resolve_runtime_option(context, "openclaw_plugin_enabled"))
    env_enabled = _coerce_bool_option(os.getenv("SOMA_OPENCLAW_PLUGIN_ENABLED"))
    if explicit_enabled is True or env_enabled is True:
        raise RuntimeError(
            "OpenClaw SOMA plugin is enabled, but no plugin checkout was found. "
            "Set SOMA_OPENCLAW_PLUGIN_PATH or --openclaw-plugin-path."
        )
    return None


def _load_plugin_manifest(path: Path) -> dict[str, Any]:
    manifest_path = path / PLUGIN_MANIFEST_NAME
    if not manifest_path.is_file():
        raise RuntimeError(f"OpenClaw plugin manifest is missing: {manifest_path}")
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"OpenClaw plugin manifest is invalid JSON: {manifest_path}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"OpenClaw plugin manifest must contain a JSON object: {manifest_path}")
    return payload


def _resolve_plugin_id(path: Path) -> str:
    plugin_id = str(_load_plugin_manifest(path).get("id") or "").strip()
    if plugin_id != SOMA_PLUGIN_ID:
        raise RuntimeError(
            "OpenClaw plugin manifest has unsupported id: "
            f"{plugin_id!r}; expected {SOMA_PLUGIN_ID!r}"
        )
    return plugin_id


def _resolve_plugin_entrypoint(path: Path) -> str:
    if (path / PLUGIN_ENTRYPOINT).is_file():
        return PLUGIN_ENTRYPOINT
    raise RuntimeError(f"OpenClaw plugin checkout must provide {PLUGIN_ENTRYPOINT}: {path}")


def _resolve_plugin_reinstall_on_run_start(context: RuntimeExecutionContext) -> bool:
    option_value = _coerce_bool_option(
        _resolve_runtime_option(context, "openclaw_plugin_reinstall_on_run_start")
    )
    if option_value is not None:
        return option_value
    return _coerce_bool_option(os.getenv("SOMA_OPENCLAW_PLUGIN_REINSTALL_ON_RUN_START")) or False


def _build_agent_id(context: RuntimeExecutionContext) -> str:
    return _slug(_benchmark_key(context), prefix="bench-", limit=64)


def _build_session_id(context: RuntimeExecutionContext) -> str:
    session_salt = os.getenv(OPENCLAW_SESSION_SALT_ENV, "").strip() or str(int(time.time() * 1000))
    return _slug(f"{context.instance.benchmark_id}-{session_salt}", prefix="run-", limit=72)


def _resolve_keep_gateway_enabled(context: RuntimeExecutionContext) -> bool:
    raw_value = _resolve_runtime_option(context, "openclaw_keep_gateway")
    option_value = _coerce_bool_option(raw_value)
    if option_value is not None:
        return option_value

    env_value = os.getenv("SOMA_OPENCLAW_KEEP_GATEWAY")
    return _coerce_bool_option(env_value) or False


def _resolve_keep_workspace_enabled(context: RuntimeExecutionContext) -> bool:
    raw_value = _resolve_runtime_option(context, "openclaw_keep_workspace")
    option_value = _coerce_bool_option(raw_value)
    if option_value is not None:
        return option_value

    env_value = os.getenv("SOMA_OPENCLAW_KEEP_WORKSPACE")
    return _coerce_bool_option(env_value) or False
def _resolve_requested_openclaw_user(context: RuntimeExecutionContext) -> str | None:
    for value in (
        _resolve_runtime_option(context, "openclaw_user"),
        _resolve_runtime_option(context, "openclaw_container_user"),
        os.getenv("SOMA_OPENCLAW_USER"),
        os.getenv("SOMA_OPENCLAW_CONTAINER_USER"),
    ):
        if not isinstance(value, str) or not value.strip():
            continue
        normalized = value.strip()
        if normalized.lower() == "current":
            if not hasattr(os, "getuid") or not hasattr(os, "getgid"):
                raise RuntimeError("openclaw_user=current is not supported on this platform")
            return f"{os.getuid()}:{os.getgid()}"
        return normalized
    return None


def _resolve_openclaw_user(context: RuntimeExecutionContext) -> str:
    requested_user = _resolve_requested_openclaw_user(context)
    if requested_user is not None:
        return requested_user
    raise RuntimeError(
        "OpenClaw requires an explicit gateway/CLI container user. "
        "Set openclaw_user or openclaw_container_user (CLI: --openclaw-current-user) "
        "or provide SOMA_OPENCLAW_USER / SOMA_OPENCLAW_CONTAINER_USER."
    )


def _resolve_workspace_container_user(context: RuntimeExecutionContext) -> str | None:
    requested_user = _resolve_requested_openclaw_user(context)
    if requested_user is None:
        return None
    if requested_user.lower() == "root" or _DOCKER_NUMERIC_USER_RE.fullmatch(requested_user):
        return requested_user
    if hasattr(os, "getuid") and hasattr(os, "getgid"):
        return f"{os.getuid()}:{os.getgid()}"
    return requested_user


def _path_owner_spec(path: Path) -> str:
    stat_result = path.stat()
    return f"{stat_result.st_uid}:{stat_result.st_gid}"


def _remove_plugin_venv(
    context: RuntimeExecutionContext,
    *,
    plugin_path: Path,
    venv_path: Path,
) -> None:
    if not venv_path.exists():
        return
    try:
        shutil.rmtree(venv_path)
        return
    except PermissionError:
        pass

    relative_venv_path = venv_path.resolve().relative_to(plugin_path.resolve())
    container_venv_path = Path(_container_plugin_path()) / relative_venv_path
    cleanup_result = _run_command(
        [
            "docker",
            "run",
            "--rm",
            "--user",
            "root",
            "-v",
            f"{plugin_path}:{_container_plugin_path()}:rw",
            _resolve_gateway_image(context),
            "sh",
            "-lc",
            f"rm -rf {shlex.quote(str(container_venv_path))}",
        ]
    )
    if cleanup_result.returncode != 0:
        raise RuntimeError(
            "Failed to remove the OpenClaw SOMA plugin runtime before reinstall. "
            f"{(cleanup_result.stderr or cleanup_result.stdout or '').strip()}"
        )


def _resolve_ws_watchdog_retries(context: RuntimeExecutionContext) -> int:
    for value in (
        _resolve_runtime_option(context, "openclaw_ws_watchdog_retries"),
        os.getenv("SOMA_OPENCLAW_WS_WATCHDOG_RETRIES"),
    ):
        if value is None:
            continue
        try:
            retries = int(value)
        except (TypeError, ValueError):
            continue
        return max(0, min(retries, OPENCLAW_WS_WATCHDOG_MAX_RETRIES))
    return OPENCLAW_WS_WATCHDOG_DEFAULT_RETRIES


def _resolve_ws_watchdog_backoff_seconds(context: RuntimeExecutionContext) -> float:
    for value in (
        _resolve_runtime_option(context, "openclaw_ws_watchdog_backoff_seconds"),
        os.getenv("SOMA_OPENCLAW_WS_WATCHDOG_BACKOFF_SECONDS"),
    ):
        if value is None:
            continue
        try:
            backoff = float(value)
        except (TypeError, ValueError):
            continue
        return max(0.0, min(backoff, OPENCLAW_WS_WATCHDOG_MAX_BACKOFF_SECONDS))
    return OPENCLAW_WS_WATCHDOG_DEFAULT_BACKOFF_SECONDS


def _is_gateway_ws_1006_error(*, stderr: str, stdout: str) -> bool:
    combined = f"{stderr}\n{stdout}".lower()
    return "gatewaytransporterror" in combined and "1006" in combined


def _is_gateway_container_missing_error(*, stderr: str, stdout: str) -> bool:
    combined = f"{stderr}\n{stdout}".lower()
    return "joining network namespace of container" in combined and "no such container" in combined


def _gateway_recovery_lock(gateway_name: str) -> threading.Lock:
    with _GATEWAY_RECOVERY_LOCK:
        lock = _GATEWAY_RECOVERY_LOCKS.get(gateway_name)
        if lock is None:
            lock = threading.Lock()
            _GATEWAY_RECOVERY_LOCKS[gateway_name] = lock
        return lock


def _recover_gateway_for_retry(
    context: RuntimeExecutionContext,
    *,
    gateway_name: str,
    agent_id: str,
) -> None:
    with _gateway_recovery_lock(gateway_name):
        if _docker_container_running(gateway_name):
            return
        emit_progress(
            f"[{context.instance.instance_id}] recovering missing OpenClaw gateway container {gateway_name}",
            component="openclaw",
        )
        recovered_gateway_name, _ = _start_gateway_container(context, restart_existing=True)
        _wait_for_gateway_ready(context, gateway_name=recovered_gateway_name)
        _wait_for_agent_registered(context, gateway_name=recovered_gateway_name, agent_id=agent_id)


def _extract_ws_close_details(error_text: str) -> tuple[int | None, str | None]:
    if not error_text:
        return None, None
    match = _WS_CLOSE_RE.search(error_text)
    if match:
        try:
            code = int(match.group("code"))
        except (TypeError, ValueError):
            code = None
        reason = str(match.group("reason") or "").strip() or None
        return code, reason

    lowered = error_text.lower()
    if "1006" in lowered and "abnormal closure" in lowered:
        return 1006, "abnormal closure"
    return None, None


def _parse_iso8601_timestamp(value: str) -> float | None:
    raw = value.strip()
    if not raw:
        return None
    if raw.endswith("Z"):
        raw = f"{raw[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()


def _last_runtime_hook_event_for_session(
    plugin_runtime: dict[str, Any],
    *,
    session_id: str,
) -> dict[str, Any] | None:
    plugin_path_raw = plugin_runtime.get("plugin_path")
    if not isinstance(plugin_path_raw, str) or not plugin_path_raw.strip():
        return None
    hooks_path = Path(plugin_path_raw).expanduser().resolve() / "logs" / "runtime-hooks.jsonl"
    if not hooks_path.is_file():
        return None

    last_payload: dict[str, Any] | None = None
    for _, payload in _jsonl_rows(hooks_path):
        if str(payload.get("sessionId") or "") == session_id:
            last_payload = payload

    if last_payload is None:
        return None

    event: dict[str, Any] = {
        "timestamp": str(last_payload.get("timestamp") or "").strip() or None,
        "event_type": str(last_payload.get("eventType") or "").strip() or None,
    }
    if "reason" in last_payload:
        event["reason"] = last_payload.get("reason")
    if "inputMessageCount" in last_payload:
        event["input_message_count"] = last_payload.get("inputMessageCount")
    if "outputMessageCount" in last_payload:
        event["output_message_count"] = last_payload.get("outputMessageCount")
    return event


def _gateway_container_state_snapshot(gateway_name: str) -> dict[str, Any]:
    inspect_result = _run_command(["docker", "inspect", gateway_name])
    if inspect_result.returncode != 0:
        return {
            "inspect_error": (inspect_result.stderr or inspect_result.stdout or "").strip() or "inspect_failed",
            "running": False,
        }
    try:
        payload = json.loads(inspect_result.stdout or "[]")
    except json.JSONDecodeError:
        return {
            "inspect_error": "invalid_inspect_json",
            "running": False,
        }
    if not isinstance(payload, list) or not payload or not isinstance(payload[0], dict):
        return {
            "inspect_error": "empty_inspect_payload",
            "running": False,
        }
    state = payload[0].get("State")
    if not isinstance(state, dict):
        return {
            "inspect_error": "missing_state",
            "running": False,
        }
    return {
        "running": bool(state.get("Running")),
        "status": str(state.get("Status") or ""),
        "restart_count": int(payload[0].get("RestartCount") or 0),
        "oom_killed": bool(state.get("OOMKilled")),
        "exit_code": state.get("ExitCode"),
        "error": str(state.get("Error") or ""),
        "started_at": str(state.get("StartedAt") or ""),
        "finished_at": str(state.get("FinishedAt") or ""),
        "pid": state.get("Pid"),
    }


def _resolve_gateway_image(context: RuntimeExecutionContext) -> str:
    for value in (
        _resolve_runtime_option(context, "openclaw_gateway_image"),
        os.getenv("SOMA_OPENCLAW_GATEWAY_IMAGE"),
        _resolve_runtime_option(context, "openclaw_container_image"),
        os.getenv("SOMA_OPENCLAW_CONTAINER_IMAGE"),
        OPENCLAW_GATEWAY_IMAGE,
    ):
        if isinstance(value, str) and value.strip():
            return value.strip()
    return OPENCLAW_GATEWAY_IMAGE


def _resolve_benchmark_image(context: RuntimeExecutionContext) -> str:
    for value in (
        context.run_payload.get("runtime_container_image"),
        context.instance.hidden_eval.get("docker_image"),
        context.instance.hidden_eval.get("image_name"),
    ):
        if isinstance(value, str) and value.strip():
            return value.strip()
    runtime_options = _resolve_runtime_options(context)
    fallback_image = resolve_benchmark_runtime_image(
        instance_id=context.instance.instance_id,
        hidden_eval=context.instance.hidden_eval,
        runtime_options=runtime_options,
    )
    if fallback_image:
        return fallback_image
    raise RuntimeError(
        "OpenClaw backend could not resolve the benchmark Docker image. "
        "Expected runtime_container_image or hidden_eval.docker_image/image_name."
    )


def _resolve_gateway_token(context: RuntimeExecutionContext) -> str:
    for value in (
        _resolve_runtime_option(context, "openclaw_gateway_token"),
        os.getenv("SOMA_OPENCLAW_GATEWAY_TOKEN"),
        os.getenv("OPENCLAW_GATEWAY_TOKEN"),
    ):
        if isinstance(value, str) and value.strip():
            return value.strip()
    digest = hashlib.sha256(str(context.output_dir.resolve()).encode("utf-8")).hexdigest()[:24]
    return f"soma-openclaw-{digest}"


def _resolve_gateway_name(context: RuntimeExecutionContext) -> str:
    for value in (
        _resolve_runtime_option(context, "openclaw_gateway_name"),
        os.getenv("SOMA_OPENCLAW_GATEWAY_NAME"),
    ):
        if isinstance(value, str) and value.strip():
            return value.strip()
    digest = hashlib.sha256(str(context.output_dir.resolve()).encode("utf-8")).hexdigest()[:12]
    return f"soma-openclaw-gateway-{digest}"

def _resolve_gateway_port(context: RuntimeExecutionContext) -> int:
    for value in (
        _resolve_runtime_option(context, "openclaw_gateway_port"),
        os.getenv("SOMA_OPENCLAW_GATEWAY_PORT"),
    ):
        if value is None:
            continue
        try:
            port = int(value)
        except (TypeError, ValueError):
            continue
        if port > 0:
            return port
    return OPENCLAW_GATEWAY_PORT


def _resolve_dind_image(context: RuntimeExecutionContext) -> str:
    for value in (
        _resolve_runtime_option(context, "openclaw_dind_image"),
        os.getenv("SOMA_OPENCLAW_DIND_IMAGE"),
    ):
        if isinstance(value, str) and value.strip():
            return value.strip()
    return OPENCLAW_DIND_IMAGE


def _resolve_requested_private_network_name(context: RuntimeExecutionContext) -> str | None:
    for value in (
        _resolve_runtime_option(context, "openclaw_private_network_name"),
        os.getenv("SOMA_OPENCLAW_PRIVATE_NETWORK_NAME"),
    ):
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _resolve_private_network_name(context: RuntimeExecutionContext) -> str:
    requested_name = _resolve_requested_private_network_name(context)
    if requested_name is not None:
        return requested_name
    digest = hashlib.sha256(str(context.output_dir.resolve()).encode("utf-8")).hexdigest()[:12]
    return f"soma-openclaw-net-{digest}"


def _resolve_isolated_control_network_name(context: RuntimeExecutionContext) -> str:
    return f"{_resolve_gateway_name(context)}-control"


def _dind_container_name(context: RuntimeExecutionContext) -> str:
    return f"{_resolve_gateway_name(context)}-dind"


def _isolated_docker_host(context: RuntimeExecutionContext) -> str:
    return f"tcp://{_dind_container_name(context)}:{OPENCLAW_DIND_PORT}"


def _openclaw_container_hardening_args() -> list[str]:
    return [
        "--security-opt",
        "no-new-privileges",
        "--pids-limit",
        str(OPENCLAW_GATEWAY_PIDS_LIMIT),
        "--ulimit",
        f"nofile={OPENCLAW_GATEWAY_ULIMIT_NOFILE}:{OPENCLAW_GATEWAY_ULIMIT_NOFILE}",
    ]


def _resolve_agent_args(context: RuntimeExecutionContext) -> list[str]:
    for value in (
        _resolve_runtime_option(context, "openclaw_agent_args"),
        os.getenv("SOMA_OPENCLAW_AGENT_ARGS"),
        _resolve_runtime_option(context, "openclaw_command"),
        os.getenv("SOMA_OPENCLAW_COMMAND"),
    ):
        if isinstance(value, str) and value.strip():
            return shlex.split(value)
    return []


def _resolve_run_id_header_value(context: RuntimeExecutionContext) -> str:
    for value in (
        _resolve_runtime_option(context, "openclaw_run_id_header_value"),
        os.getenv("SOMA_OPENCLAW_RUN_ID_HEADER_VALUE"),
    ):
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _is_openrouter_context(context: RuntimeExecutionContext) -> bool:
    model = str(context.llm_config.get("model", "")).strip()
    base_url = str(context.llm_config.get("base_url", "")).strip()
    if base_url:
        return "openrouter" in base_url.lower()
    return model.startswith("openrouter/")


def _uses_openai_compatible_proxy_without_api_key(context: RuntimeExecutionContext) -> bool:
    if _is_openrouter_context(context):
        return False
    model = str(context.llm_config.get("model", "")).strip()
    base_url = str(context.llm_config.get("base_url", "")).strip()
    ignore_api_key = _coerce_bool_option(_resolve_runtime_option(context, "openclaw_ignore_api_key"))
    return bool(ignore_api_key and base_url and model and "/" in model and not model.startswith("openai/"))


def _resolve_openclaw_model(context: RuntimeExecutionContext) -> str:
    model = str(context.llm_config.get("model", "")).strip()
    if not model:
        return model
    if _is_openrouter_context(context) and not model.startswith("openrouter/"):
        return f"openrouter/{model}"
    if _uses_openai_compatible_proxy_without_api_key(context):
        return f"openai/{model}"
    return model


def _docker_container_running(name: str) -> bool:
    result = _run_command(["docker", "inspect", "-f", "{{.State.Running}}", name])
    return result.returncode == 0 and (result.stdout or "").strip().lower() == "true"


def _docker_image_exists(image_ref: str) -> bool:
    result = _run_command(["docker", "image", "inspect", image_ref])
    return result.returncode == 0


def _ensure_docker_image_available(image_ref: str, *, role: str) -> None:
    if _docker_image_exists(image_ref):
        return

    pull_result = _run_command(["docker", "pull", image_ref])
    if pull_result.returncode == 0 and _docker_image_exists(image_ref):
        return

    message = (pull_result.stderr or pull_result.stdout or "").strip()
    raise RuntimeError(
        f"{role} Docker image is not available locally and automatic pull failed for {image_ref}: {message}"
    )


def _docker_remove_container(name: str) -> None:
    _run_command(["docker", "rm", "-f", name])


def _docker_remove_network(name: str) -> None:
    if name in {"", "bridge", "host", "none"}:
        return
    _run_command(["docker", "network", "rm", name])


def _docker_list_container_ids_by_label(label: str) -> list[str]:
    result = _run_command(["docker", "ps", "-aq", "--filter", f"label={label}"])
    if result.returncode != 0:
        return []
    return [line.strip() for line in (result.stdout or "").splitlines() if line.strip()]


def _docker_network_contains_container(network_name: str, container_name: str) -> bool:
    inspect_result = _run_command(["docker", "network", "inspect", network_name])
    if inspect_result.returncode != 0:
        return False
    try:
        payload = json.loads(inspect_result.stdout or "[]")
    except json.JSONDecodeError:
        return False
    if not isinstance(payload, list) or not payload or not isinstance(payload[0], dict):
        return False
    containers = payload[0].get("Containers")
    if not isinstance(containers, dict):
        return False
    return any(
        isinstance(entry, dict) and entry.get("Name") == container_name
        for entry in containers.values()
    )


def _docker_connect_container_to_network(*, container_name: str, network_name: str) -> None:
    if _docker_network_contains_container(network_name, container_name):
        return
    result = _run_command(["docker", "network", "connect", network_name, container_name])
    if result.returncode != 0:
        raise RuntimeError(
            f"Failed to connect container {container_name!r} to Docker network {network_name!r}: "
            f"{(result.stderr or result.stdout or '').strip()}"
        )


def _run_isolated_docker_command(
    context: RuntimeExecutionContext,
    args: list[str],
) -> subprocess.CompletedProcess[str]:
    return _run_command(["docker", "exec", _dind_container_name(context), "docker", *args])


def _isolated_docker_list_container_ids_by_label(
    context: RuntimeExecutionContext,
    label: str,
) -> list[str]:
    result = _run_isolated_docker_command(context, ["ps", "-aq", "--filter", f"label={label}"])
    if result.returncode != 0:
        return []
    return [line.strip() for line in (result.stdout or "").splitlines() if line.strip()]


def _isolated_docker_remove_container(context: RuntimeExecutionContext, name: str) -> None:
    _run_isolated_docker_command(context, ["rm", "-f", name])


def _build_sandbox_session_key(*, agent_id: str, session_id: str) -> str:
    return f"agent:{agent_id}:explicit:{session_id}"


def _cleanup_sandbox_containers(
    context: RuntimeExecutionContext,
    *,
    agent_id: str,
    session_id: str,
) -> None:
    session_key = _build_sandbox_session_key(agent_id=agent_id, session_id=session_id)
    if _docker_container_running(_dind_container_name(context)):
        container_ids = _isolated_docker_list_container_ids_by_label(
            context,
            f"openclaw.sessionKey={session_key}",
        )
        for container_id in container_ids:
            _isolated_docker_remove_container(context, container_id)
        return

    for container_id in _docker_list_container_ids_by_label(f"openclaw.sessionKey={session_key}"):
        _docker_remove_container(container_id)


def _reset_directory(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _ensure_private_network(context: RuntimeExecutionContext) -> tuple[str, bool]:
    name = _resolve_private_network_name(context)
    if name in {"", "bridge", "host", "none"}:
        return name, False
    if _run_command(["docker", "network", "inspect", name]).returncode == 0:
        return name, False
    create = _run_command(["docker", "network", "create", name])
    if create.returncode != 0 and _run_command(["docker", "network", "inspect", name]).returncode != 0:
        raise RuntimeError(
            "Failed to create the private OpenClaw network "
            f"{name!r}: {(create.stderr or create.stdout or '').strip()}"
        )
    return name, True


def _ensure_isolated_control_network(context: RuntimeExecutionContext) -> tuple[str, bool]:
    name = _resolve_isolated_control_network_name(context)
    if _run_command(["docker", "network", "inspect", name]).returncode == 0:
        return name, False
    create = _run_command(["docker", "network", "create", "--internal", name])
    if create.returncode != 0 and _run_command(["docker", "network", "inspect", name]).returncode != 0:
        raise RuntimeError(
            "Failed to create the isolated OpenClaw control network "
            f"{name!r}: {(create.stderr or create.stdout or '').strip()}"
        )
    return name, True


def _ensure_isolated_sandbox_network(context: RuntimeExecutionContext) -> None:
    name = _resolve_private_network_name(context)
    if name in {"", "none", "bridge", "host"}:
        return

    inspect = _run_isolated_docker_command(context, ["network", "inspect", name])
    if inspect.returncode == 0:
        return

    create = _run_isolated_docker_command(context, ["network", "create", "--internal", name])
    if create.returncode != 0 and _run_isolated_docker_command(context, ["network", "inspect", name]).returncode != 0:
        raise RuntimeError(
            "Failed to create the isolated OpenClaw sandbox network "
            f"{name!r}: {(create.stderr or create.stdout or '').strip()}"
        )


def _teardown_private_network(context: RuntimeExecutionContext) -> None:
    if _resolve_requested_private_network_name(context) is not None:
        return
    _docker_remove_network(_resolve_private_network_name(context))


def _teardown_isolated_control_network(context: RuntimeExecutionContext) -> None:
    _docker_remove_network(_resolve_isolated_control_network_name(context))


def _wait_for_isolated_daemon_ready(*, dind_name: str) -> None:
    last_error = ""
    for _ in range(OPENCLAW_DIND_START_RETRIES):
        if not _docker_container_running(dind_name):
            logs = _run_command(["docker", "logs", dind_name])
            raise RuntimeError(
                "OpenClaw isolated Docker daemon exited before becoming ready: "
                f"{(logs.stderr or logs.stdout or '').strip()}"
            )
        probe = _run_command(["docker", "exec", dind_name, "docker", "info"])
        if probe.returncode == 0:
            return
        last_error = (probe.stderr or probe.stdout or "").strip()
        time.sleep(OPENCLAW_DIND_START_SLEEP_SECONDS)
    raise RuntimeError(f"OpenClaw isolated Docker daemon did not become ready: {last_error}")


def _ensure_isolated_daemon(context: RuntimeExecutionContext) -> str:
    network_name, network_created = _ensure_isolated_control_network(context)

    dind_name = _dind_container_name(context)
    if _docker_container_running(dind_name):
        _ensure_isolated_sandbox_network(context)
        return _isolated_docker_host(context)

    _docker_remove_container(dind_name)
    _ensure_docker_image_available(_resolve_dind_image(context), role="OpenClaw isolated Docker daemon")
    args = [
        "docker",
        "run",
        "-d",
        "--rm",
        "--privileged",
        "--name",
        dind_name,
        "--network",
        network_name,
        "-e",
        "DOCKER_TLS_CERTDIR=",
        _resolve_dind_image(context),
    ]
    result = _run_command(args)
    if result.returncode != 0:
        if network_created:
            _docker_remove_network(network_name)
        raise RuntimeError(
            "Failed to start the OpenClaw isolated Docker daemon: "
            f"{(result.stderr or result.stdout or '').strip()}"
        )

    _wait_for_isolated_daemon_ready(dind_name=dind_name)
    _ensure_isolated_sandbox_network(context)
    return _isolated_docker_host(context)


def _teardown_isolated_daemon(context: RuntimeExecutionContext) -> None:
    _docker_remove_container(_dind_container_name(context))


def _ensure_image_in_isolated_daemon(
    context: RuntimeExecutionContext,
    image: str,
    *,
    role: str,
) -> None:
    _ensure_isolated_daemon(context)
    dind_name = _dind_container_name(context)
    if _run_command(["docker", "exec", dind_name, "docker", "image", "inspect", image]).returncode == 0:
        return
    if _run_command(["docker", "exec", dind_name, "docker", "pull", image]).returncode == 0:
        if _run_command(["docker", "exec", dind_name, "docker", "image", "inspect", image]).returncode == 0:
            return

    _ensure_docker_image_available(image, role=role)
    save_proc = subprocess.Popen(["docker", "save", image], stdout=subprocess.PIPE)
    try:
        load_proc = subprocess.run(
            ["docker", "exec", "-i", dind_name, "docker", "load"],
            stdin=save_proc.stdout,
            capture_output=True,
            check=False,
        )
    finally:
        if save_proc.stdout is not None:
            save_proc.stdout.close()
        save_proc.wait()

    if load_proc.returncode != 0 or save_proc.returncode != 0:
        load_message = b"".join(part for part in (load_proc.stderr, load_proc.stdout) if part)
        raise RuntimeError(
            f"Failed to load {role} image {image!r} into the isolated Docker daemon: "
            f"{load_message.decode('utf-8', errors='replace').strip()}"
        )


def _materialize_host_repo_from_benchmark_image(context: RuntimeExecutionContext) -> None:
    repo_dir = _repo_dir(context)
    _reset_directory(repo_dir)

    args = [
        "docker",
        "run",
        "--rm",
        "--network",
        "none",
        "--security-opt",
        "no-new-privileges",
        "--entrypoint",
        "sh",
    ]
    workspace_user = _resolve_workspace_container_user(context)
    if workspace_user is not None:
        args.extend(["--user", workspace_user])
    args.extend(
        [
            "-v",
            f"{repo_dir.resolve()}:/out:rw",
            _resolve_benchmark_image(context),
            "-lc",
            (
                "if [ -d /testbed ]; then "
                "cp -a --no-preserve=ownership /testbed/. /out/; "
                "else echo 'OpenClaw sandbox expected /testbed in the benchmark image' >&2; exit 1; fi"
            ),
        ]
    )
    result = _run_command(args)
    if result.returncode != 0:
        message = (result.stderr or result.stdout or "").strip()
        raise RuntimeError(f"Failed to materialize benchmark repository into host workspace: {message}")

    if not any(repo_dir.iterdir()):
        raise RuntimeError("Benchmark repository materialization produced an empty host workspace")


def _gateway_process_env(context: RuntimeExecutionContext) -> dict[str, str]:
    env: dict[str, str] = {
        "OPENCLAW_STATE_DIR": OPENCLAW_CONTAINER_STATE_DIR,
        "OPENCLAW_GATEWAY_TOKEN": _resolve_gateway_token(context),
        "OPENCLAW_DISABLE_BONJOUR": "1",
        "DOCKER_HOST": _isolated_docker_host(context),
    }
    model = str(context.llm_config.get("model", "")).strip()
    base_url = str(context.llm_config.get("base_url", "")).strip()
    api_key = str(context.llm_config.get("api_key", "")).strip()
    is_openrouter = _is_openrouter_context(context)
    if api_key and is_openrouter:
        env["OPENROUTER_API_KEY"] = api_key
    elif api_key:
        env["OPENAI_API_KEY"] = api_key
    if base_url and not is_openrouter:
        env["OPENAI_BASE_URL"] = base_url
    return env


def _validate_plugin(path: Path) -> None:
    _resolve_plugin_id(path)
    for required_file in ("index.js", "requirements.txt", _resolve_plugin_entrypoint(path)):
        if not (path / required_file).is_file():
            raise RuntimeError(f"OpenClaw plugin is missing {required_file}: {path}")


def _plugin_requirements_digest(path: Path) -> str:
    requirements_path = path / "requirements.txt"
    return hashlib.sha256(requirements_path.read_bytes()).hexdigest()


def _plugin_mount_args(context: RuntimeExecutionContext) -> list[str]:
    plugin_path = _resolve_plugin_path(context)
    if plugin_path is None:
        return []
    return ["-v", f"{plugin_path}:{_container_plugin_path()}:rw"]


def _ensure_plugin_runtime(context: RuntimeExecutionContext) -> dict[str, Any]:
    plugin_path = _resolve_plugin_path(context)
    if plugin_path is None:
        return {"enabled": False}

    _validate_plugin(plugin_path)
    _ensure_docker_image_available(_resolve_gateway_image(context), role="OpenClaw gateway")
    digest = _plugin_requirements_digest(plugin_path)
    venv_path = plugin_path / PLUGIN_VENV_DIRNAME
    marker_path = venv_path / PLUGIN_REQUIREMENTS_MARKER
    run_key = f"{context.output_dir.resolve()}::{plugin_path}"
    reinstall_requested = _resolve_plugin_reinstall_on_run_start(context)
    reinstall_performed = False
    if reinstall_requested and run_key not in _PLUGIN_REINSTALLED_FOR_RUNS:
        _remove_plugin_venv(context, plugin_path=plugin_path, venv_path=venv_path)
        reinstall_performed = True
    if (venv_path / "bin" / "python").is_file() and marker_path.is_file():
        if marker_path.read_text(encoding="utf-8").strip() == digest:
            if reinstall_requested:
                _PLUGIN_REINSTALLED_FOR_RUNS.add(run_key)
            return {
                "enabled": True,
                "plugin_path": str(plugin_path),
                "container_plugin_path": _container_plugin_path(),
                "venv_path": str(venv_path),
                "dependency_install": "cached",
                "reinstall_on_run_start": reinstall_requested,
                "reinstall_performed": False,
            }

    plugin_owner = _path_owner_spec(plugin_path)
    install_script = "\n".join(
        [
            "set -e",
            (
                "probe_dir=$(mktemp -d); "
                "if python3 -m venv \"$probe_dir/venv\" >/dev/null 2>&1 && "
                "\"$probe_dir/venv/bin/python\" -m pip --version >/dev/null 2>&1; then "
                "rm -rf \"$probe_dir\"; "
                "else "
                "rm -rf \"$probe_dir\"; "
                "if command -v apt-get >/dev/null 2>&1; then "
                "apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y python3-venv python3-pip; "
                "elif command -v apk >/dev/null 2>&1; then "
                "apk add --no-cache py3-pip py3-virtualenv; "
                "fi; "
                "fi"
            ),
            f"cd {shlex.quote(_container_plugin_path())}",
            f"rm -rf {shlex.quote(PLUGIN_VENV_DIRNAME)}",
            f"python3 -m venv --clear {shlex.quote(PLUGIN_VENV_DIRNAME)}",
            (
                f"{shlex.quote(PLUGIN_VENV_DIRNAME)}/bin/python "
                "-m pip install --disable-pip-version-check --no-cache-dir -r requirements.txt"
            ),
            (
                "printf '%s\\n' "
                f"{shlex.quote(digest)} "
                f"> {shlex.quote(PLUGIN_VENV_DIRNAME)}/{PLUGIN_REQUIREMENTS_MARKER}"
            ),
            f"chown -R {shlex.quote(plugin_owner)} {shlex.quote(PLUGIN_VENV_DIRNAME)}",
        ]
    )
    args = [
        "docker",
        "run",
        "--rm",
        "--user",
        "root",
        "-v",
        f"{plugin_path}:{_container_plugin_path()}:rw",
        _resolve_gateway_image(context),
        "sh",
        "-lc",
        install_script,
    ]
    result = _run_command(args)
    if result.returncode != 0:
        raise RuntimeError(
            "Failed to prepare the OpenClaw SOMA plugin runtime. "
            "The OpenClaw gateway image must provide python3 and a package manager capable of installing venv/pip. "
            f"{(result.stderr or result.stdout or '').strip()}"
        )

    _PLUGIN_REINSTALLED_FOR_RUNS.add(run_key)
    return {
        "enabled": True,
        "plugin_path": str(plugin_path),
        "container_plugin_path": _container_plugin_path(),
        "venv_path": str(venv_path),
        "dependency_install": "installed",
        "reinstall_on_run_start": reinstall_requested,
        "reinstall_performed": reinstall_performed,
    }


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def _jsonl_rows(path: Path) -> list[tuple[int, dict[str, Any]]]:
    if not path.is_file():
        return []

    rows: list[tuple[int, dict[str, Any]]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            raw_line = line.strip()
            if not raw_line:
                continue
            try:
                payload = json.loads(raw_line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                rows.append((line_number, payload))
    return rows


def _finite_number(value: Any) -> int | float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)) and math.isfinite(value):
        return value
    return None


def _token_count(value: Any) -> int | None:
    number = _finite_number(value)
    if number is None or number < 0:
        return None
    return int(number)


def _usage_number(value: Any) -> int | float | None:
    number = _finite_number(value)
    if number is None:
        return None
    return int(number) if isinstance(number, int) or float(number).is_integer() else number


def _first_token_count(mapping: dict[str, Any], *keys: str) -> int | None:
    for key in keys:
        value = _token_count(mapping.get(key))
        if value is not None:
            return value
    return None


def _first_usage_number(mapping: dict[str, Any], *keys: str) -> int | float | None:
    for key in keys:
        value = _usage_number(mapping.get(key))
        if value is not None:
            return value
    return None


def _normalize_cost_payload(cost: Any) -> dict[str, int | float] | None:
    if not isinstance(cost, dict):
        return None

    aliases = {
        "input": ("input", "inputCost", "input_cost"),
        "output": ("output", "outputCost", "output_cost"),
        "cache_read": ("cacheRead", "cache_read", "cacheReadCost", "cache_read_cost"),
        "cache_write": ("cacheWrite", "cache_write", "cacheWriteCost", "cache_write_cost"),
        "total": ("total", "totalCost", "total_cost"),
    }
    normalized = {
        field: value
        for field, keys in aliases.items()
        if (value := _first_usage_number(cost, *keys)) is not None
    }
    return normalized or None


def _normalize_provider_usage(usage: Any) -> dict[str, Any] | None:
    if not isinstance(usage, dict):
        return None

    input_tokens = _first_token_count(usage, "input", "input_tokens", "inputTokens")
    output_tokens = _first_token_count(usage, "output", "output_tokens", "outputTokens")
    cache_read_tokens = _first_token_count(
        usage,
        "cacheRead",
        "cache_read",
        "cache_read_input_tokens",
        "cacheReadInputTokens",
    )
    cache_write_tokens = _first_token_count(
        usage,
        "cacheWrite",
        "cache_write",
        "cache_creation_input_tokens",
        "cacheCreationInputTokens",
    )
    total_tokens = _first_token_count(usage, "totalTokens", "total_tokens", "total")

    if total_tokens is None:
        component_values = [
            value
            for value in (input_tokens, output_tokens, cache_read_tokens, cache_write_tokens)
            if value is not None
        ]
        if component_values:
            total_tokens = sum(component_values)

    if not any(
        value is not None
        for value in (input_tokens, output_tokens, cache_read_tokens, cache_write_tokens, total_tokens)
    ):
        return None

    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cache_read_tokens": cache_read_tokens,
        "cache_write_tokens": cache_write_tokens,
        "total_tokens": total_tokens,
        "cost": _normalize_cost_payload(usage.get("cost")),
    }


def _sum_usage_values(usages: list[dict[str, Any]]) -> dict[str, Any]:
    totals: dict[str, Any] = {
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_read_tokens": 0,
        "cache_write_tokens": 0,
        "total_tokens": 0,
    }
    for usage in usages:
        for key in ("input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens", "total_tokens"):
            value = usage.get(key)
            if isinstance(value, int):
                totals[key] += value

        cost = usage.get("cost")
        if isinstance(cost, dict):
            cost_totals = totals.setdefault("cost", {})
            for key, value in cost.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    cost_totals[key] = cost_totals.get(key, 0) + value

    return totals


def _assistant_texts_count(assistant_texts: Any) -> int | None:
    if not isinstance(assistant_texts, list):
        return None
    if assistant_texts and isinstance(assistant_texts[-1], dict) and assistant_texts[-1].get("truncated") is True:
        original_length = _token_count(assistant_texts[-1].get("originalLength"))
        if original_length is not None:
            return original_length
    return len(assistant_texts)


def _session_file_path(context: RuntimeExecutionContext, *, agent_id: str, session_id: str) -> Path:
    return _state_dir(context) / "agents" / agent_id / "sessions" / f"{session_id}.jsonl"


def _session_trajectory_file_path(context: RuntimeExecutionContext, *, agent_id: str, session_id: str) -> Path:
    return _state_dir(context) / "agents" / agent_id / "sessions" / f"{session_id}.trajectory.jsonl"


def _session_index_path(context: RuntimeExecutionContext, *, agent_id: str) -> Path:
    return _state_dir(context) / "agents" / agent_id / "sessions" / "sessions.json"


def _plugin_log_path(context: RuntimeExecutionContext, *, session_id: str) -> Path:
    return _state_dir(context) / "plugin-artifacts" / "logs" / f"{session_id}.jsonl"


def _collect_transcript_token_usage(session_file: Path) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    for line_number, row in _jsonl_rows(session_file):
        if row.get("type") != "message":
            continue
        message = row.get("message")
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        usage = _normalize_provider_usage(message.get("usage") or row.get("usage"))
        if usage is None:
            continue
        entries.append(
            {
                "line": line_number,
                "message_id": row.get("id"),
                "timestamp": row.get("timestamp"),
                "usage": usage,
            }
        )

    latest = entries[-1] if entries else None
    return {
        "session_file": str(session_file),
        "model_calls_count": len(entries),
        "assistant_usage_count": len(entries),
        "total": _sum_usage_values([entry["usage"] for entry in entries]),
        "latest": latest,
    }


def _collect_trajectory_token_usage(trajectory_file: Path) -> dict[str, Any]:
    latest_completion: dict[str, Any] | None = None
    latest_artifacts: dict[str, Any] | None = None
    completion_count = 0

    for line_number, row in _jsonl_rows(trajectory_file):
        row_type = row.get("type")
        if row_type == "model.completed":
            completion_count += 1
            data = row.get("data")
            if not isinstance(data, dict):
                continue

            usage = _normalize_provider_usage(data.get("usage"))
            if usage is None:
                continue

            prompt_cache_usage: dict[str, Any] | None = None
            prompt_cache = data.get("promptCache")
            if isinstance(prompt_cache, dict):
                prompt_cache_usage = _normalize_provider_usage(prompt_cache.get("lastCallUsage"))

            assistant_texts = data.get("assistantTexts")
            model_calls_count = _assistant_texts_count(assistant_texts)

            latest_completion = {
                "line": line_number,
                "seq": _token_count(row.get("seq")),
                "timestamp": row.get("ts") or row.get("timestamp"),
                "usage": usage,
                "last_call_usage": prompt_cache_usage,
                "model_calls_count": model_calls_count,
            }
            continue

        if row_type != "trace.artifacts":
            continue

        data = row.get("data")
        if not isinstance(data, dict):
            continue

        usage = _normalize_provider_usage(data.get("usage"))
        if usage is None:
            continue

        prompt_cache_usage = None
        prompt_cache = data.get("promptCache")
        if isinstance(prompt_cache, dict):
            prompt_cache_usage = _normalize_provider_usage(prompt_cache.get("lastCallUsage"))

        assistant_texts = data.get("assistantTexts")
        model_calls_count = _assistant_texts_count(assistant_texts)

        latest_artifacts = {
            "line": line_number,
            "seq": _token_count(row.get("seq")),
            "timestamp": row.get("ts") or row.get("timestamp"),
            "usage": usage,
            "last_call_usage": prompt_cache_usage,
            "model_calls_count": model_calls_count,
        }

    preferred_usage = latest_artifacts or latest_completion
    total_usage = preferred_usage["usage"] if preferred_usage else _sum_usage_values([])
    model_calls_count = (
        preferred_usage.get("model_calls_count")
        if isinstance(preferred_usage, dict)
        else None
    )
    if not isinstance(model_calls_count, int) and isinstance(latest_completion, dict):
        completion_calls = latest_completion.get("model_calls_count")
        if isinstance(completion_calls, int) and completion_calls >= 0:
            model_calls_count = completion_calls
    if not isinstance(model_calls_count, int) and completion_count > 0:
        model_calls_count = completion_count

    if latest_artifacts is not None:
        tokens_source = "trajectory.trace.artifacts.data.usage"
    elif latest_completion is not None:
        tokens_source = "trajectory.model.completed.data.usage"
    else:
        tokens_source = "trajectory.none"

    if isinstance(preferred_usage, dict) and isinstance(preferred_usage.get("model_calls_count"), int):
        if latest_artifacts is not None:
            calls_source = "trajectory.trace.artifacts.data.assistantTexts"
        elif latest_completion is not None:
            calls_source = "trajectory.model.completed.data.assistantTexts"
        else:
            calls_source = "trajectory.none"
    elif isinstance(latest_completion, dict) and isinstance(latest_completion.get("model_calls_count"), int):
        calls_source = "trajectory.model.completed.data.assistantTexts"
    elif completion_count > 0:
        calls_source = "trajectory.model.completed.count"
    else:
        calls_source = "trajectory.none"

    return {
        "trajectory_file": str(trajectory_file),
        "model_completed_count": completion_count,
        "model_calls_count": model_calls_count if isinstance(model_calls_count, int) and model_calls_count >= 0 else None,
        "assistant_usage_count": model_calls_count if isinstance(model_calls_count, int) and model_calls_count >= 0 else None,
        "total": total_usage,
        "latest": preferred_usage,
        "last_call_usage": (
            preferred_usage.get("last_call_usage")
            if isinstance(preferred_usage, dict)
            else (latest_completion.get("last_call_usage") if isinstance(latest_completion, dict) else None)
        ),
        "tokens_source": tokens_source,
        "calls_source": calls_source,
    }


def _collect_session_index_token_usage(
    context: RuntimeExecutionContext,
    *,
    agent_id: str,
    session_id: str,
) -> dict[str, Any] | None:
    path = _session_index_path(context, agent_id=agent_id)
    if not path.is_file():
        return None

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"path": str(path), "error": "invalid-json"}
    if not isinstance(payload, dict):
        return {"path": str(path), "error": "invalid-payload"}

    session_key = _build_sandbox_session_key(agent_id=agent_id, session_id=session_id)
    entry = payload.get(session_key)
    if not isinstance(entry, dict):
        entry = next(
            (
                candidate
                for candidate in payload.values()
                if isinstance(candidate, dict) and candidate.get("sessionId") == session_id
            ),
            None,
        )
    if not isinstance(entry, dict):
        return {"path": str(path), "error": "missing-session-entry"}

    return {
        "path": str(path),
        "total_tokens": _token_count(entry.get("totalTokens")),
        "total_tokens_fresh": entry.get("totalTokensFresh") if isinstance(entry.get("totalTokensFresh"), bool) else None,
        "message_count": _token_count(entry.get("messageCount")),
        "compaction_count": _token_count(entry.get("compactionCount")),
        "updated_at": _token_count(entry.get("updatedAt")),
    }


def _collect_plugin_token_usage(context: RuntimeExecutionContext, *, session_id: str) -> dict[str, Any] | None:
    path = _plugin_log_path(context, session_id=session_id)
    if not path.is_file():
        return None

    latest_evaluate: dict[str, Any] | None = None
    for _, row in _jsonl_rows(path):
        if row.get("event") == "proactive.evaluate":
            latest_evaluate = row

    if latest_evaluate is None:
        return {"path": str(path), "evaluate_count": 0}

    return {
        "path": str(path),
        "estimated_tokens": _token_count(latest_evaluate.get("estimatedTokens")),
        "token_source": latest_evaluate.get("tokenSource"),
        "threshold_tokens": _token_count(latest_evaluate.get("thresholdTokens")),
        "runtime_current_token_count": _token_count(latest_evaluate.get("runtimeCurrentTokenCount")),
        "transcript_real_token_count": _token_count(latest_evaluate.get("transcriptRealTokenCount")),
        "transcript_token_line": _token_count(latest_evaluate.get("transcriptTokenLine")),
        "fresh_session_token_count": _token_count(latest_evaluate.get("freshSessionTokenCount")),
    }


def _collect_openclaw_token_usage(
    context: RuntimeExecutionContext,
    *,
    agent_id: str,
    session_id: str,
) -> dict[str, Any]:
    session_file = _session_file_path(context, agent_id=agent_id, session_id=session_id)
    transcript_usage = _collect_transcript_token_usage(session_file)
    trajectory_file = _session_trajectory_file_path(context, agent_id=agent_id, session_id=session_id)
    trajectory_usage = _collect_trajectory_token_usage(trajectory_file)
    trajectory_total_tokens = _token_count((trajectory_usage.get("total") or {}).get("total_tokens"))
    transcript_total_tokens = _token_count((transcript_usage.get("total") or {}).get("total_tokens"))
    use_trajectory = bool(
        _token_count(trajectory_usage.get("model_completed_count"))
        and isinstance(trajectory_usage.get("total"), dict)
        and trajectory_total_tokens is not None
        and (
            trajectory_total_tokens > 0
            or transcript_total_tokens in {None, 0}
        )
    )

    base_usage = trajectory_usage if use_trajectory else transcript_usage
    calls_source = (
        str(base_usage.get("calls_source"))
        if use_trajectory and isinstance(base_usage.get("calls_source"), str)
        else ("trajectory.model.completed.data.assistantTexts" if use_trajectory else "transcript.assistant_message_usage")
    )
    tokens_source = (
        str(base_usage.get("tokens_source"))
        if use_trajectory and isinstance(base_usage.get("tokens_source"), str)
        else ("trajectory.model.completed.data.usage.total" if use_trajectory else "transcript.assistant_message_usage_sum")
    )
    if use_trajectory and _token_count(base_usage.get("model_calls_count")) is None:
        fallback_calls = _token_count(transcript_usage.get("model_calls_count"))
        if fallback_calls is not None:
            base_usage["model_calls_count"] = fallback_calls
            base_usage["assistant_usage_count"] = fallback_calls
            calls_source = "transcript.assistant_message_usage_fallback"

    payload = {
        "source": "openclaw.trajectory" if use_trajectory else "openclaw.transcript",
        "counts_source": {
            "tokens": tokens_source,
            "calls": calls_source,
        },
        "session_file": transcript_usage.get("session_file"),
        **base_usage,
        "trajectory_fallback": transcript_usage if use_trajectory else None,
        "session_index": _collect_session_index_token_usage(
            context,
            agent_id=agent_id,
            session_id=session_id,
        ),
        "plugin": _collect_plugin_token_usage(context, session_id=session_id),
    }
    return payload


def _append_unique_string(values: list[Any], value: str) -> None:
    if value not in values:
        values.append(value)


def _ensure_plugin_config(context: RuntimeExecutionContext, config: dict[str, Any]) -> dict[str, Any]:
    plugin_path = _resolve_plugin_path(context)
    if plugin_path is None:
        plugins = config.get("plugins")
        if isinstance(plugins, dict):
            slots = plugins.get("slots")
            if isinstance(slots, dict) and slots.get("contextEngine") == SOMA_PLUGIN_ID:
                slots["contextEngine"] = "legacy"
            entries = plugins.get("entries")
            if isinstance(entries, dict):
                entry = entries.get(SOMA_PLUGIN_ID)
                if isinstance(entry, dict):
                    entry["enabled"] = False
        return {"enabled": False}

    _validate_plugin(plugin_path)
    plugin_id = _resolve_plugin_id(plugin_path)
    container_plugin_path = _container_plugin_path()
    container_python_bin = f"{container_plugin_path}/{PLUGIN_VENV_DIRNAME}/bin/python"
    container_script_path = f"{container_plugin_path}/{_resolve_plugin_entrypoint(plugin_path)}"

    plugins = config.setdefault("plugins", {})
    if not isinstance(plugins, dict):
        plugins = {}
        config["plugins"] = plugins
    plugins["enabled"] = True

    load = plugins.setdefault("load", {})
    if not isinstance(load, dict):
        load = {}
        plugins["load"] = load
    load_paths = load.setdefault("paths", [])
    if not isinstance(load_paths, list):
        load_paths = []
        load["paths"] = load_paths
    _append_unique_string(load_paths, container_plugin_path)

    deny = plugins.get("deny")
    if isinstance(deny, list):
        plugins["deny"] = [value for value in deny if value != SOMA_PLUGIN_ID]

    allow = plugins.get("allow")
    if isinstance(allow, list) and allow:
        _append_unique_string(allow, plugin_id)

    slots = plugins.setdefault("slots", {})
    if not isinstance(slots, dict):
        slots = {}
        plugins["slots"] = slots
    slots["contextEngine"] = plugin_id

    entries = plugins.setdefault("entries", {})
    if not isinstance(entries, dict):
        entries = {}
        plugins["entries"] = entries
    plugin_entry = entries.setdefault(plugin_id, {})
    if not isinstance(plugin_entry, dict):
        plugin_entry = {}
        entries[plugin_id] = plugin_entry
    plugin_entry["enabled"] = True

    plugin_entry.pop("config", None)

    return {
        "enabled": True,
        "plugin_id": plugin_id,
        "plugin_path": str(plugin_path),
        "container_plugin_path": container_plugin_path,
        "python_bin": container_python_bin,
        "script_path": container_script_path,
    }


def _build_runtime_env_script() -> str:
    lines = [
        "#!/bin/sh",
        "for candidate in /opt/conda/envs/testbed/bin /opt/miniconda3/envs/testbed/bin /opt/conda/bin /opt/miniconda3/bin; do",
        '  case ":$PATH:" in',
        '    *":${candidate}:"*) ;;',
        '    *) [ -d "$candidate" ] && PATH="$candidate:$PATH" ;;',
        "  esac",
        "done",
        "export PATH",
        'if [ -f /opt/conda/etc/profile.d/conda.sh ]; then',
        '  . /opt/conda/etc/profile.d/conda.sh >/dev/null 2>&1 || true',
        '  conda activate testbed >/dev/null 2>&1 || true',
        'elif [ -f /opt/miniconda3/etc/profile.d/conda.sh ]; then',
        '  . /opt/miniconda3/etc/profile.d/conda.sh >/dev/null 2>&1 || true',
        '  conda activate testbed >/dev/null 2>&1 || true',
        "fi",
    ]
    return "\n".join(lines) + "\n"


def _build_runtime_env_setup_command(context: RuntimeExecutionContext) -> str:
    script_path = shlex.quote(_container_runtime_env_script_path(context))
    script_body = _build_runtime_env_script()
    quoted_body = shlex.quote(script_body)
    return f"printf %s {quoted_body} > {script_path} && chmod 755 {script_path}"


def _agents_file_content(context: RuntimeExecutionContext, *, repo_path: str) -> str:
    return (
        "\n".join(
            [
                f"You are the dedicated agent for benchmark problem {context.instance.benchmark_id}.",
                f"The editable repository is mounted at {repo_path}.",
                "The workspace root is the repository root for this benchmark run.",
                "For read/edit/write file tools, use paths relative to the repository root, for example django_migration_linter/migration_linter.py.",
                "Do not pass absolute /workspace/... paths to file tools; reserve absolute paths for shell commands only when necessary.",
                "The benchmark image already provides the runtime dependencies for that repository.",
                "Your shell already receives the benchmark PATH/bootstrap automatically; do not manually source the helper script unless you are debugging shell startup.",
                "Prefer python3 over python when invoking Python explicitly.",
                "Keep changes focused on the requested benchmark task and validate your work when possible.",
            ]
        )
        + "\n"
    )


def _tools_file_content() -> str:
    return (
        "Use file tools with repository-relative paths, not absolute /workspace paths. "
        "Use shell tools from the repo root workdir.\n"
    )


def _identity_file_content(context: RuntimeExecutionContext) -> str:
    return f"name: SOMA Benchmark Agent {context.instance.benchmark_id}\nrole: coding-agent\n"


def _build_write_file_setup_command(path: str, content: str) -> str:
    return f"printf %s {shlex.quote(content)} > {shlex.quote(path)}"


def _build_workspace_bootstrap_setup_commands(context: RuntimeExecutionContext) -> list[str]:
    repo_path = _container_repo_path(context)
    commands = [
        _build_write_file_setup_command(f"{repo_path}/AGENTS.md", _agents_file_content(context, repo_path=repo_path)),
        _build_write_file_setup_command(f"{repo_path}/TOOLS.md", _tools_file_content()),
        _build_write_file_setup_command(f"{repo_path}/IDENTITY.md", _identity_file_content(context)),
        _build_runtime_env_setup_command(context),
    ]
    commands.append(
        f'if [ -d {shlex.quote(repo_path)}/.git ]; then '
        f'mkdir -p {shlex.quote(repo_path)}/.git/info && '
        f'for name in AGENTS.md TOOLS.md IDENTITY.md {OPENCLAW_RUNTIME_ENV_SCRIPT_NAME}; do '
        f'grep -qxF "$name" {shlex.quote(repo_path)}/.git/info/exclude || printf "%s\\n" "$name" >> {shlex.quote(repo_path)}/.git/info/exclude; '
        'done; '
        'fi'
    )
    return commands


def _build_sandbox_env(context: RuntimeExecutionContext) -> dict[str, str]:
    script_path = _container_runtime_env_script_path(context)
    return {
        "BASH_ENV": script_path,
        "ENV": script_path,
    }


def _ensure_workspace_files(context: RuntimeExecutionContext, *, repo_path: str) -> None:
    workspace_dir = _agent_workspace_dir(context)
    _reset_directory(workspace_dir)
    (workspace_dir / "AGENTS.md").write_text(_agents_file_content(context, repo_path=repo_path), encoding="utf-8")
    (workspace_dir / "TOOLS.md").write_text(_tools_file_content(), encoding="utf-8")
    (workspace_dir / "IDENTITY.md").write_text(_identity_file_content(context), encoding="utf-8")


def _load_config(context: RuntimeExecutionContext) -> dict[str, Any]:
    path = _config_path(context)
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        return payload
    return {}


def _build_provider_config(context: RuntimeExecutionContext) -> dict[str, Any]:
    base_url = str(context.llm_config.get("base_url", "")).strip()
    api_key = str(context.llm_config.get("api_key", "")).strip()
    providers: dict[str, Any] = {}
    is_openrouter = _is_openrouter_context(context)
    model = _resolve_openclaw_model(context)
    use_openai_proxy_without_api_key = _uses_openai_compatible_proxy_without_api_key(context)
    is_openai_like = not is_openrouter and (
        not model or model.startswith("openai/") or "/" not in model or use_openai_proxy_without_api_key
    )

    if is_openai_like and (api_key or use_openai_proxy_without_api_key):
        model_id = model.removeprefix("openai/") if model.startswith("openai/") else model
        openai_provider: dict[str, Any] = {
            "api": "openai-completions",
            "baseUrl": base_url or "https://api.openai.com/v1",
            "models": [{"id": model_id, "name": model_id}] if model_id else [],
        }
        if api_key:
            openai_provider["apiKey"] = "${OPENAI_API_KEY}"
        elif use_openai_proxy_without_api_key:
            openai_provider["apiKey"] = "sk-local-proxy-placeholder"
        run_id_header_value = _resolve_run_id_header_value(context)
        if run_id_header_value:
            openai_provider["headers"] = {"X-Run-Id": run_id_header_value}
        providers["openai"] = openai_provider

    return providers


def _build_agent_entry(context: RuntimeExecutionContext) -> dict[str, Any]:
    agent_entry = {
        "id": _build_agent_id(context),
        "workspace": str(_repo_dir(context).resolve()),
        "model": _resolve_openclaw_model(context),
        "sandbox": {
            "mode": "all",
            "backend": "docker",
            "scope": "session",
            "workspaceAccess": "rw",
            "docker": {
                "image": _resolve_benchmark_image(context),
                "workdir": _container_repo_path(context),
                "network": _resolve_private_network_name(context),
                "readOnlyRoot": False,
                "dangerouslyAllowExternalBindSources": True,
                "dangerouslyAllowReservedContainerTargets": True,
                "env": _build_sandbox_env(context),
                "binds": [
                    f"{_artifacts_dir(context).resolve()}:{OPENCLAW_CONTAINER_ARTIFACTS_PATH}:rw",
                ],
            },
        },
    }
    workspace_user = _resolve_workspace_container_user(context)
    if workspace_user is not None:
        agent_entry["sandbox"]["docker"]["user"] = workspace_user
    agent_entry["sandbox"]["docker"]["setupCommand"] = _build_prepare_repo_command(context)
    return agent_entry


def _configure_agent_config_common(context: RuntimeExecutionContext, config: dict[str, Any]) -> None:
    gateway = config.setdefault("gateway", {})
    if not isinstance(gateway, dict):
        gateway = {}
        config["gateway"] = gateway
    gateway.update(
        {
            "mode": "local",
            "bind": "loopback",
            "auth": {"token": "${OPENCLAW_GATEWAY_TOKEN}"},
        }
    )
    tools = config.setdefault("tools", {})
    if not isinstance(tools, dict):
        tools = {}
        config["tools"] = tools
    tools.setdefault("profile", "coding")
    deny = tools.setdefault("deny", [])
    if not isinstance(deny, list):
        deny = []
        tools["deny"] = deny
    _append_unique_string(deny, "group:web")
    web_tools = tools.setdefault("web", {})
    if not isinstance(web_tools, dict):
        web_tools = {}
        tools["web"] = web_tools
    search_tools = web_tools.setdefault("search", {})
    if not isinstance(search_tools, dict):
        search_tools = {}
        web_tools["search"] = search_tools
    search_tools["enabled"] = False

    agents = config.setdefault("agents", {})
    if not isinstance(agents, dict):
        agents = {}
        config["agents"] = agents
    agent_defaults = agents.setdefault("defaults", {})
    if not isinstance(agent_defaults, dict):
        agent_defaults = {}
        agents["defaults"] = agent_defaults
    agent_defaults.setdefault("workspace", f"{OPENCLAW_CONTAINER_STATE_DIR}/workspace-default")
    agent_defaults["skipBootstrap"] = True
    heartbeat = agent_defaults.setdefault("heartbeat", {})
    if not isinstance(heartbeat, dict):
        heartbeat = {}
        agent_defaults["heartbeat"] = heartbeat
    heartbeat["every"] = OPENCLAW_HEARTBEAT_DISABLED_INTERVAL
    agents_list = agents.setdefault("list", [])
    if not isinstance(agents_list, list):
        agents_list = []
        agents["list"] = agents_list

    providers = _build_provider_config(context)
    if providers:
        models = config.setdefault("models", {})
        models.setdefault("providers", {})
        models["providers"].update(providers)

    env = config.setdefault("env", {})
    if not isinstance(env, dict):
        env = {}
        config["env"] = env
    env_vars = env.setdefault("vars", {})
    if not isinstance(env_vars, dict):
        env_vars = {}
        env["vars"] = env_vars
    gateway_env = _gateway_process_env(context)
    for key in ("OPENROUTER_API_KEY", "OPENAI_API_KEY", "OPENAI_BASE_URL"):
        if key in gateway_env:
            env_vars[key] = gateway_env[key]

    _ensure_plugin_config(context, config)


def _upsert_agent_entry(config: dict[str, Any], agent_entry: dict[str, Any]) -> None:
    agents = config.setdefault("agents", {})
    if not isinstance(agents, dict):
        agents = {}
        config["agents"] = agents
    agents_list = agents.setdefault("list", [])
    if not isinstance(agents_list, list):
        agents_list = []
        agents["list"] = agents_list

    existing_index: int | None = None
    for index, candidate in enumerate(agents_list):
        if isinstance(candidate, dict) and candidate.get("id") == agent_entry["id"]:
            existing_index = index
            break
    if existing_index is None:
        agents_list.append(agent_entry)
    else:
        agents_list[existing_index] = agent_entry


def _ensure_agent_config(context: RuntimeExecutionContext) -> Path:
    state_dir = _state_dir(context)
    state_dir.mkdir(parents=True, exist_ok=True)
    config = _load_config(context)
    _configure_agent_config_common(context, config)
    _upsert_agent_entry(config, _build_agent_entry(context))
    return _write_json(_config_path(context), config)


def _ensure_agent_configs(contexts: list[RuntimeExecutionContext]) -> Path:
    if not contexts:
        raise RuntimeError("OpenClaw batch execution requires at least one context.")

    primary = contexts[0]
    state_dir = _state_dir(primary)
    state_dir.mkdir(parents=True, exist_ok=True)
    config = _load_config(primary)
    _configure_agent_config_common(primary, config)
    for context in contexts:
        _upsert_agent_entry(config, _build_agent_entry(context))
    return _write_json(_config_path(primary), config)


def _collect_prepare_repo_commands(context: RuntimeExecutionContext) -> list[str]:
    repo_path = _container_repo_path(context)
    commands = [
        f"mkdir -p {shlex.quote(repo_path)}",
        f"mkdir -p {shlex.quote(OPENCLAW_CONTAINER_ARTIFACTS_PATH)}",
        (
            f'if [ -d /testbed ]; then '
            f'find {shlex.quote(repo_path)} -mindepth 1 -maxdepth 1 -exec rm -rf {{}} + ; '
            f'cp -r /testbed/. {shlex.quote(repo_path)}/ ; '
            'else echo "OpenClaw sandbox expected /testbed in the benchmark image" >&2; exit 1; fi'
        ),
        (
            f'if [ -d {shlex.quote(repo_path)}/.git ]; then '
            f'git -C {shlex.quote(repo_path)} reset --hard && '
            f'git -C {shlex.quote(repo_path)} clean -fdx; '
            'fi'
        ),
    ]

    if context.instance.base_commit:
        commands.append(
            f'if [ -d {shlex.quote(repo_path)}/.git ]; then '
            f'git -C {shlex.quote(repo_path)} -c advice.detachedHead=false checkout -f -q {shlex.quote(context.instance.base_commit)}; '
            'fi'
        )

    commands.extend(_build_workspace_bootstrap_setup_commands(context))

    return commands


def _build_prepare_repo_command(context: RuntimeExecutionContext) -> str:
    return "\n".join(["set -e"] + _collect_prepare_repo_commands(context))


def _build_message(context: RuntimeExecutionContext, *, repo_path: str) -> str:
    user_prompt = context.instance.user_prompt.strip()
    resume_prompt = context.instance.resume_prompt.strip()
    parts = [
        f"Solve benchmark problem {context.instance.benchmark_id} for repository {context.instance.repo}.",
        f"The editable repository is available at {repo_path}.",
        "For read/edit/write file tools, always use paths relative to the repository root, not absolute /workspace paths.",
        "The benchmark container already provides the repository dependencies; avoid reinstalling them unless the task explicitly requires it.",
        f"Write any auxiliary outputs under {OPENCLAW_CONTAINER_ARTIFACTS_PATH} if needed.",
        "Inspect the repo state before making changes and keep the solution focused on the benchmark task.",
    ]
    if user_prompt:
        parts.extend(["", "Original task prompt:", user_prompt])
    if resume_prompt and resume_prompt != user_prompt:
        parts.extend(["", "Current benchmark instruction:", resume_prompt])
    if context.instance.expected_next_action.strip():
        parts.append(f"Expected next action: {context.instance.expected_next_action.strip()}")
    if context.instance.expected_next_stage.strip():
        parts.append(f"Target stage: {context.instance.expected_next_stage.strip()}")
    return "\n".join(parts)


def _cli_base_args(context: RuntimeExecutionContext, *, gateway_name: str) -> list[str]:
    state_dir = _state_dir(context)
    openclaw_user = _resolve_openclaw_user(context)
    args = [
        "docker",
        "run",
        "--rm",
        "--user",
        openclaw_user,
        "--network",
        f"container:{gateway_name}",
        "-v",
        f"{state_dir}:{OPENCLAW_CONTAINER_STATE_DIR}",
    ]
    args.extend(_openclaw_container_hardening_args())
    args.extend(_plugin_mount_args(context))
    docker_binary = _host_docker_binary()
    if docker_binary:
        args.extend(["-v", f"{docker_binary}:/usr/local/bin/docker:ro"])
    env_payload = _gateway_process_env(context)
    for key, value in env_payload.items():
        args.extend(["-e", f"{key}={value}"])
    args.append(_resolve_gateway_image(context))
    args.append("openclaw")
    return args


def _run_openclaw_cli(
    context: RuntimeExecutionContext,
    *,
    gateway_name: str,
    cli_args: list[str],
) -> subprocess.CompletedProcess[str]:
    return _run_command(_cli_base_args(context, gateway_name=gateway_name) + cli_args)


def _start_gateway_container(
    context: RuntimeExecutionContext,
    *,
    restart_existing: bool = False,
) -> tuple[str, bool]:
    gateway_name = _resolve_gateway_name(context)
    if _docker_container_running(gateway_name):
        if not restart_existing:
            _ensure_isolated_daemon(context)
            _docker_connect_container_to_network(
                container_name=gateway_name,
                network_name=_resolve_isolated_control_network_name(context),
            )
            return gateway_name, False
        _docker_remove_container(gateway_name)
    else:
        _docker_remove_container(gateway_name)

    _ensure_docker_image_available(_resolve_gateway_image(context), role="OpenClaw gateway")
    _ensure_isolated_daemon(context)
    state_dir = _state_dir(context)
    problems_root = _gateway_workspace_root(context)
    network_name, network_created = _ensure_private_network(context)
    openclaw_user = _resolve_openclaw_user(context)
    state_dir.mkdir(parents=True, exist_ok=True)
    problems_root.mkdir(parents=True, exist_ok=True)
    args = [
        "docker",
        "run",
        "-d",
        "--rm",
        "--user",
        openclaw_user,
        "--name",
        gateway_name,
        "--network",
        network_name,
        "-v",
        f"{state_dir}:{OPENCLAW_CONTAINER_STATE_DIR}",
        "-v",
        f"{problems_root}:{OPENCLAW_CONTAINER_PROBLEMS_ROOT}",
    ]
    args.extend(_openclaw_container_hardening_args())
    if str(problems_root) != OPENCLAW_CONTAINER_PROBLEMS_ROOT:
        args.extend(["-v", f"{problems_root}:{problems_root}"])
    args.extend(_plugin_mount_args(context))
    docker_binary = _host_docker_binary()
    if docker_binary:
        args.extend(["-v", f"{docker_binary}:/usr/local/bin/docker:ro"])
    env_payload = _gateway_process_env(context)
    for key, value in env_payload.items():
        args.extend(["-e", f"{key}={value}"])
    args.extend(
        [
            _resolve_gateway_image(context),
            "openclaw",
            "gateway",
            "--port",
            str(_resolve_gateway_port(context)),
            "--bind",
            "loopback",
        ]
    )
    result = _run_command(args)
    if result.returncode != 0:
        _teardown_isolated_daemon(context)
        _teardown_isolated_control_network(context)
        if network_created:
            _docker_remove_network(network_name)
        raise RuntimeError(
            "Failed to start the shared OpenClaw gateway container: "
            f"{(result.stderr or result.stdout or '').strip()}"
        )
    try:
        _docker_connect_container_to_network(
            container_name=gateway_name,
            network_name=_resolve_isolated_control_network_name(context),
        )
    except Exception:
        _docker_remove_container(gateway_name)
        _teardown_isolated_daemon(context)
        _teardown_isolated_control_network(context)
        if network_created:
            _docker_remove_network(network_name)
        raise
    return gateway_name, True


def _wait_for_gateway_ready(context: RuntimeExecutionContext, *, gateway_name: str) -> None:
    last_error = ""
    for _ in range(OPENCLAW_GATEWAY_START_RETRIES):
        if not _docker_container_running(gateway_name):
            inspect_result = _run_command(["docker", "logs", gateway_name])
            raise RuntimeError(
                "OpenClaw gateway container exited before becoming ready: "
                f"{(inspect_result.stderr or inspect_result.stdout or '').strip()}"
            )
        probe = _run_openclaw_cli(context, gateway_name=gateway_name, cli_args=["health", "--json"])
        if probe.returncode == 0:
            return
        last_error = (probe.stderr or probe.stdout or "").strip()
        time.sleep(OPENCLAW_GATEWAY_START_SLEEP_SECONDS)
    raise RuntimeError(f"OpenClaw gateway did not become ready: {last_error}")


def _wait_for_agent_registered(
    context: RuntimeExecutionContext,
    *,
    gateway_name: str,
    agent_id: str,
) -> None:
    last_error = ""
    for _ in range(OPENCLAW_GATEWAY_START_RETRIES):
        result = _run_openclaw_cli(context, gateway_name=gateway_name, cli_args=["agents", "list", "--json"])
        if result.returncode == 0:
            try:
                payload = json.loads(result.stdout or "{}")
            except json.JSONDecodeError:
                payload = {}
            agents = payload.get("agents") if isinstance(payload, dict) else payload if isinstance(payload, list) else None
            if isinstance(agents, list) and any(
                isinstance(agent, dict) and agent.get("id") == agent_id for agent in agents
            ):
                return
        last_error = (result.stderr or result.stdout or "").strip()
        time.sleep(OPENCLAW_GATEWAY_START_SLEEP_SECONDS)
    raise RuntimeError(f"OpenClaw gateway did not register benchmark agent {agent_id!r}: {last_error}")


def _write_logs(run_dir: Path, *, stdout: str, stderr: str) -> tuple[Path, Path]:
    stdout_path = run_dir / "stdout.log"
    stderr_path = run_dir / "stderr.log"
    stdout_path.write_text(stdout, encoding="utf-8")
    stderr_path.write_text(stderr, encoding="utf-8")
    return stdout_path, stderr_path


def _error_summary(stderr: str, stdout: str) -> str:
    for candidate in (stderr, stdout):
        normalized = " ".join(candidate.split())
        if normalized:
            return normalized[:400]
    return "OpenClaw execution failed"


def _build_request_metadata(
    context: RuntimeExecutionContext,
    *,
    agent_id: str,
    session_id: str,
    gateway_name: str,
    repo_path: str,
) -> dict[str, Any]:
    return {
        "benchmark_id": context.instance.benchmark_id,
        "instance_id": context.instance.instance_id,
        "agent_id": agent_id,
        "session_id": session_id,
        "gateway_name": gateway_name,
        "gateway_image": _resolve_gateway_image(context),
        "benchmark_image": _resolve_benchmark_image(context),
        "repo": context.instance.repo,
        "repo_path": repo_path,
        "state_dir": str(_state_dir(context)),
        "workspace_dir": str(_agent_workspace_dir(context)),
        "problem_root": str(_problem_root(context)),
        "runtime_options": _resolve_runtime_options(context),
        "plugin": _ensure_plugin_config(context, _load_config(context)),
    }


def _runtime_error_result(context: RuntimeExecutionContext, exc: Exception) -> RuntimeExecutionResult:
    return RuntimeExecutionResult(
        benchmark_id=context.instance.benchmark_id,
        instance_id=context.instance.instance_id,
        backend_name=context.backend_name,
        status="runtime-error",
        error=str(exc),
        metadata={
            "error_type": type(exc).__name__,
        },
    )

def _prepare_openclaw_context(context: RuntimeExecutionContext) -> None:
    emit_progress(
        f"[{context.instance.instance_id}] ensuring benchmark image {_resolve_benchmark_image(context)}",
        component="openclaw",
    )
    _ensure_docker_image_available(_resolve_benchmark_image(context), role="Benchmark sandbox")
    emit_progress(
        f"[{context.instance.instance_id}] preparing workspace at {_problem_root(context)}",
        component="openclaw",
    )
    _ensure_workspace_files(context, repo_path=_container_repo_path(context))
    emit_progress(
        f"[{context.instance.instance_id}] materializing repository into host workspace",
        component="openclaw",
    )
    _materialize_host_repo_from_benchmark_image(context)


def _validate_batch_contexts(contexts: list[RuntimeExecutionContext]) -> None:
    if not contexts:
        raise RuntimeError("OpenClaw batch execution requires at least one context.")
    gateway_name = _resolve_gateway_name(contexts[0])
    state_dir = _state_dir(contexts[0])
    seen_keys: set[str] = set()
    for context in contexts:
        if context.workspace != "docker":
            raise RuntimeError(OPENCLAW_WORKSPACE_ERROR.format(workspace=context.workspace))
        if _resolve_gateway_name(context) != gateway_name:
            raise RuntimeError("OpenClaw batch execution requires all contexts to use the same gateway name.")
        if _state_dir(context) != state_dir:
            raise RuntimeError("OpenClaw batch execution requires all contexts to use the same state directory.")
        benchmark_key = _benchmark_key(context)
        if benchmark_key in seen_keys:
            raise RuntimeError(f"OpenClaw batch execution received duplicate benchmark_id: {benchmark_key}")
        seen_keys.add(benchmark_key)


def _run_openclaw_agent_request(
    context: RuntimeExecutionContext,
    *,
    gateway_name: str,
    keep_gateway: bool,
    keep_workspace: bool,
    gateway_started: bool,
    plugin_runtime: dict[str, Any],
) -> RuntimeExecutionResult:
    repo_path = _container_repo_path(context)
    config_path = _config_path(context)
    agent_id: str | None = None
    session_id: str | None = None
    attempted_session_ids: list[str] = []
    try:
        agent_id = _build_agent_id(context)
        session_id = _build_session_id(context)
        _ensure_image_in_isolated_daemon(
            context,
            _resolve_benchmark_image(context),
            role="Benchmark sandbox",
        )

        run_dir = context.instance_dir / "openclaw"
        run_dir.mkdir(parents=True, exist_ok=True)
        _repo_dir(context).mkdir(parents=True, exist_ok=True)
        _artifacts_dir(context).mkdir(parents=True, exist_ok=True)
        max_ws_retries = _resolve_ws_watchdog_retries(context)
        ws_retry_backoff_seconds = _resolve_ws_watchdog_backoff_seconds(context)
        retry_error_summaries: list[str] = []
        result: subprocess.CompletedProcess[str] | None = None
        final_session_id: str | None = None
        final_attempt_finished_at = time.time()
        request_path = run_dir / "request.json"
        message_path = run_dir / "message.txt"

        for attempt_index in range(max_ws_retries + 1):
            current_session_id = (
                session_id
                if attempt_index == 0
                else _slug(f"{session_id}-ws-retry-{attempt_index}", limit=72)
            )
            attempted_session_ids.append(current_session_id)

            if not _docker_container_running(gateway_name):
                _recover_gateway_for_retry(
                    context,
                    gateway_name=gateway_name,
                    agent_id=agent_id,
                )

            request_metadata = _build_request_metadata(
                context,
                agent_id=agent_id,
                session_id=current_session_id,
                gateway_name=gateway_name,
                repo_path=repo_path,
            )
            request_metadata["ws_watchdog_attempt"] = attempt_index + 1
            _write_json(request_path, request_metadata)
            message_path.write_text(_build_message(context, repo_path=repo_path) + "\n", encoding="utf-8")

            cli_args = [
                "agent",
                "--agent",
                agent_id,
                "--session-id",
                current_session_id,
                "--message",
                message_path.read_text(encoding="utf-8").strip(),
                "--json",
            ]
            cli_args.extend(_resolve_agent_args(context))

            emit_progress(
                f"[{context.instance.instance_id}] running openclaw agent session {current_session_id}",
                component="openclaw",
            )
            attempt_result = _run_openclaw_cli(context, gateway_name=gateway_name, cli_args=cli_args)
            final_attempt_finished_at = time.time()

            (run_dir / f"stdout.attempt{attempt_index + 1}.log").write_text(
                attempt_result.stdout or "",
                encoding="utf-8",
            )
            (run_dir / f"stderr.attempt{attempt_index + 1}.log").write_text(
                attempt_result.stderr or "",
                encoding="utf-8",
            )

            if attempt_result.returncode == 0:
                result = attempt_result
                final_session_id = current_session_id
                break

            if (
                attempt_index < max_ws_retries
                and _is_gateway_ws_1006_error(stderr=attempt_result.stderr or "", stdout=attempt_result.stdout or "")
            ):
                summary = _error_summary(attempt_result.stderr or "", attempt_result.stdout or "")
                retry_error_summaries.append(summary)
                emit_progress(
                    (
                        f"[{context.instance.instance_id}] watchdog retry after WS 1006 "
                        f"(attempt {attempt_index + 1}/{max_ws_retries + 1})"
                    ),
                    component="openclaw",
                )
                if ws_retry_backoff_seconds > 0:
                    time.sleep(ws_retry_backoff_seconds * (2**attempt_index))
                _recover_gateway_for_retry(
                    context,
                    gateway_name=gateway_name,
                    agent_id=agent_id,
                )
                continue

            if (
                attempt_index < max_ws_retries
                and _is_gateway_container_missing_error(
                    stderr=attempt_result.stderr or "",
                    stdout=attempt_result.stdout or "",
                )
            ):
                emit_progress(
                    (
                        f"[{context.instance.instance_id}] watchdog retry after missing gateway container "
                        f"(attempt {attempt_index + 1}/{max_ws_retries + 1})"
                    ),
                    component="openclaw",
                )
                _recover_gateway_for_retry(
                    context,
                    gateway_name=gateway_name,
                    agent_id=agent_id,
                )
                continue

            result = attempt_result
            final_session_id = current_session_id
            break

        if result is None or final_session_id is None:
            raise RuntimeError("OpenClaw watchdog failed to produce a terminal agent result")

        session_id = final_session_id
        stdout_path, stderr_path = _write_logs(run_dir, stdout=result.stdout, stderr=result.stderr)
        emit_progress(
            f"[{context.instance.instance_id}] openclaw agent finished with exit code {result.returncode}",
            component="openclaw",
        )
        error_summary = _error_summary(result.stderr, result.stdout)
        ws_close_code, ws_close_reason = _extract_ws_close_details(error_summary)
        last_hook_event = _last_runtime_hook_event_for_session(plugin_runtime, session_id=session_id)
        delta_to_fail_s: float | None = None
        if result.returncode != 0 and isinstance(last_hook_event, dict):
            hook_ts = _parse_iso8601_timestamp(str(last_hook_event.get("timestamp") or ""))
            if hook_ts is not None:
                delta_to_fail_s = max(0.0, final_attempt_finished_at - hook_ts)
        metadata = {
            "gateway_name": gateway_name,
            "gateway_image": _resolve_gateway_image(context),
            "benchmark_image": _resolve_benchmark_image(context),
            "agent_id": agent_id,
            "session_id": session_id,
            "repo_path": repo_path,
            "workspace_seeded_from_testbed": True,
            "keep_gateway": keep_gateway,
            "gateway_started": gateway_started,
            "state_dir": str(_state_dir(context)),
            "config_path": str(config_path),
            "plugin": plugin_runtime,
            "request_path": str(request_path),
            "message_path": str(message_path),
            "workspace_dir": str(_agent_workspace_dir(context)),
            "problem_root": str(_problem_root(context)),
            "artifacts_dir": str(_artifacts_dir(context)),
            "command": "openclaw agent",
            "stdout_log": str(stdout_path),
            "stderr_log": str(stderr_path),
            "exit_code": result.returncode,
            "runtime_options": _resolve_runtime_options(context),
            "ws_close_code": ws_close_code,
            "ws_close_reason": ws_close_reason,
            "last_hook_event": last_hook_event,
            "delta_to_fail_s": delta_to_fail_s,
            "gateway_container_state_at_fail": (
                _gateway_container_state_snapshot(gateway_name) if result.returncode != 0 else None
            ),
            "ws_watchdog": {
                "max_retries": max_ws_retries,
                "attempts": len(attempted_session_ids),
                "triggered_retries": max(0, len(attempted_session_ids) - 1),
                "attempt_session_ids": attempted_session_ids,
                "retry_error_summaries": retry_error_summaries,
            },
        }
        try:
            metadata["token_usage"] = _collect_openclaw_token_usage(
                context,
                agent_id=agent_id,
                session_id=session_id,
            )
        except Exception as exc:
            metadata["token_usage"] = {
                "source": "openclaw.transcript",
                "error": str(exc),
            }
        patch_eval_dir = run_dir / "patch-eval"
        emit_progress(
            f"[{context.instance.instance_id}] capturing repository patch",
            component="openclaw",
        )
        patch_capture = capture_repo_patch(repo_dir=_repo_dir(context), output_dir=patch_eval_dir)
        metadata["patch_capture"] = patch_capture
        emit_progress(
            f"[{context.instance.instance_id}] patch capture status: {patch_capture.get('status')} changes={patch_capture.get('has_changes')}",
            component="openclaw",
        )
        emit_progress(
            f"[{context.instance.instance_id}] starting SWE-rebench patch evaluation",
            component="openclaw",
        )
        metadata["patch_evaluation"] = maybe_run_swerebench_evaluation(
            instance_id=context.instance.instance_id,
            benchmark_name=str(context.instance.hidden_eval.get("benchmark", "")).strip() or context.instance.repo,
            split=str(_resolve_runtime_option(context, "swerebench_split") or "test").strip() or "test",
            model_name=str(context.llm_config.get("model") or context.backend_name or "unknown-model"),
            patch_capture=patch_capture,
            output_dir=patch_eval_dir,
            runtime_options=_resolve_runtime_options(context),
        )
        emit_progress(
            f"[{context.instance.instance_id}] SWE-rebench patch evaluation status: {metadata['patch_evaluation'].get('status', 'unknown')}",
            component="openclaw",
        )
        if result.returncode == 0:
            return RuntimeExecutionResult(
                benchmark_id=context.instance.benchmark_id,
                instance_id=context.instance.instance_id,
                backend_name=context.backend_name,
                status="completed",
                metadata=metadata,
            )

        return RuntimeExecutionResult(
            benchmark_id=context.instance.benchmark_id,
            instance_id=context.instance.instance_id,
            backend_name=context.backend_name,
            status="runtime-error",
            error=error_summary,
            metadata=metadata,
        )
    finally:
        emit_progress(
            f"[{context.instance.instance_id}] cleaning up OpenClaw resources",
            component="openclaw",
        )
        if agent_id and not keep_gateway:
            for tracked_session_id in sorted(set(attempted_session_ids)):
                _cleanup_sandbox_containers(
                    context,
                    agent_id=agent_id,
                    session_id=tracked_session_id,
                )
        if not keep_workspace:
            problem_root = _problem_root(context)
            if problem_root.exists():
                emit_progress(
                    f"[{context.instance.instance_id}] removing problem workspace at {problem_root}",
                    component="openclaw",
                )
                shutil.rmtree(problem_root, ignore_errors=True)


def run_openclaw_instance(context: RuntimeExecutionContext) -> RuntimeExecutionResult:
    if context.workspace != "docker":
        raise RuntimeError(OPENCLAW_WORKSPACE_ERROR.format(workspace=context.workspace))

    keep_gateway = _resolve_keep_gateway_enabled(context)
    keep_workspace = _resolve_keep_workspace_enabled(context)
    _prepare_openclaw_context(context)
    _ensure_agent_config(context)
    plugin_runtime = _ensure_plugin_runtime(context)
    gateway_name, gateway_started = _start_gateway_container(context)
    try:
        _wait_for_gateway_ready(context, gateway_name=gateway_name)
        _wait_for_agent_registered(context, gateway_name=gateway_name, agent_id=_build_agent_id(context))
        return _run_openclaw_agent_request(
            context,
            gateway_name=gateway_name,
            keep_gateway=keep_gateway,
            keep_workspace=keep_workspace,
            gateway_started=gateway_started,
            plugin_runtime=plugin_runtime,
        )
    finally:
        if gateway_started and not keep_gateway:
            _docker_remove_container(gateway_name)
            _teardown_isolated_daemon(context)
            _teardown_isolated_control_network(context)
            _teardown_private_network(context)
        if not keep_workspace:
            workspace_root = _gateway_workspace_root(context)
            if workspace_root.exists():
                shutil.rmtree(workspace_root, ignore_errors=True)


def run_openclaw_batch(
    contexts: list[RuntimeExecutionContext],
    concurrency: int,
) -> list[RuntimeExecutionResult]:
    _validate_batch_contexts(contexts)
    primary = contexts[0]
    keep_gateway = _resolve_keep_gateway_enabled(primary)
    keep_workspace = _resolve_keep_workspace_enabled(primary)
    gateway_name = _resolve_gateway_name(primary)
    gateway_started = False
    results: list[RuntimeExecutionResult | None] = [None] * len(contexts)

    try:
        runnable: list[tuple[int, RuntimeExecutionContext]] = []
        for index, context in enumerate(contexts):
            try:
                _prepare_openclaw_context(context)
            except Exception as exc:
                results[index] = _runtime_error_result(context, exc)
            else:
                runnable.append((index, context))

        if not runnable:
            return [result for result in results if result is not None]

        runnable_contexts = [context for _, context in runnable]
        gateway_context = runnable_contexts[0]
        try:
            _ensure_agent_configs(runnable_contexts)
            plugin_runtime = _ensure_plugin_runtime(gateway_context)
        except Exception as exc:
            for index, context in runnable:
                results[index] = _runtime_error_result(context, exc)
            return [result for result in results if result is not None]

        gateway_name, gateway_started = _start_gateway_container(
            gateway_context,
            restart_existing=not keep_gateway,
        )
        _wait_for_gateway_ready(gateway_context, gateway_name=gateway_name)

        ready: list[tuple[int, RuntimeExecutionContext]] = []
        for index, context in runnable:
            try:
                _wait_for_agent_registered(context, gateway_name=gateway_name, agent_id=_build_agent_id(context))
            except Exception as exc:
                results[index] = _runtime_error_result(context, exc)
            else:
                ready.append((index, context))

        if not ready:
            return [result for result in results if result is not None]

        worker_count = max(1, min(concurrency, len(ready)))
        if worker_count == 1:
            for index, context in ready:
                try:
                    results[index] = _run_openclaw_agent_request(
                        context,
                        gateway_name=gateway_name,
                        keep_gateway=keep_gateway,
                        keep_workspace=keep_workspace,
                        gateway_started=gateway_started,
                        plugin_runtime=plugin_runtime,
                    )
                except Exception as exc:
                    results[index] = _runtime_error_result(context, exc)
            return [result for result in results if result is not None]

        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            future_to_index = {
                executor.submit(
                    _run_openclaw_agent_request,
                    context,
                    gateway_name=gateway_name,
                    keep_gateway=keep_gateway,
                    keep_workspace=keep_workspace,
                    gateway_started=gateway_started,
                    plugin_runtime=plugin_runtime,
                ): index
                for index, context in ready
            }
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    results[index] = future.result()
                except Exception as exc:
                    results[index] = _runtime_error_result(contexts[index], exc)
        return [result for result in results if result is not None]
    finally:
        if gateway_started and not keep_gateway:
            _docker_remove_container(gateway_name)
            _teardown_isolated_daemon(primary)
            _teardown_isolated_control_network(primary)
            _teardown_private_network(primary)
        if not keep_workspace:
            workspace_root = _gateway_workspace_root(primary)
            if workspace_root.exists():
                shutil.rmtree(workspace_root, ignore_errors=True)


OPENCLAW_BACKEND = RuntimeBackend(
    name="openclaw",
    default_image=OPENCLAW_GATEWAY_IMAGE,
    default_build_target="container",
    execution_hint=(
        "Use --execute to run benchmark instances through one shared OpenClaw gateway and a per-problem agent. "
        "OpenClaw now seeds the benchmark repo into the agent workspace and reuses the benchmark image dependencies by default. "
        "Use --openclaw-plugin-path to override the SOMA plugin checkout or "
        "--openclaw-disable-plugin to keep the built-in context engine."
    ),
    execute=run_openclaw_instance,
    execute_batch=run_openclaw_batch,
)
