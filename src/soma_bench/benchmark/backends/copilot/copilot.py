from __future__ import annotations

import json
import os
import re
import secrets
import shlex
import shutil
import subprocess
import hashlib
import tempfile
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from ..base import RuntimeBackend, RuntimeExecutionContext, RuntimeExecutionResult
from ... import dind_utils
from ...swerebench_eval import capture_repo_patch
from ...progress import emit_progress
from ...registry_auth import docker_env_for_image, uses_token_auth
from ...swebench_images import derive_prebaked_dind_image, resolve_prebaked_dind_repo

COPILOT_WORKSPACE_ERROR = (
    "Copilot backend currently supports only docker workspace execution. "
    "Received workspace={workspace!r}."
)

COPILOT_DEFAULT_COMPOSE_FILE = str(
    (Path(__file__).resolve().parent / "copilot-cli-container" / "docker-compose.yml").resolve()
)
COPILOT_DEFAULT_SERVICE = "copilot"
COPILOT_DEFAULT_PROXY_SERVICE = "proxy"
COPILOT_DEFAULT_PROXY_PORT = 8080
COPILOT_COMPRESSION_SERVICE_NAME = "compression-service"
COPILOT_COMPRESSION_SERVICE_PORT = 8000
COPILOT_DEFAULT_COMPRESSION_IMAGE = "soma-copilot-compression-service:latest"
COPILOT_DEFAULT_COMPRESSION_SCRIPT_CONTAINER_PATH = "/app/miner/base_miner.py"
COPILOT_DEFAULT_SWE_SANDBOX_SERVICE = "swe-sandbox"
COPILOT_DEFAULT_SWE_SANDBOX_REPO_PATH = "/testbed"
COPILOT_DEFAULT_DIND_SERVICE = "dind"
COPILOT_DEFAULT_DIND_IMAGE = "docker:dind"
# 2376 is dind's mutual-TLS port. The plaintext 2375 listener only exists when TLS is
# explicitly disabled via SOMA_COPILOT_DIND_TLS=false, which is not the supported path.
COPILOT_DIND_TLS_PORT = 2376
COPILOT_DIND_PLAINTEXT_PORT = 2375
COPILOT_DIND_CLIENT_CERT_DIR = "/certs/client"
# dind's entrypoint issues its server certificate for "docker", "localhost" and the
# container hostname - never for the compose service name. Clients must therefore address
# the daemon as "docker" (compose publishes that network alias) to keep TLS hostname
# verification enabled instead of disabling it.
COPILOT_DIND_TLS_HOSTNAME = "docker"
COPILOT_DEFAULT_DIND_TLS = True
# dind's own default. Passed through to the daemon explicitly (the compose `dind` service
# declares it) rather than relied upon, so that the TLS-off path can send an empty value.
COPILOT_DIND_TLS_CERTDIR = "/certs"
# Bridge networks marked `internal` still hand containers a gateway address, and that
# address is the host - packets to it land in INPUT, which the internal-network DROP rules
# (FORWARD only) never inspect. `isolated` (Docker >= 28) allocates no gateway at all, which
# is what actually makes the host unreachable. `nat` restores the old, reachable behavior
# for engines too old to support it.
COPILOT_SANDBOX_GATEWAY_MODE_ISOLATED = "isolated"
COPILOT_SANDBOX_GATEWAY_MODE_NAT = "nat"
COPILOT_DEFAULT_SANDBOX_GATEWAY_MODE = COPILOT_SANDBOX_GATEWAY_MODE_ISOLATED
COPILOT_DIND_WORKSPACE_MOUNT = "/workspace"
COPILOT_SWE_EXEC_BROKER_SERVICE = "swe-exec-broker"
COPILOT_DEFAULT_SWE_EXEC_BROKER = True
COPILOT_DEFAULT_SWE_EXEC_BROKER_PORT = 8100
COPILOT_DEFAULT_SWE_EXEC_TIMEOUT_SECONDS = 900
COPILOT_SWE_EXEC_BROKER_READY_RETRIES = 60
COPILOT_SWE_EXEC_BROKER_READY_SLEEP_SECONDS = 0.5
COPILOT_SWE_EXEC_CLIENT_CONTAINER_PATH = "/usr/local/bin/swe-exec"
COPILOT_WORKSPACE_VOLUME_BASENAME = "copilot-workspace"
COPILOT_DEFAULT_IMAGE = "local/copilot-cli:latest"
COPILOT_DEFAULT_NETWORK_ISOLATION = True
COPILOT_DEFAULT_KEEP_STACK = False
COPILOT_DEFAULT_SWE_SANDBOX_ENABLED = True
COPILOT_PROXY_UPSTREAM_ENV = "COMPACT_BENCH_LLM_BASE_URL"
COPILOT_DEFAULT_OUTPUT_FORMAT = "json"
COPILOT_DEFAULT_RUN_ROOT_DIRNAME = "soma-benchmark-copilot-runs"
COPILOT_DEFAULT_CLEANUP_REPO = True
COPILOT_DEFAULT_COMPRESSION_AUTOBUILD = False
COPILOT_DEFAULT_SHARED_PROXY_BATCH = False
COPILOT_DEFAULT_SHARED_PROXY_TEARDOWN = True
COPILOT_DEFAULT_RUN_RETENTION_SECONDS = 24 * 60 * 60
COPILOT_DEFAULT_RUN_CLEANUP_INTERVAL_SECONDS = 5 * 60
COPILOT_DEFAULT_PRESERVE_RUN_DIRS = False
COPILOT_PROJECT_NAME_PREFIX = "soma-copilot-"
COPILOT_STALE_SIDECAR_MAX_AGE_ENV = "SOMA_COPILOT_STALE_SIDECAR_MAX_AGE_SECONDS"
COPILOT_STALE_CLEANUP_INTERVAL_ENV = "SOMA_COPILOT_STALE_CLEANUP_INTERVAL_SECONDS"
COPILOT_DEFAULT_STALE_SIDECAR_MAX_AGE_SECONDS = 30 * 60
COPILOT_DEFAULT_STALE_CLEANUP_INTERVAL_SECONDS = 5 * 60

_run_root_cleanup_lock = threading.Lock()
_last_run_root_cleanup_monotonic = 0.0
_stale_docker_resource_cleanup_lock = threading.Lock()
_last_stale_docker_resource_cleanup_monotonic = 0.0
_COMPOSE_PROJECT_CONTAINER_RE = re.compile(
    r"^(" + re.escape(COPILOT_PROJECT_NAME_PREFIX) + r"[0-9a-f]+)"
    r"-(dind|proxy|compression-service|swe-exec-broker)-\S+$"
)
# Networks and volumes outlive containers: `compose down` removing the containers and then
# failing (or being killed) partway leaves these behind, and nothing that enumerates
# containers can see them again. Each orphan holds a /16 out of Docker's finite address
# pool and a workspace volume now seeded from the task image, so they cannot be ignored.
_COMPOSE_PROJECT_NETWORK_RE = re.compile(
    r"^(" + re.escape(COPILOT_PROJECT_NAME_PREFIX) + r"[0-9a-f]+)"
    r"_(copilot-sandbox|copilot-egress|copilot-dind)$"
)
_COMPOSE_PROJECT_VOLUME_RE = re.compile(
    r"^(" + re.escape(COPILOT_PROJECT_NAME_PREFIX) + r"[0-9a-f]+)"
    r"_(copilot-home|copilot-dind-certs|" + re.escape(COPILOT_WORKSPACE_VOLUME_BASENAME) + r")$"
)


def _resolve_runtime_options(context: RuntimeExecutionContext) -> dict[str, Any]:
    value = context.run_payload.get("runtime_options")
    if isinstance(value, dict):
        return value
    return {}


def _resolve_runtime_option(context: RuntimeExecutionContext, name: str) -> Any:
    runtime_options = _resolve_runtime_options(context)
    return runtime_options.get(name)


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


def _resolve_compose_file(context: RuntimeExecutionContext) -> Path:
    for value in (
        _resolve_runtime_option(context, "copilot_compose_file"),
        os.getenv("SOMA_COPILOT_COMPOSE_FILE"),
        COPILOT_DEFAULT_COMPOSE_FILE,
    ):
        if isinstance(value, str) and value.strip():
            path = Path(value.strip()).expanduser().resolve()
            if path.is_file():
                return path
    raise RuntimeError(
        "Copilot docker-compose file was not found. Set SOMA_COPILOT_COMPOSE_FILE "
        "or runtime_options.copilot_compose_file."
    )


def _resolve_compose_service(context: RuntimeExecutionContext) -> str:
    for value in (
        _resolve_runtime_option(context, "copilot_service"),
        os.getenv("SOMA_COPILOT_SERVICE"),
        COPILOT_DEFAULT_SERVICE,
    ):
        if isinstance(value, str) and value.strip():
            return value.strip()
    return COPILOT_DEFAULT_SERVICE


def _resolve_extra_args(context: RuntimeExecutionContext) -> list[str]:
    for value in (
        _resolve_runtime_option(context, "copilot_args"),
        os.getenv("SOMA_COPILOT_ARGS"),
    ):
        if isinstance(value, str) and value.strip():
            return shlex.split(value)
        if isinstance(value, list) and all(isinstance(item, str) for item in value):
            return list(value)
    return ["--allow-all", "--no-ask-user"]


def _resolve_output_format(context: RuntimeExecutionContext) -> str:
    for value in (
        _resolve_runtime_option(context, "copilot_output_format"),
        os.getenv("SOMA_COPILOT_OUTPUT_FORMAT"),
        COPILOT_DEFAULT_OUTPUT_FORMAT,
    ):
        if isinstance(value, str) and value.strip():
            normalized = value.strip().lower()
            if normalized in {"text", "json"}:
                return normalized
    return COPILOT_DEFAULT_OUTPUT_FORMAT


def _resolve_network_isolation_enabled(context: RuntimeExecutionContext) -> bool:
    for value in (
        os.getenv("SOMA_COPILOT_NETWORK_ISOLATION"),
        _resolve_runtime_option(context, "copilot_network_isolation"),
    ):
        option = _coerce_bool_option(value)
        if option is not None:
            return option
    return COPILOT_DEFAULT_NETWORK_ISOLATION


def _resolve_keep_compose_stack(context: RuntimeExecutionContext) -> bool:
    for value in (
        os.getenv("SOMA_COPILOT_KEEP_STACK"),
        _resolve_runtime_option(context, "copilot_keep_stack"),
    ):
        option = _coerce_bool_option(value)
        if option is not None:
            return option
    return COPILOT_DEFAULT_KEEP_STACK


def _resolve_cleanup_repo_enabled(context: RuntimeExecutionContext) -> bool:
    for value in (
        os.getenv("SOMA_COPILOT_CLEANUP_REPO"),
        _resolve_runtime_option(context, "copilot_cleanup_repo"),
    ):
        option = _coerce_bool_option(value)
        if option is not None:
            return option
    return COPILOT_DEFAULT_CLEANUP_REPO


def _coerce_positive_int_option(raw_value: Any, default: int) -> int:
    try:
        parsed = int(str(raw_value).strip())
    except (TypeError, ValueError, AttributeError):
        return default
    return parsed if parsed > 0 else default


def _resolve_agent_timeout_seconds(context: RuntimeExecutionContext) -> float | None:
    for value in (
        os.getenv("SOMA_COPILOT_AGENT_TIMEOUT_SECONDS"),
        _resolve_runtime_option(context, "copilot_agent_timeout_seconds"),
    ):
        if value is None:
            continue
        try:
            parsed = float(str(value).strip())
        except (TypeError, ValueError):
            continue
        if parsed > 0:
            return parsed
    return None


def _resolve_preserve_run_dirs_enabled(context: RuntimeExecutionContext) -> bool:
    for value in (
        os.getenv("SOMA_COPILOT_PRESERVE_RUN_DIRS"),
        _resolve_runtime_option(context, "copilot_preserve_run_dirs"),
    ):
        option = _coerce_bool_option(value)
        if option is not None:
            return option
    return COPILOT_DEFAULT_PRESERVE_RUN_DIRS


def _resolve_run_retention_seconds(context: RuntimeExecutionContext) -> int:
    return _coerce_positive_int_option(
        _resolve_runtime_option(context, "copilot_run_retention_seconds")
        or os.getenv("SOMA_COPILOT_RUN_RETENTION_SECONDS"),
        COPILOT_DEFAULT_RUN_RETENTION_SECONDS,
    )


def _resolve_run_cleanup_interval_seconds(context: RuntimeExecutionContext) -> int:
    return _coerce_positive_int_option(
        _resolve_runtime_option(context, "copilot_run_cleanup_interval_seconds")
        or os.getenv("SOMA_COPILOT_RUN_CLEANUP_INTERVAL_SECONDS"),
        COPILOT_DEFAULT_RUN_CLEANUP_INTERVAL_SECONDS,
    )


def _maybe_cleanup_stale_run_dirs(context: RuntimeExecutionContext, *, base_root: Path) -> None:
    """Remove old ``run-*`` directories left behind under ``base_root``.

    Each run persists a full workspace snapshot (patch-eval/workspace-snapshot) that is
    never otherwise deleted, so left unattended this directory grows without bound.
    Mirrors the retention-based sweep sandbox_service already runs for its own output root.
    """
    if _resolve_preserve_run_dirs_enabled(context):
        return

    retention_seconds = _resolve_run_retention_seconds(context)
    cleanup_interval_seconds = _resolve_run_cleanup_interval_seconds(context)

    global _last_run_root_cleanup_monotonic
    now_monotonic = time.monotonic()
    with _run_root_cleanup_lock:
        if now_monotonic - _last_run_root_cleanup_monotonic < cleanup_interval_seconds:
            return
        _last_run_root_cleanup_monotonic = now_monotonic

    if not base_root.is_dir():
        return

    cutoff_epoch = time.time() - retention_seconds
    removed_count = 0
    for candidate in base_root.iterdir():
        if not candidate.name.startswith("run-") or not candidate.is_dir():
            continue
        try:
            modified_epoch = candidate.stat().st_mtime
        except OSError:
            continue
        if modified_epoch >= cutoff_epoch:
            continue
        shutil.rmtree(candidate, ignore_errors=True)
        if not candidate.exists():
            removed_count += 1

    if removed_count > 0:
        emit_progress(
            f"[copilot] removed {removed_count} stale run director"
            f"{'y' if removed_count == 1 else 'ies'} under {base_root} "
            f"(retention_seconds={retention_seconds})",
            component="copilot",
        )


def _resolve_stale_sidecar_max_age_seconds(context: RuntimeExecutionContext) -> int:
    return _coerce_positive_int_option(
        _resolve_runtime_option(context, "copilot_stale_sidecar_max_age_seconds")
        or os.getenv(COPILOT_STALE_SIDECAR_MAX_AGE_ENV),
        COPILOT_DEFAULT_STALE_SIDECAR_MAX_AGE_SECONDS,
    )


def _resolve_stale_docker_cleanup_interval_seconds(context: RuntimeExecutionContext) -> int:
    return _coerce_positive_int_option(
        _resolve_runtime_option(context, "copilot_stale_cleanup_interval_seconds")
        or os.getenv(COPILOT_STALE_CLEANUP_INTERVAL_ENV),
        COPILOT_DEFAULT_STALE_CLEANUP_INTERVAL_SECONDS,
    )


def _docker_container_created_epoch(name: str) -> float | None:
    result = _run_command(["docker", "inspect", "-f", "{{.Created}}", name])
    if result.returncode != 0:
        return None
    return _parse_docker_timestamp((result.stdout or "").strip())


def _docker_network_created_epoch(name: str) -> float | None:
    # `{{json .Created}}` rather than `{{.Created}}`: the latter renders Go's own
    # time.Time formatting ("... +0000 UTC"), which is not parseable as RFC 3339.
    result = _run_command(["docker", "network", "inspect", "-f", "{{json .Created}}", name])
    if result.returncode != 0:
        return None
    return _parse_docker_timestamp((result.stdout or "").strip().strip('"'))


def _docker_volume_created_epoch(name: str) -> float | None:
    result = _run_command(["docker", "volume", "inspect", "-f", "{{json .CreatedAt}}", name])
    if result.returncode != 0:
        return None
    return _parse_docker_timestamp((result.stdout or "").strip().strip('"'))


def _parse_docker_timestamp(raw: str) -> float | None:
    if not raw:
        return None
    normalized = raw
    if normalized.endswith("Z"):
        normalized = f"{normalized[:-1]}+00:00"
    normalized = re.sub(r"\.(\d{6})\d+(?=[+-]\d{2}:\d{2}$)", r".\1", normalized)
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()


def _find_stale_copilot_projects(*, keep_project: str, max_age_seconds: int) -> list[str]:
    """Projects whose every *remaining* resource is old enough to tear down.

    Discovery deliberately does not go by containers alone. A project whose containers are
    already gone - torn down by a `compose down` that then failed, or by a run killed
    between the two steps - keeps its networks and volumes indefinitely, and `docker ps`
    can never surface it again. Sweeping those requires enumerating the networks and
    volumes directly.
    """
    ages: dict[str, list[float | None]] = {}

    def _record(project: str, age: float | None) -> None:
        if project != keep_project:
            ages.setdefault(project, []).append(age)

    sources = (
        (["docker", "ps", "-a", "--format", "{{.Names}}"],
         _COMPOSE_PROJECT_CONTAINER_RE, _docker_container_created_epoch),
        (["docker", "network", "ls", "--format", "{{.Name}}"],
         _COMPOSE_PROJECT_NETWORK_RE, _docker_network_created_epoch),
        (["docker", "volume", "ls", "--format", "{{.Name}}"],
         _COMPOSE_PROJECT_VOLUME_RE, _docker_volume_created_epoch),
    )
    for command, pattern, age_of in sources:
        listing = _run_command(command)
        if listing.returncode != 0:
            continue
        for name in (listing.stdout or "").splitlines():
            name = name.strip()
            match = pattern.match(name)
            if match:
                _record(match.group(1), age_of(name))

    now_epoch = time.time()
    stale: list[str] = []
    for project, resource_ages in ages.items():
        # A project is only stale once every resource belonging to it is old enough - a
        # freshly started concurrent run must never be torn down mid-flight, and its
        # network exists before its containers do.
        if any(age is None for age in resource_ages):
            continue
        if all(now_epoch - age >= max_age_seconds for age in resource_ages):
            stale.append(project)
    return stale


def _remove_orphaned_project_resources(*, project: str) -> None:
    """Remove any network/volume of `project` that survived `compose down`.

    `compose down` is the normal path and usually suffices, but it is also the step that
    failed for every orphan this sweep exists to collect - so the resources are removed
    by name as well. Both commands refuse to touch anything still in use, which is the
    safety net against racing a live run.
    """
    for name in (
        f"{project}_copilot-sandbox",
        f"{project}_copilot-egress",
        f"{project}_copilot-dind",
    ):
        result = _run_command(["docker", "network", "rm", name])
        if result.returncode == 0:
            emit_progress(f"[copilot] removed orphaned network {name}", component="copilot")

    for name in (
        f"{project}_copilot-home",
        f"{project}_copilot-dind-certs",
        _workspace_volume_name(compose_project=project),
    ):
        result = _run_command(["docker", "volume", "rm", name])
        if result.returncode == 0:
            emit_progress(f"[copilot] removed orphaned volume {name}", component="copilot")


def _teardown_stale_copilot_project(context: RuntimeExecutionContext, *, project: str) -> bool:
    compose_file = _resolve_compose_file(context)
    env = os.environ.copy()
    env["COMPOSE_PROFILES"] = "copilot-sidecars"
    workspace_volume = _workspace_volume_name(compose_project=project)
    env["COPILOT_WORKSPACE_VOLUME_NAME"] = workspace_volume

    down_result = _run_command(
        [
            "docker", "compose", "-f", str(compose_file),
            "--project-name", project,
            "down", "--remove-orphans", "--volumes",
        ],
        env=env,
    )
    _remove_workspace_volume(volume_name=workspace_volume)
    _remove_orphaned_project_resources(project=project)
    if down_result.returncode != 0:
        emit_progress(
            f"[copilot] failed to tear down stale stack {project!r}: "
            f"{(down_result.stderr or down_result.stdout or '').strip()}",
            component="copilot",
        )
        return False
    return True


def _maybe_cleanup_stale_copilot_docker_resources(
    context: RuntimeExecutionContext,
    *,
    keep_project: str,
) -> None:
    """Tear down dind/proxy/compression-service stacks orphaned by a crashed or killed run.

    The `finally` block that normally runs `docker compose down --volumes` never fires if the
    process is SIGKILLed mid-run, leaving the dind sidecar (and its multi-GB nested image
    storage in an anonymous /var/lib/docker volume) on disk indefinitely. This sweep catches
    those by age, mirroring the equivalent stale-resource sweep in the openclaw backend.
    """
    cleanup_interval_seconds = _resolve_stale_docker_cleanup_interval_seconds(context)

    global _last_stale_docker_resource_cleanup_monotonic
    now_monotonic = time.monotonic()
    with _stale_docker_resource_cleanup_lock:
        if now_monotonic - _last_stale_docker_resource_cleanup_monotonic < cleanup_interval_seconds:
            return
        _last_stale_docker_resource_cleanup_monotonic = now_monotonic

    max_age_seconds = _resolve_stale_sidecar_max_age_seconds(context)
    stale_projects = _find_stale_copilot_projects(keep_project=keep_project, max_age_seconds=max_age_seconds)

    removed_count = 0
    for project in stale_projects:
        if _teardown_stale_copilot_project(context, project=project):
            removed_count += 1

    if removed_count > 0:
        emit_progress(
            f"[copilot] cleaned {removed_count} stale orphaned stack"
            f"{'s' if removed_count != 1 else ''} (max_age_seconds={max_age_seconds})",
            component="copilot",
        )


def _resolve_proxy_service(context: RuntimeExecutionContext) -> str:
    for value in (
        _resolve_runtime_option(context, "copilot_proxy_service"),
        os.getenv("SOMA_COPILOT_PROXY_SERVICE"),
        COPILOT_DEFAULT_PROXY_SERVICE,
    ):
        if isinstance(value, str) and value.strip():
            return value.strip()
    return COPILOT_DEFAULT_PROXY_SERVICE


def _resolve_compression_service_image(context: RuntimeExecutionContext) -> str:
    for value in (
        _resolve_runtime_option(context, "copilot_compression_service_image"),
        os.getenv("SOMA_COPILOT_COMPRESSION_SERVICE_IMAGE"),
        COPILOT_DEFAULT_COMPRESSION_IMAGE,
    ):
        if isinstance(value, str) and value.strip():
            return value.strip()
    return COPILOT_DEFAULT_COMPRESSION_IMAGE


def _resolve_compression_service_autobuild(context: RuntimeExecutionContext) -> bool:
    for value in (
        _resolve_runtime_option(context, "copilot_compression_service_autobuild"),
        os.getenv("SOMA_COPILOT_COMPRESSION_SERVICE_AUTOBUILD"),
    ):
        option = _coerce_bool_option(value)
        if option is not None:
            return option
    return COPILOT_DEFAULT_COMPRESSION_AUTOBUILD


def _resolve_use_compose_compression_service(context: RuntimeExecutionContext) -> bool:
    for value in (
        _resolve_runtime_option(context, "copilot_use_compose_compression_service"),
        os.getenv("SOMA_COPILOT_USE_COMPOSE_COMPRESSION_SERVICE"),
    ):
        option = _coerce_bool_option(value)
        if option is not None:
            return option
    return True


def _resolve_compression_service_build_context(context: RuntimeExecutionContext) -> Path:
    default_path = Path(__file__).resolve().parents[5] / "src" / "compression_service"
    for value in (
        _resolve_runtime_option(context, "copilot_compression_service_build_context"),
        os.getenv("SOMA_COPILOT_COMPRESSION_SERVICE_BUILD_CONTEXT"),
    ):
        if isinstance(value, str) and value.strip():
            return Path(value.strip()).expanduser().resolve()
    return default_path.resolve()


def _default_compression_script_path() -> Path:
    return (Path(__file__).resolve().parents[5] / "src" / "compression_service" / "app" / "base_miner.py").resolve()


def _resolve_compression_script_path(context: RuntimeExecutionContext) -> Path:
    for value in (
        _resolve_runtime_option(context, "copilot_compression_script_path"),
        os.getenv("SOMA_COPILOT_COMPRESSION_SCRIPT_PATH"),
    ):
        if isinstance(value, str) and value.strip():
            return Path(value.strip()).expanduser().resolve()
    return _default_compression_script_path()


def _validate_compression_script_path(context: RuntimeExecutionContext) -> Path:
    script_path = _resolve_compression_script_path(context)
    if not script_path.is_file():
        raise RuntimeError(
            "Copilot compression mode received an invalid compressor script path: "
            f"{script_path}. Expected an existing file. "
            "Set --copilot-compression-script-path, runtime_options.copilot_compression_script_path, "
            "or SOMA_COPILOT_COMPRESSION_SCRIPT_PATH."
        )
    return script_path


def _build_compression_service_image(
    *,
    image_name: str,
    build_context: Path,
) -> tuple[str, str]:
    if not build_context.is_dir():
        raise RuntimeError(
            "Copilot compression-service autobuild build context was not found: "
            f"{build_context}"
        )

    build_result = _run_command([
        "docker",
        "build",
        "-t",
        image_name,
        str(build_context),
    ])
    if build_result.returncode != 0:
        raise RuntimeError(
            "Failed to build compression-service image for Copilot backend. "
            f"{(build_result.stderr or build_result.stdout or '').strip()}"
        )
    return (build_result.stdout or "", build_result.stderr or "")


def _resolve_stack_services(
    context: RuntimeExecutionContext,
    *,
    proxy_service: str,
    compression_enabled: bool,
    include_proxy: bool,
) -> list[str]:
    services: list[str] = []
    # In shared-proxy mode, run containers should not launch their own proxy sidecar.
    if include_proxy:
        services.append(proxy_service)
    if compression_enabled:
        services.append(COPILOT_COMPRESSION_SERVICE_NAME)
    return services


def _resolve_swe_sandbox_enabled(context: RuntimeExecutionContext) -> bool:
    for value in (
        _resolve_runtime_option(context, "copilot_swe_sandbox"),
        os.getenv("SOMA_COPILOT_SWE_SANDBOX"),
    ):
        option = _coerce_bool_option(value)
        if option is not None:
            return option
    return COPILOT_DEFAULT_SWE_SANDBOX_ENABLED


def _resolve_swe_sandbox_service(context: RuntimeExecutionContext) -> str:
    for value in (
        _resolve_runtime_option(context, "copilot_swe_sandbox_service"),
        os.getenv("SOMA_COPILOT_SWE_SANDBOX_SERVICE"),
        COPILOT_DEFAULT_SWE_SANDBOX_SERVICE,
    ):
        if isinstance(value, str) and value.strip():
            return value.strip()
    return COPILOT_DEFAULT_SWE_SANDBOX_SERVICE


def _resolve_swe_sandbox_repo_path(context: RuntimeExecutionContext) -> str:
    for value in (
        _resolve_runtime_option(context, "copilot_swe_sandbox_repo_path"),
        os.getenv("SOMA_COPILOT_SWE_SANDBOX_REPO_PATH"),
        COPILOT_DEFAULT_SWE_SANDBOX_REPO_PATH,
    ):
        if isinstance(value, str) and value.strip():
            return value.strip()
    return COPILOT_DEFAULT_SWE_SANDBOX_REPO_PATH


def _resolve_swe_sandbox_image(context: RuntimeExecutionContext) -> str | None:
    for value in (
        _resolve_runtime_option(context, "copilot_swe_sandbox_image"),
        os.getenv("SOMA_COPILOT_SWE_SANDBOX_IMAGE"),
        context.run_payload.get("runtime_container_image"),
        context.instance.hidden_eval.get("docker_image"),
        context.instance.hidden_eval.get("image_name"),
    ):
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _resolve_dind_service(context: RuntimeExecutionContext) -> str:
    for value in (
        _resolve_runtime_option(context, "copilot_dind_service"),
        os.getenv("SOMA_COPILOT_DIND_SERVICE"),
        COPILOT_DEFAULT_DIND_SERVICE,
    ):
        if isinstance(value, str) and value.strip():
            return value.strip()
    return COPILOT_DEFAULT_DIND_SERVICE


def _resolve_dind_image(context: RuntimeExecutionContext) -> str:
    for value in (
        _resolve_runtime_option(context, "copilot_dind_image"),
        os.getenv("SOMA_COPILOT_DIND_IMAGE"),
        COPILOT_DEFAULT_DIND_IMAGE,
    ):
        if isinstance(value, str) and value.strip():
            return value.strip()
    return COPILOT_DEFAULT_DIND_IMAGE


def _resolve_dind_tls_enabled(context: RuntimeExecutionContext) -> bool:
    """Whether dind serves its API over mutual TLS (the default and supported path).

    Disabling this reverts to a plaintext, unauthenticated daemon on 2375 - anything that
    can route to dind then has host-root-equivalent access. Only ever useful for local
    debugging, and it forces the iptables isolation below to become mandatory.
    """
    for value in (
        _resolve_runtime_option(context, "copilot_dind_tls"),
        os.getenv("SOMA_COPILOT_DIND_TLS"),
    ):
        option = _coerce_bool_option(value)
        if option is not None:
            return option
    return COPILOT_DEFAULT_DIND_TLS


def _resolve_sandbox_gateway_mode(context: RuntimeExecutionContext) -> str:
    """
    Gateway mode for the agent's internal networks - `isolated` keeps the host off-limits.
    """
    for value in (
        _resolve_runtime_option(context, "copilot_sandbox_gateway_mode"),
        os.getenv("SOMA_COPILOT_SANDBOX_GATEWAY_MODE"),
    ):
        candidate = str(value or "").strip().lower()
        if candidate in (COPILOT_SANDBOX_GATEWAY_MODE_ISOLATED, COPILOT_SANDBOX_GATEWAY_MODE_NAT):
            return candidate
        if candidate:
            emit_progress(
                f"[copilot] ignoring unsupported sandbox gateway mode {candidate!r}; "
                f"using {COPILOT_DEFAULT_SANDBOX_GATEWAY_MODE!r}",
                component="copilot",
            )
    return COPILOT_DEFAULT_SANDBOX_GATEWAY_MODE


def _network_gateway_address(network: str) -> str:
    result = _run_command(
        ["docker", "network", "inspect", network, "--format", "{{json .IPAM.Config}}"]
    )
    if result.returncode != 0 or not (result.stdout or "").strip():
        return ""
    try:
        entries = json.loads((result.stdout or "").strip())
    except (json.JSONDecodeError, ValueError):
        return ""
    if not isinstance(entries, list):
        return ""
    for entry in entries:
        if isinstance(entry, dict) and str(entry.get("Gateway", "")).strip():
            return str(entry["Gateway"]).strip()
    return ""


def _warn_if_networks_reach_host(*, compose_project: str, network_suffixes: tuple[str, ...]) -> None:
    """Report any agent-facing network that still has a gateway, i.e. a route to the host.

    Compose applies the driver option at network *creation*, so a network left over from an
    earlier run - or one created by an engine that quietly ignored the option - can still
    carry a gateway even though the compose file asks for `isolated`. Checking the live
    network is the only way to know which of the two you actually got.
    """
    for suffix in network_suffixes:
        network = f"{compose_project}_{suffix}"
        gateway = _network_gateway_address(network)
        if gateway:
            emit_progress(
                f"[copilot] WARNING: network {network} has gateway {gateway}; the agent can "
                "reach host services bound to 0.0.0.0 through it. Expected no gateway "
                "(gateway_mode_ipv4=isolated, Docker >= 28).",
                component="copilot",
            )


def _resolve_dind_port(context: RuntimeExecutionContext) -> int:
    return (
        COPILOT_DIND_TLS_PORT
        if _resolve_dind_tls_enabled(context)
        else COPILOT_DIND_PLAINTEXT_PORT
    )


def _isolated_docker_host(*, dind_service: str, port: int = COPILOT_DIND_TLS_PORT) -> str:
    return f"tcp://{dind_service}:{port}"


def _resolve_swe_exec_broker_enabled(context: RuntimeExecutionContext) -> bool:
    """
    Whether the agent reaches the SWE sandbox through the scoped exec broker.
    """
    for value in (
        _resolve_runtime_option(context, "copilot_swe_exec_broker"),
        os.getenv("SOMA_COPILOT_SWE_EXEC_BROKER"),
    ):
        option = _coerce_bool_option(value)
        if option is not None:
            return option
    return COPILOT_DEFAULT_SWE_EXEC_BROKER


def _resolve_swe_exec_broker_port(context: RuntimeExecutionContext) -> int:
    return _coerce_positive_int_option(
        _resolve_runtime_option(context, "copilot_swe_exec_broker_port")
        or os.getenv("SOMA_COPILOT_SWE_EXEC_BROKER_PORT"),
        COPILOT_DEFAULT_SWE_EXEC_BROKER_PORT,
    )


def _resolve_swe_exec_timeout_seconds(context: RuntimeExecutionContext) -> int:
    return _coerce_positive_int_option(
        _resolve_runtime_option(context, "copilot_swe_exec_timeout_seconds")
        or os.getenv("SOMA_COPILOT_SWE_EXEC_TIMEOUT_SECONDS"),
        COPILOT_DEFAULT_SWE_EXEC_TIMEOUT_SECONDS,
    )


def _swe_exec_broker_script_path() -> Path:
    return Path(__file__).resolve().parent / "copilot-cli-container" / "swe_exec_broker.py"


def _swe_exec_client_path() -> Path:
    return Path(__file__).resolve().parent / "copilot-cli-container" / "swe-exec"


def _resolve_use_host_docker_socket(context: RuntimeExecutionContext) -> bool:
    for value in (
        _resolve_runtime_option(context, "copilot_use_host_docker_socket"),
        os.getenv("SOMA_COPILOT_USE_HOST_DOCKER_SOCKET"),
    ):
        option = _coerce_bool_option(value)
        if option is not None:
            return option
    return False


def _host_docker_binary() -> str | None:
    docker_binary = shutil.which("docker")
    if docker_binary and os.path.isabs(docker_binary):
        return docker_binary
    return None


def _docker_socket_mount_args() -> list[str]:
    """Bridge the agent straight to the host Docker daemon (legacy, insecure fallback).

    Only used when SOMA_COPILOT_USE_HOST_DOCKER_SOCKET is set - the agent then sees and can
    control every container on the host, not just its own SWE sandbox. The default path
    (dind) never grants this.
    """
    docker_sock = "/var/run/docker.sock"
    if not os.path.exists(docker_sock):
        raise RuntimeError(
            "Copilot SWE sandbox bridge requires Docker socket mount at /var/run/docker.sock, but it was not found."
        )

    args: list[str] = ["-v", f"{docker_sock}:{docker_sock}"]
    docker_binary = _host_docker_binary()
    if docker_binary is not None:
        args.extend(["-v", f"{docker_binary}:/usr/local/bin/docker:ro"])

    return args


def _resolve_prebaked_dind_repo(context: RuntimeExecutionContext) -> str | None:
    return resolve_prebaked_dind_repo(_resolve_runtime_options(context))


def _try_pull_docker_image(image_ref: str) -> bool:
    # Auth comes from DOCKERHUB_USERNAME/DOCKERHUB_TOKEN via a process-private DOCKER_CONFIG
    # (see registry_auth.py), so a private prebaked repo pulls on any host without a prior
    # `docker login` - and without depending on ~/.docker/config.json staying valid.
    result = _run_command(["docker", "pull", image_ref], env=docker_env_for_image(image_ref))
    if result.returncode == 0:
        return True
    auth_mode = "DOCKERHUB_USERNAME/DOCKERHUB_TOKEN" if uses_token_auth(image_ref) else "ambient docker config"
    emit_progress(
        f"[copilot] pull failed for prebaked dind image {image_ref!r} (auth: {auth_mode}): "
        f"{(result.stderr or result.stdout or '').strip()}",
        component="copilot",
    )
    return False


def _resolve_effective_dind_image(
    context: RuntimeExecutionContext,
    *,
    swe_sandbox_image: str,
    default_dind_image: str,
) -> str:
    dind_image = _resolve_dind_image(context)
    if dind_image != COPILOT_DEFAULT_DIND_IMAGE:
        # Explicit override (runtime option or SOMA_COPILOT_DIND_IMAGE) always wins.
        return dind_image

    repo = _resolve_prebaked_dind_repo(context)
    if not repo:
        return default_dind_image

    candidate = derive_prebaked_dind_image(context.instance.instance_id, repo=repo)
    if dind_utils.docker_image_exists(candidate) or _try_pull_docker_image(candidate):
        return candidate

    emit_progress(
        f"[copilot] prebaked dind image {candidate!r} unavailable; "
        "falling back to on-demand save/load into a plain dind daemon.",
        component="copilot",
    )
    return default_dind_image


def _start_nested_swe_sandbox(
    *,
    dind_container_id: str,
    image: str,
    container_name: str,
    workspace_container_path: str,
    repo_path: str,
) -> str:
    # Fresh dind per run, but guard against a leftover container from a crashed prior attempt.
    _run_command(["docker", "exec", dind_container_id, "docker", "rm", "-f", container_name])
    run_result = _run_command([
        "docker", "exec", dind_container_id, "docker", "run", "-d",
        "--name", container_name,
        "-v", f"{workspace_container_path}:{repo_path}",
        image,
        "sh", "-lc", "while true; do sleep 3600; done",
    ])
    if run_result.returncode != 0:
        raise RuntimeError(
            "Failed to start Copilot SWE sandbox inside the isolated Docker daemon (dind). "
            f"{(run_result.stderr or run_result.stdout or '').strip()}"
        )
    container_id = (run_result.stdout or "").strip()
    if not container_id:
        raise RuntimeError(
            "Copilot SWE sandbox did not report a container id after starting inside the isolated Docker daemon."
        )
    return container_id


# SWE-bench's own harness builds every task image around a conda env named exactly
# "testbed" (see e.g. its setup_env.sh / constants.py) - we saw this hold across every repo
# family tested (django, sympy, pytest, requests, astropy). That env is not on PATH nor
# activated by default in a one-off shell: conda activation is a bash function sourced from
# conda.sh, which `bash -lc`/`sh -lc` never runs on their own, so every agent independently
# rediscovers the right interpreter path by trial and error each run. This probe is
# defensive rather than hardcoded to that one convention: it checks the common install
# roots directly, then falls back to asking conda itself, and gives up cleanly (empty
# string) if it finds nothing - swe-exec then just leaves PATH untouched.
_SWE_SANDBOX_ENV_DISCOVERY_SCRIPT = r"""
for root in /opt/miniconda3 /opt/miniconda /opt/conda /root/miniconda3 /root/miniconda /usr/local/miniconda3; do
  if [ -x "$root/envs/testbed/bin/python" ]; then
    echo "$root/envs/testbed/bin"
    exit 0
  fi
done
if command -v conda >/dev/null 2>&1; then
  envdir="$(conda env list 2>/dev/null | awk '$1=="testbed"{print $NF}')"
  if [ -n "$envdir" ] && [ -x "$envdir/bin/python" ]; then
    echo "$envdir/bin"
    exit 0
  fi
fi
exit 1
"""


def _discover_swe_sandbox_env_bin_dir(*, dind_container_id: str, sandbox_container_id: str) -> str:
    """Best-effort discovery of the task env's interpreter directory, for swe-exec's PATH.

    Reaches the nested sandbox container the same way _start_nested_swe_sandbox does: a
    host-side `docker exec` into dind, which in turn execs into the sandbox container on its
    own isolated daemon. Never raises - a failed probe just means swe-exec falls back to
    whatever PATH the sandbox image's own shell resolves, i.e. today's behavior.
    """
    result = _run_command([
        "docker", "exec", dind_container_id,
        "docker", "exec", sandbox_container_id,
        "sh", "-c", _SWE_SANDBOX_ENV_DISCOVERY_SCRIPT,
    ])
    if result.returncode != 0:
        return ""
    return (result.stdout or "").strip().splitlines()[0] if (result.stdout or "").strip() else ""


def _start_swe_exec_broker(
    *,
    compose_prefix: list[str],
    cwd: Path,
    env: dict[str, str],
    broker_port: int,
) -> str:
    """Bring up the scoped exec broker and wait until it answers /healthz.

    Started after the nested sandbox exists because the broker pins one container id and
    refuses to exec anywhere else.
    """
    up_result = _run_command(
        [*compose_prefix, "up", "-d", COPILOT_SWE_EXEC_BROKER_SERVICE],
        cwd=cwd,
        env=env,
    )
    if up_result.returncode != 0:
        raise RuntimeError(
            "Failed to start the Copilot SWE exec broker. "
            f"{(up_result.stderr or up_result.stdout or '').strip()}"
        )

    ps_result = _run_command(
        [*compose_prefix, "ps", "-q", COPILOT_SWE_EXEC_BROKER_SERVICE],
        cwd=cwd,
        env=env,
    )
    broker_container_id = (
        (ps_result.stdout or "").strip().splitlines()[0]
        if (ps_result.stdout or "").strip()
        else ""
    )
    if not broker_container_id:
        raise RuntimeError(
            "Copilot SWE exec broker is not running after compose up. "
            f"{(ps_result.stderr or ps_result.stdout or '').strip()}"
        )

    # Probed from inside the broker container: both of its networks are internal, so the
    # host has no route to it.
    probe_script = (
        "import urllib.request;"
        f"urllib.request.urlopen('http://127.0.0.1:{broker_port}/healthz', timeout=2).read()"
    )
    last_error = ""
    for _ in range(COPILOT_SWE_EXEC_BROKER_READY_RETRIES):
        probe = _run_command(
            ["docker", "exec", broker_container_id, "python", "-c", probe_script]
        )
        if probe.returncode == 0:
            return broker_container_id
        last_error = (probe.stderr or probe.stdout or "").strip()
        if not _docker_container_running(broker_container_id):
            logs = _run_command(["docker", "logs", "--tail", "40", broker_container_id])
            raise RuntimeError(
                "Copilot SWE exec broker exited during startup. "
                f"{(logs.stderr or logs.stdout or '').strip()}"
            )
        time.sleep(COPILOT_SWE_EXEC_BROKER_READY_SLEEP_SECONDS)

    raise RuntimeError(f"Copilot SWE exec broker did not become ready: {last_error}")


def _docker_container_running(container: str) -> bool:
    result = _run_command(["docker", "inspect", "-f", "{{.State.Running}}", container])
    return result.returncode == 0 and (result.stdout or "").strip() == "true"


def _resolve_proxy_port(context: RuntimeExecutionContext, *, proxy_service: str) -> int:
    for value in (
        _resolve_runtime_option(context, "copilot_proxy_port"),
        os.getenv("SOMA_COPILOT_PROXY_PORT"),
    ):
        if value is None:
            continue
        try:
            port = int(value)
        except (TypeError, ValueError):
            continue
        if port > 0:
            return port
    if proxy_service == COPILOT_COMPRESSION_SERVICE_NAME:
        return COPILOT_COMPRESSION_SERVICE_PORT
    return COPILOT_DEFAULT_PROXY_PORT


def _resolve_copilot_run_id_header_value(context: RuntimeExecutionContext) -> str:
    for value in (
        _resolve_runtime_option(context, "copilot_run_id_header_value"),
        os.getenv("SOMA_COPILOT_RUN_ID_HEADER_VALUE"),
    ):
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _normalize_run_id(value: str) -> str:
    sanitized = "".join(char for char in value if char.isalnum() or char in {"-", "_"})
    return sanitized[:40]


def _resolve_execution_run_id(context: RuntimeExecutionContext) -> str:
    for value in (
        _resolve_runtime_option(context, "copilot_run_id"),
        os.getenv("SOMA_COPILOT_RUN_ID"),
    ):
        if isinstance(value, str) and value.strip():
            normalized = _normalize_run_id(value.strip())
            if normalized:
                context.run_payload["_copilot_resolved_run_id"] = normalized
                return normalized

    cached = context.run_payload.get("_copilot_resolved_run_id")
    if isinstance(cached, str) and cached.strip():
        return cached.strip()

    generated = uuid.uuid4().hex[:12]
    context.run_payload["_copilot_resolved_run_id"] = generated
    return generated


def _resolve_compose_project_name(context: RuntimeExecutionContext) -> str:
    for value in (
        os.getenv("SOMA_COPILOT_COMPOSE_PROJECT"),
        _resolve_runtime_option(context, "copilot_compose_project"),
    ):
        if isinstance(value, str) and value.strip():
            return value.strip()

    run_id = _resolve_execution_run_id(context)
    digest = hashlib.sha256(
        f"{context.output_dir.resolve()}::{run_id}".encode("utf-8")
    ).hexdigest()[:10]
    return f"soma-copilot-{digest}"


def _resolve_shared_proxy_batch_enabled(context: RuntimeExecutionContext) -> bool:
    for value in (
        os.getenv("SOMA_COPILOT_SHARED_PROXY"),
        _resolve_runtime_option(context, "copilot_shared_proxy"),
    ):
        option = _coerce_bool_option(value)
        if option is not None:
            return option
    return COPILOT_DEFAULT_SHARED_PROXY_BATCH


def _resolve_shared_proxy_teardown_enabled(context: RuntimeExecutionContext) -> bool:
    for value in (
        os.getenv("SOMA_COPILOT_SHARED_PROXY_TEARDOWN"),
        _resolve_runtime_option(context, "copilot_shared_proxy_teardown"),
    ):
        option = _coerce_bool_option(value)
        if option is not None:
            return option
    return COPILOT_DEFAULT_SHARED_PROXY_TEARDOWN


def _with_runtime_option_overrides(
    context: RuntimeExecutionContext,
    overrides: dict[str, Any],
) -> RuntimeExecutionContext:
    run_payload = dict(context.run_payload)
    runtime_options = dict(_resolve_runtime_options(context))
    runtime_options.update(overrides)
    run_payload["runtime_options"] = runtime_options
    return RuntimeExecutionContext(
        backend_name=context.backend_name,
        instance=context.instance,
        run_payload=run_payload,
        llm_config=context.llm_config,
        workspace=context.workspace,
        max_iterations=context.max_iterations,
        output_dir=context.output_dir,
        instance_dir=context.instance_dir,
    )


def _teardown_shared_compose_stack(context: RuntimeExecutionContext, *, compose_project: str) -> None:
    compose_file = _resolve_compose_file(context)
    env = os.environ.copy()
    existing_profiles = str(env.get("COMPOSE_PROFILES", "")).strip()
    if existing_profiles:
        profiles = {item.strip() for item in existing_profiles.split(",") if item.strip()}
        profiles.add("copilot-sidecars")
        env["COMPOSE_PROFILES"] = ",".join(sorted(profiles))
    else:
        env["COMPOSE_PROFILES"] = "copilot-sidecars"

    down_result = _run_command(
        [
            "docker",
            "compose",
            "-f",
            str(compose_file),
            "--project-name",
            compose_project,
            "down",
            "--remove-orphans",
            "--volumes",
        ],
        env=env,
    )
    if down_result.returncode != 0:
        emit_progress(
            "[copilot] failed to tear down shared compose stack: "
            f"{(down_result.stderr or down_result.stdout or '').strip()}",
            component="copilot",
        )


def _normalize_proxy_upstream_base_url(value: str) -> str:
    parsed = urlsplit(value)
    if not parsed.scheme or not parsed.netloc:
        raise RuntimeError("Copilot LLM base URL must be an absolute http(s) URL.")
    path = parsed.path or "/"
    if not path.endswith("/"):
        path = f"{path}/"
    return urlunsplit((parsed.scheme, parsed.netloc, path, "", ""))


def _workspace_volume_name(*, compose_project: str) -> str:
    return f"{compose_project}_{COPILOT_WORKSPACE_VOLUME_BASENAME}"


def _resolve_compose_service_image(
    *,
    compose_file: Path,
    compose_project: str,
    service: str,
    env: dict[str, str] | None = None,
) -> str:
    """Image Compose will actually run for `service` (falls back to the built-in default)."""
    result = _run_command(
        [
            "docker",
            "compose",
            "-f",
            str(compose_file),
            "--project-name",
            compose_project,
            "config",
            "--images",
            service,
        ],
        env=env,
    )
    if result.returncode == 0:
        for line in (result.stdout or "").splitlines():
            candidate = line.strip()
            if candidate:
                return candidate
    emit_progress(
        f"[copilot] could not resolve the image for compose service {service!r}; "
        f"assuming {COPILOT_DEFAULT_IMAGE}",
        component="copilot",
    )
    return COPILOT_DEFAULT_IMAGE


def _resolve_image_user(image: str) -> tuple[int, int] | None:
    """Numeric uid:gid the agent CLI runs as inside `image`, or None if unresolvable.

    `docker image inspect` only reports the configured user *name* (`copilot`), so the
    id has to be read from inside the image itself.
    """
    result = _run_command(
        ["docker", "run", "--rm", "--entrypoint", "sh", image, "-c", "id -u; id -g"]
    )
    ids = [line.strip() for line in (result.stdout or "").splitlines() if line.strip()]
    if result.returncode != 0 or len(ids) < 2 or not all(value.isdigit() for value in ids[:2]):
        emit_progress(
            f"[copilot] could not resolve the container user of {image!r}: "
            f"{(result.stderr or result.stdout or '').strip()}",
            component="copilot",
        )
        return None
    return int(ids[0]), int(ids[1])


def _seed_workspace_volume(
    *,
    compose_project: str,
    repo_root: Path,
    owner: tuple[int, int] | None = None,
    sandbox_image: str | None = None,
    sandbox_repo_path: str | None = None,
    base_commit: str | None = None,
) -> str:
    volume_name = _workspace_volume_name(compose_project=compose_project)
    _run_command(["docker", "volume", "rm", "-f", volume_name])
    create_result = _run_command(["docker", "volume", "create", volume_name])
    if create_result.returncode != 0:
        raise RuntimeError(
            "Failed to create Copilot workspace Docker volume. "
            f"{(create_result.stderr or create_result.stdout or '').strip()}"
        )

    seeded_from_image = False
    if sandbox_image and sandbox_repo_path:
        seeded_from_image = _seed_workspace_volume_from_sandbox_image(
            volume_name=volume_name,
            sandbox_image=sandbox_image,
            sandbox_repo_path=sandbox_repo_path,
            base_commit=base_commit,
            owner=owner,
        )
        if not seeded_from_image:
            emit_progress(
                "[copilot] could not seed workspace from the SWE sandbox image "
                f"{sandbox_image!r}; falling back to a bare checkout of {repo_root} "
                "(the task image's pre-built dependencies, compiled extensions and any "
                "generated version files will NOT be present)",
                component="copilot",
            )

    if not seeded_from_image:
        _seed_workspace_volume_from_repo_root(volume_name=volume_name, repo_root=repo_root, owner=owner)
    return volume_name


def _seed_workspace_volume_from_repo_root(
    *,
    volume_name: str,
    repo_root: Path,
    owner: tuple[int, int] | None = None,
) -> None:
    # The copy runs as root, which would leave the whole workspace root-owned - the agent
    # CLI runs as a non-root user (`USER copilot`), so its file tools would then fail with
    # EACCES on every write under the repo. Handing ownership to that user keeps the agent
    # unprivileged instead of granting it root.
    seed_script = "rm -rf /workspace/* /workspace/.[!.]* /workspace/..?* 2>/dev/null || true; cp -a /src/. /workspace/"
    if owner is not None:
        seed_script += f"; chown -R {owner[0]}:{owner[1]} /workspace"

    copy_result = _run_command(
        [
            "docker",
            "run",
            "--rm",
            "-v",
            f"{volume_name}:/workspace",
            "-v",
            f"{repo_root.resolve()}:/src:ro",
            "alpine:3.20",
            "sh",
            "-lc",
            seed_script,
        ]
    )
    if copy_result.returncode != 0:
        _remove_workspace_volume(volume_name=volume_name)
        raise RuntimeError(
            "Failed to seed Copilot workspace volume from repository checkout. "
            f"{(copy_result.stderr or copy_result.stdout or '').strip()}"
        )


def _seed_workspace_volume_from_sandbox_image(
    *,
    volume_name: str,
    sandbox_image: str,
    sandbox_repo_path: str,
    base_commit: str | None,
    owner: tuple[int, int] | None = None,
) -> bool:
    """Seed the workspace volume from the SWE-bench task image's own checkout.

    A bare external `git clone` (the previous, only, seeding path) can never reproduce what
    the task image already has at its repo path: dependencies installed, native extensions
    compiled, and any version file a build step generates (setuptools_scm's is gitignored by
    convention - a fresh clone is guaranteed not to have it). Copying the image's own
    checkout instead preserves all of that; we only need to move it from whatever commit the
    image's dependency-install step left it at over to the instance's `base_commit`.

    Intentionally does NOT run `git clean -dfx` afterward: that would delete exactly the
    untracked/gitignored build artifacts (compiled extensions, generated version files) this
    function exists to keep. `git checkout --force` already replaces every tracked file with
    the `base_commit` version and removes tracked files that commit doesn't have; leaving
    untracked build output in place is the point, not an oversight.

    Returns False (leaving the volume untouched for the caller's fallback) on any failure -
    this is a best-effort improvement, not a hard requirement for the run to proceed.
    """
    seed_mount = "/__soma_seed_target"
    seed_script = (
        f"rm -rf {seed_mount}/* {seed_mount}/.[!.]* {seed_mount}/..?* 2>/dev/null || true; "
        f"cp -a {sandbox_repo_path}/. {seed_mount}/"
    )
    copy_result = _run_command(
        [
            "docker",
            "run",
            "--rm",
            "--entrypoint",
            "sh",
            "-v",
            f"{volume_name}:{seed_mount}",
            sandbox_image,
            "-lc",
            seed_script,
        ]
    )
    if copy_result.returncode != 0:
        emit_progress(
            f"[copilot] seeding workspace from sandbox image {sandbox_image!r} failed: "
            f"{(copy_result.stderr or copy_result.stdout or '').strip()}",
            component="copilot",
        )
        return False

    if base_commit:
        checkout_script = (
            f"git config --global --add safe.directory {sandbox_repo_path} 2>/dev/null || true; "
            f"cd {sandbox_repo_path} && git checkout --force {shlex.quote(base_commit)}"
        )
        checkout_result = _run_command(
            [
                "docker",
                "run",
                "--rm",
                "--entrypoint",
                "sh",
                "-v",
                f"{volume_name}:{sandbox_repo_path}",
                sandbox_image,
                "-lc",
                checkout_script,
            ]
        )
        if checkout_result.returncode != 0:
            emit_progress(
                f"[copilot] could not checkout base_commit {base_commit} onto the "
                f"image-seeded workspace: "
                f"{(checkout_result.stderr or checkout_result.stdout or '').strip()}",
                component="copilot",
            )
            return False

    if owner is not None:
        chown_result = _run_command(
            [
                "docker",
                "run",
                "--rm",
                "-v",
                f"{volume_name}:/workspace",
                "alpine:3.20",
                "chown",
                "-R",
                f"{owner[0]}:{owner[1]}",
                "/workspace",
            ]
        )
        if chown_result.returncode != 0:
            emit_progress(
                f"[copilot] could not chown image-seeded workspace to {owner[0]}:{owner[1]}: "
                f"{(chown_result.stderr or chown_result.stdout or '').strip()}",
                component="copilot",
            )
            return False

    return True


def _remove_workspace_volume(*, volume_name: str) -> None:
    result = _run_command(["docker", "volume", "rm", "-f", volume_name])
    if result.returncode != 0:
        details = (result.stderr or result.stdout or "").strip()
        if "No such volume" in details:
            return
        emit_progress(
            f"[copilot] failed to remove workspace volume {volume_name}: {details}",
            component="copilot",
        )


def _capture_workspace_patch(*, volume_name: str, tmp_run_dir: Path, base_commit: str | None = None) -> dict[str, Any]:
    patch_eval_dir = tmp_run_dir / "patch-eval"
    patch_eval_dir.mkdir(parents=True, exist_ok=True)
    patch_path = patch_eval_dir / "agent.patch"
    snapshot_dir = patch_eval_dir / "workspace-snapshot"
    if snapshot_dir.exists():
        shutil.rmtree(snapshot_dir, ignore_errors=True)
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    copy_result = _run_command(
        [
            "docker",
            "run",
            "--rm",
            "-v",
            f"{volume_name}:/workspace:ro",
            "-v",
            f"{snapshot_dir.resolve()}:/snapshot",
            "alpine:3.20",
            "sh",
            "-lc",
            "cp -a /workspace/. /snapshot/",
        ]
    )
    if copy_result.returncode != 0:
        patch_path.write_text("", encoding="utf-8")
        return {
            "status": "error",
            "error": (
                copy_result.stderr
                or copy_result.stdout
                or "failed to copy workspace volume snapshot for patch capture"
            ).strip(),
            "repo_dir": str(snapshot_dir),
            "patch_path": str(patch_path),
            "has_changes": False,
            "line_count": 0,
            "size_bytes": 0,
        }

    return capture_repo_patch(repo_dir=snapshot_dir, output_dir=patch_eval_dir, base_ref=base_commit)


def _run_command(
    args: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        cwd=str(cwd) if cwd is not None else None,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def _ensure_bridge_netfilter() -> bool:
    """Load br_netfilter and enable bridge iptables filtering if not already active.

    Same-bridge Docker traffic bypasses iptables FORWARD by default (L2 path).
    This is required once per host before iptables DROP rules take effect.
    Non-fatal: returns False with a warning when privileges are insufficient.
    """
    sysctl_path = Path("/proc/sys/net/bridge/bridge-nf-call-iptables")

    if not Path("/sys/module/br_netfilter").is_dir():
        result = _run_command(["modprobe", "br_netfilter"])
        if result.returncode != 0:
            emit_progress(
                "[copilot] br_netfilter could not be loaded — iptables isolation will not be "
                f"effective: {(result.stderr or result.stdout or '').strip()}",
                component="copilot",
            )
            return False
        emit_progress("[copilot] loaded br_netfilter kernel module", component="copilot")

    if not sysctl_path.is_file():
        emit_progress(
            "[copilot] bridge-nf-call-iptables sysctl unavailable — isolation not effective",
            component="copilot",
        )
        return False

    try:
        if sysctl_path.read_text().strip() != "1":
            result = _run_command(["sysctl", "-w", "net.bridge.bridge-nf-call-iptables=1"])
            if result.returncode != 0:
                emit_progress(
                    f"[copilot] failed to set bridge-nf-call-iptables=1 — isolation not effective: "
                    f"{(result.stderr or result.stdout or '').strip()}",
                    component="copilot",
                )
                return False
            emit_progress("[copilot] set net.bridge.bridge-nf-call-iptables=1", component="copilot")
    except OSError as exc:
        emit_progress(f"[copilot] bridge-nf-call-iptables read/write error: {exc}", component="copilot")
        return False

    return True


def _get_container_network_ips(container_name: str) -> dict[str, str]:
    result = _run_command(["docker", "inspect", "--format", "{{json .NetworkSettings.Networks}}", container_name])
    if result.returncode != 0 or not (result.stdout or "").strip():
        return {}
    try:
        networks = json.loads((result.stdout or "").strip())
    except (json.JSONDecodeError, ValueError):
        return {}
    return {
        name: info.get("IPAddress", "")
        for name, info in networks.items()
        if isinstance(info, dict) and info.get("IPAddress")
    }


def _iptables_purge_stale_soma_rules() -> None:
    """Remove any leftover soma-iso-* rules from crashed/killed previous runs.

    When a benchmark process is SIGKILL-ed the finally block may not run, leaving
    rules that reference IPs Docker has since reassigned to different containers.
    """
    for chain in ("DOCKER-USER", "FORWARD"):
        while True:
            result = _run_command(["iptables", "-L", chain, "--line-numbers", "-n"])
            if result.returncode != 0:
                break
            line_to_delete = None
            for line in (result.stdout or "").splitlines():
                if "soma-iso-" in line:
                    try:
                        line_to_delete = line.split()[0]
                    except IndexError:
                        pass
                    break
            if line_to_delete is None:
                break
            _run_command(["iptables", "-D", chain, line_to_delete])


def _iptables_insert_drop(*, src_ip: str, dst_ip: str, comment: str) -> bool:
    chain = "DOCKER-USER"
    if _run_command(["iptables", "-L", chain, "-n"]).returncode != 0:
        chain = "FORWARD"
    # --ctstate NEW: only block connection initiations, not reply packets for
    # connections proxy opened toward compression-service.
    result = _run_command([
        "iptables", "-I", chain, "1",
        "-s", src_ip, "-d", dst_ip, "-p", "tcp",
        "-m", "conntrack", "--ctstate", "NEW",
        "-m", "comment", "--comment", comment,
        "-j", "DROP",
    ])
    return result.returncode == 0


def _iptables_delete_drop(*, src_ip: str, dst_ip: str, comment: str) -> None:
    for chain in ("DOCKER-USER", "FORWARD"):
        _run_command([
            "iptables", "-D", chain,
            "-s", src_ip, "-d", dst_ip, "-p", "tcp",
            "-m", "conntrack", "--ctstate", "NEW",
            "-m", "comment", "--comment", comment,
            "-j", "DROP",
        ])


def _apply_proxy_compression_isolation(
    *,
    compose_project: str,
    proxy_service: str | None = None,
    dind_service: str | None = None,
    exec_broker_service: str | None = None,
    require_dind_isolation: bool = False,
    purge_stale_rules: bool = True,
) -> list[tuple[str, str, str]]:
    """
    Fence the miner's compression service off from the rest of the sandbox network.
    Returns inserted rules as (src_ip, dst_ip, comment) tuples for later removal.
    """
    if not _ensure_bridge_netfilter():
        if require_dind_isolation:
            raise RuntimeError(
                "Copilot requires iptables isolation for the dind API because mutual TLS is "
                "disabled (SOMA_COPILOT_DIND_TLS=false), but bridge netfilter is unavailable. "
                "Re-enable TLS or run on a host where iptables can filter bridged traffic."
            )
        return []

    if purge_stale_rules:
        _iptables_purge_stale_soma_rules()

    comp_name = f"{compose_project}-{COPILOT_COMPRESSION_SERVICE_NAME}-1"
    comp_ips = _get_container_network_ips(comp_name)
    sandbox_key = next((k for k in comp_ips if "sandbox" in k), None)
    comp_ip = comp_ips.get(sandbox_key, "") if sandbox_key else next(iter(comp_ips.values()), "")

    def _peer_ip(service: str) -> str:
        peer_ips = _get_container_network_ips(f"{compose_project}-{service}-1")
        if sandbox_key and sandbox_key in peer_ips:
            return peer_ips[sandbox_key]
        return next(iter(peer_ips.values()), "")

    comment = f"soma-iso-{compose_project[-16:]}"
    rules: list[tuple[str, str, str]] = []

    targets: list[tuple[str, str]] = []
    if proxy_service:
        targets.append(("proxy", proxy_service))
    if dind_service:
        targets.append(("dind", dind_service))
    if exec_broker_service:
        targets.append(("swe-exec-broker", exec_broker_service))

    for label, service in targets:
        peer_ip = _peer_ip(service)
        if not comp_ip or not peer_ip:
            message = (
                f"[copilot] iptables isolation skipped for {label}: could not resolve "
                f"container IPs (comp={comp_ip!r} {label}={peer_ip!r})"
            )
            if label == "dind" and require_dind_isolation:
                raise RuntimeError(message.replace("[copilot] ", ""))
            emit_progress(message, component="copilot")
            continue

        if _iptables_insert_drop(src_ip=comp_ip, dst_ip=peer_ip, comment=comment):
            emit_progress(
                f"[copilot] iptables DROP: compression ({comp_ip}) -> {label} ({peer_ip}) tcp",
                component="copilot",
            )
            rules.append((comp_ip, peer_ip, comment))
            continue

        message = (
            f"[copilot] iptables DROP rule insert failed for {label} "
            "(insufficient privileges?)"
        )
        if label == "dind" and require_dind_isolation:
            raise RuntimeError(message.replace("[copilot] ", ""))
        emit_progress(message, component="copilot")

    return rules


def _remove_proxy_compression_isolation(rules: list[tuple[str, str, str]]) -> None:
    for src_ip, dst_ip, comment in rules:
        _iptables_delete_drop(src_ip=src_ip, dst_ip=dst_ip, comment=comment)
        emit_progress(
            f"[copilot] iptables removed isolation rule: ({src_ip}) -> ({dst_ip})",
            component="copilot",
        )


def _resolve_repo_clone_url(repo: str) -> str:
    normalized = repo.strip()
    if not normalized:
        raise RuntimeError("Benchmark instance does not define repository slug.")
    if normalized.startswith(("http://", "https://", "git@")):
        return normalized
    if "/" not in normalized:
        raise RuntimeError(f"Unsupported benchmark repository value: {repo!r}")
    return f"https://github.com/{normalized}.git"


def _resolve_checkout_workspace_root(context: RuntimeExecutionContext) -> Path:
    run_root = _resolve_run_root(context)
    instance_root = _resolve_instance_run_root(context, run_root=run_root)
    return instance_root


def _resolve_run_root(context: RuntimeExecutionContext) -> Path:
    override = _resolve_runtime_option(context, "copilot_run_root")
    if not isinstance(override, str) or not override.strip():
        override = os.getenv("SOMA_COPILOT_RUN_ROOT")
    if not isinstance(override, str) or not override.strip():
        # Backward compatible fallback: old dedicated tmp root still works if explicitly set.
        override = os.getenv("SOMA_COPILOT_TMP_ROOT")

    if isinstance(override, str) and override.strip():
        base_root = Path(override.strip()).expanduser().resolve()
    else:
        base_root = (Path(tempfile.gettempdir()) / COPILOT_DEFAULT_RUN_ROOT_DIRNAME).resolve()

    _maybe_cleanup_stale_run_dirs(context, base_root=base_root)

    run_id = _resolve_execution_run_id(context)
    return base_root / f"run-{run_id}"


def _resolve_instance_run_root(context: RuntimeExecutionContext, *, run_root: Path) -> Path:
    instance_key = str(context.instance.benchmark_id or context.instance.instance_id or "instance")
    instance_digest = hashlib.sha256(instance_key.encode("utf-8")).hexdigest()[:12]
    return run_root / f"instance-{instance_digest}"


def _resolve_tmp_run_root(context: RuntimeExecutionContext) -> Path:
    run_root = _resolve_run_root(context)
    instance_root = _resolve_instance_run_root(context, run_root=run_root)
    return instance_root


def _contains_output_format_arg(args: list[str]) -> bool:
    for idx, item in enumerate(args):
        if item == "--output-format":
            return True
        if item.startswith("--output-format="):
            return True
        if item == "-o" and idx + 1 < len(args):
            return True
    return False


def _contains_stream_arg(args: list[str]) -> bool:
    for idx, item in enumerate(args):
        if item == "--stream" and idx + 1 < len(args):
            return True
        if item.startswith("--stream="):
            return True
    return False


def _write_copilot_trajectory(
    *,
    stdout: str,
    output_format: str,
    destination_path: Path,
) -> Path | None:
    if output_format != "json":
        return None
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    trajectory_path = destination_path
    trajectory_path.write_text(_extract_jsonl_lines(stdout), encoding="utf-8")
    return trajectory_path


def _count_trajectory_steps(trajectory_path: Path) -> int | None:
    """Count agent steps from a copilot trajectory JSONL file.

    Each agent runs its own 0-indexed ``turnId`` sequence: the main agent's
    turns carry no ``agentId``, while every sub-agent spawned via the ``task``
    tool emits its own turns tagged with the spawning tool call's ``agentId``.
    A single global "last turnId" therefore misses all sub-agent steps, so we
    group turns by ``agentId`` and sum ``max(turnId) + 1`` across every agent.

    Falls back to counting assistant.turn_start events if no turn carries a
    usable ``turnId``.
    """
    try:
        if not trajectory_path.is_file():
            return None
        max_turn_id: dict[object, int] = {}
        turn_start_count = 0
        with trajectory_path.open(encoding="utf-8") as fh:
            for raw_line in fh:
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                entry_type = entry.get("type")
                if entry_type == "assistant.turn_start":
                    turn_start_count += 1
                elif entry_type != "assistant.turn_end":
                    continue
                # Both turn_start and turn_end carry the turnId; using either
                # keeps the count correct even if one side is missing.
                turn_id_raw = (entry.get("data") or {}).get("turnId")
                if turn_id_raw is None:
                    continue
                try:
                    turn_id = int(turn_id_raw)
                except (TypeError, ValueError):
                    continue
                agent_id = entry.get("agentId")
                if agent_id not in max_turn_id or turn_id > max_turn_id[agent_id]:
                    max_turn_id[agent_id] = turn_id
        if max_turn_id:
            return sum(turn_id + 1 for turn_id in max_turn_id.values())
        if turn_start_count > 0:
            return turn_start_count
        return None
    except Exception:
        return None


def _extract_jsonl_lines(stdout: str) -> str:
    valid_lines: list[str] = []
    for raw_line in stdout.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            json.loads(line)
        except json.JSONDecodeError:
            continue
        valid_lines.append(line)
    if not valid_lines:
        return ""
    return "\n".join(valid_lines) + "\n"


def _run_copilot_command_streaming(
    *,
    command: list[str],
    cwd: Path,
    env: dict[str, str],
    trajectory_path: Path | None,
    stream_log_path: Path | None,
    timeout_seconds: float | None = None,
) -> tuple[subprocess.CompletedProcess[str], bool]:
    stdout_chunks: list[str] = []
    stderr_chunks: list[str] = []
    trajectory_handle: Any = None
    stream_log_handle: Any = None
    try:
        if trajectory_path is not None:
            trajectory_path.parent.mkdir(parents=True, exist_ok=True)
            trajectory_handle = trajectory_path.open("w", encoding="utf-8")
        if stream_log_path is not None:
            stream_log_path.parent.mkdir(parents=True, exist_ok=True)
            stream_log_handle = stream_log_path.open("w", encoding="utf-8")

        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=1,
        )

        def _consume_stdout(stream: Any, sink: list[str]) -> None:
            if stream is None:
                return
            for chunk in iter(stream.readline, ""):
                sink.append(chunk)
                if stream_log_handle is not None:
                    stream_log_handle.write(chunk)
                    stream_log_handle.flush()
                if trajectory_handle is None:
                    continue
                line = chunk.strip()
                if not line:
                    continue
                try:
                    json.loads(line)
                except json.JSONDecodeError:
                    continue
                trajectory_handle.write(line + "\n")
                trajectory_handle.flush()
            stream.close()

        def _consume_stream(stream: Any, sink: list[str]) -> None:
            if stream is None:
                return
            for chunk in iter(stream.readline, ""):
                sink.append(chunk)
            stream.close()

        stdout_thread = threading.Thread(
            target=_consume_stdout,
            args=(process.stdout, stdout_chunks),
            daemon=True,
        )
        stderr_thread = threading.Thread(
            target=_consume_stream,
            args=(process.stderr, stderr_chunks),
            daemon=True,
        )
        stdout_thread.start()
        stderr_thread.start()

        timed_out = False
        if timeout_seconds is not None:
            try:
                process.wait(timeout=timeout_seconds)
            except subprocess.TimeoutExpired:
                timed_out = True
                process.terminate()
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()

        stdout_thread.join()
        stderr_thread.join()

        returncode = process.wait()
        stdout_text = "".join(stdout_chunks)
        if trajectory_path is not None and trajectory_path.stat().st_size == 0:
            trajectory_path.write_text(_extract_jsonl_lines(stdout_text), encoding="utf-8")

        return subprocess.CompletedProcess(
            args=command,
            returncode=returncode,
            stdout=stdout_text,
            stderr="".join(stderr_chunks),
        ), timed_out
    finally:
        if trajectory_handle is not None:
            trajectory_handle.close()
        if stream_log_handle is not None:
            stream_log_handle.close()


def _collect_compose_service_logs(
    *,
    compose_prefix: list[str],
    cwd: Path,
    env: dict[str, str],
    services: list[str],
    destination_dir: Path,
) -> dict[str, str]:
    destination_dir.mkdir(parents=True, exist_ok=True)
    log_paths: dict[str, str] = {}

    for service_name in services:
        result = _run_command(
            [*compose_prefix, "logs", "--timestamps", "--no-color", service_name],
            cwd=cwd,
            env=env,
        )
        log_path = destination_dir / f"{service_name}.log"
        content = result.stdout or ""
        if result.stderr:
            content = f"{content}\n# stderr\n{result.stderr}"
        log_path.write_text(content, encoding="utf-8")
        log_paths[f"{service_name}_log"] = str(log_path)

    return log_paths


_PROXY_TOKEN_USAGE_MARKER = "[proxy][token-usage] "


def _extract_proxy_token_usage(*, sidecar_log_paths: dict[str, str]) -> dict[str, int]:
    proxy_log_path = sidecar_log_paths.get("proxy_log", "")
    if not proxy_log_path:
        return {}
    log_file = Path(proxy_log_path)
    if not log_file.is_file():
        return {}
    last_usage: dict[str, int] = {}
    for line in log_file.read_text(encoding="utf-8", errors="replace").splitlines():
        idx = line.find(_PROXY_TOKEN_USAGE_MARKER)
        if idx == -1:
            continue
        try:
            parsed = json.loads(line[idx + len(_PROXY_TOKEN_USAGE_MARKER):])
            if isinstance(parsed, dict):
                last_usage = {k: v for k, v in parsed.items() if isinstance(v, int)}
        except (json.JSONDecodeError, ValueError):
            continue
    return last_usage


def _extract_message_request_logs(*, sidecar_log_paths: dict[str, str], destination_dir: Path) -> dict[str, str]:
    in_marker = "[compression-service][messages.in] "
    out_marker = "[compression-service][messages.out] "
    incoming_entries: list[str] = []
    outgoing_entries: list[str] = []

    candidate_logs = [
        sidecar_log_paths.get("compression-service_log", ""),
        sidecar_log_paths.get("proxy_log", ""),
    ]
    for raw_path in candidate_logs:
        if not raw_path:
            continue
        candidate_path = Path(raw_path)
        if not candidate_path.is_file():
            continue
        for line in candidate_path.read_text(encoding="utf-8").splitlines():
            if in_marker in line:
                incoming_entries.append(line.split(in_marker, 1)[1].strip())
            if out_marker in line:
                outgoing_entries.append(line.split(out_marker, 1)[1].strip())

    extracted_paths: dict[str, str] = {}
    if incoming_entries:
        incoming_path = destination_dir / "messages-in.jsonl"
        incoming_path.write_text("\n".join(incoming_entries) + "\n", encoding="utf-8")
        extracted_paths["messages_in_path"] = str(incoming_path)

    if outgoing_entries:
        outgoing_path = destination_dir / "messages-out.jsonl"
        outgoing_path.write_text("\n".join(outgoing_entries) + "\n", encoding="utf-8")
        extracted_paths["messages_out_path"] = str(outgoing_path)

    return extracted_paths


def _prepare_instance_repo_checkout(context: RuntimeExecutionContext) -> tuple[Path, str, str]:
    repo_slug = str(context.instance.repo or "").strip()
    base_commit = str(context.instance.base_commit or "").strip()
    if not repo_slug:
        raise RuntimeError("Benchmark instance is missing repo metadata.")
    if not base_commit:
        raise RuntimeError("Benchmark instance is missing base_commit metadata.")

    clone_url = _resolve_repo_clone_url(repo_slug)
    workspace_root = _resolve_checkout_workspace_root(context)
    repo_dir = workspace_root / "repo"
    workspace_root.mkdir(parents=True, exist_ok=True)

    if (repo_dir / ".git").is_dir():
        remote_result = _run_command(["git", "-C", str(repo_dir), "remote", "get-url", "origin"])
        existing_origin = (remote_result.stdout or "").strip() if remote_result.returncode == 0 else ""
        if existing_origin and existing_origin != clone_url:
            shutil.rmtree(repo_dir, ignore_errors=True)

    if not (repo_dir / ".git").is_dir():
        clone_result = _run_command([
            "git",
            "clone",
            "--filter=blob:none",
            clone_url,
            str(repo_dir),
        ])
        if clone_result.returncode != 0:
            raise RuntimeError(
                "Failed to clone benchmark repository for Copilot backend. "
                f"{(clone_result.stderr or clone_result.stdout or '').strip()}"
            )

    fetch_result = _run_command(["git", "-C", str(repo_dir), "fetch", "--depth", "1", "origin", base_commit])
    if fetch_result.returncode != 0:
        fallback_fetch = _run_command(["git", "-C", str(repo_dir), "fetch", "origin", base_commit])
        if fallback_fetch.returncode != 0:
            raise RuntimeError(
                "Failed to fetch benchmark base commit for Copilot backend. "
                f"{(fallback_fetch.stderr or fallback_fetch.stdout or '').strip()}"
            )

    checkout_result = _run_command(["git", "-C", str(repo_dir), "checkout", "--force", base_commit])
    if checkout_result.returncode != 0:
        raise RuntimeError(
            "Failed to checkout benchmark base commit for Copilot backend. "
            f"{(checkout_result.stderr or checkout_result.stdout or '').strip()}"
        )

    clean_result = _run_command(["git", "-C", str(repo_dir), "clean", "-fdx"])
    if clean_result.returncode != 0:
        raise RuntimeError(
            "Failed to clean benchmark repository workspace for Copilot backend. "
            f"{(clean_result.stderr or clean_result.stdout or '').strip()}"
        )

    return repo_dir, clone_url, base_commit


def _build_prompt(context: RuntimeExecutionContext) -> str:
    prompt = str(context.instance.user_prompt or "").strip()
    if not prompt:
        prompt = f"Solve benchmark instance {context.instance.instance_id}."

    resume_prompt = str(context.instance.resume_prompt or "").strip()
    if resume_prompt:
        prompt = f"{prompt}\n\nResume context:\n{resume_prompt}"

    if _resolve_swe_sandbox_enabled(context) and _resolve_swe_sandbox_image(context):
        shared_preamble = (
            f"{prompt}\n\n"
            "The repository is checked out at $SWE_BENCH_SANDBOX_REPO_PATH (your working "
            "directory) - read and edit files there with your normal file tools.\n\n"
        )
        if _resolve_swe_exec_broker_enabled(context) and not _resolve_use_host_docker_socket(context):
            prompt = (
                f"{shared_preamble}"
                "Your own shell has no interpreter or dependencies for this task. Run "
                "anything that must execute in the task environment through `swe-exec`, "
                "e.g. swe-exec 'python --version'. It executes in a sandbox "
                "container that sees the same files at the same path, and forwards stdout, "
                "stderr and the exit status. There is no Docker access - `docker` commands "
                "will fail.\n\n"
                "Only $SWE_BENCH_SANDBOX_REPO_PATH is shared with that container - put "
                "scratch scripts there, not under /tmp.\n\n"
                "`swe-exec` puts the task environment's own interpreter first on PATH, so "
                "plain `python` is the right one. Which test runner and which packages are "
                "installed varies by repository - check rather than assume."
            )
        else:
            prompt = (
                f"{shared_preamble}"
                "A separate SWE sandbox container with the task environment's dependencies "
                "is available and sees the same files at the same path. Use docker exec "
                "against $SWE_BENCH_SANDBOX_CONTAINER_NAME (or "
                "$SWE_BENCH_SANDBOX_CONTAINER_ID) to run commands there."
            )

    if _resolve_network_isolation_enabled(context):
        prompt = (
            f"{prompt}\n\n"
            "You have no internet access in this environment. Do not attempt to install "
            "packages (pip/apt/npm/etc.), download files, or reach any external host - "
            "these will fail. Work only with what is already present in the repository "
            "and the task environment."
        )
    return prompt


def _write_logs(run_dir: Path, *, stdout: str, stderr: str, command: list[str], prompt: str) -> dict[str, str]:
    run_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = run_dir / "stdout.log"
    stderr_path = run_dir / "stderr.log"
    command_path = run_dir / "command.txt"
    prompt_path = run_dir / "prompt.txt"
    stdout_path.write_text(stdout, encoding="utf-8")
    stderr_path.write_text(stderr, encoding="utf-8")
    command_path.write_text(" ".join(shlex.quote(part) for part in command) + "\n", encoding="utf-8")
    prompt_path.write_text(prompt + "\n", encoding="utf-8")
    return {
        "stdout_log": str(stdout_path),
        "stderr_log": str(stderr_path),
        "command_path": str(command_path),
        "prompt_path": str(prompt_path),
    }


def _extract_stream_error(stdout: str) -> str | None:
    """Scan Copilot's JSONL event stream for the actual failure reason.

    docker compose's own stdout/stderr only ever contains provisioning noise
    ("Volume X Created", "Container Y Created"); the real reason a run failed
    (e.g. a 404 from the model provider) is emitted by the Copilot CLI as a
    `session.error` / `model.call_failure` event on stdout.
    """
    session_error: str | None = None
    call_failure: str | None = None
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue
        data = event.get("data") if isinstance(event.get("data"), dict) else {}
        if event.get("type") == "session.error":
            message = data.get("message")
            if isinstance(message, str) and message.strip():
                session_error = message.strip()
        elif event.get("type") == "model.call_failure":
            error_message = data.get("errorMessage")
            if isinstance(error_message, str) and error_message.strip():
                call_failure = error_message.strip()
    return session_error or call_failure


def _error_summary(stderr: str, stdout: str) -> str:
    stream_error = _extract_stream_error(stdout)
    if stream_error:
        return " ".join(stream_error.split())[:400]
    for candidate in (stderr, stdout):
        normalized = " ".join(candidate.split())
        if normalized:
            return normalized[:400]
    return "Copilot execution failed"


def _resolve_benchmark_trajectory_path(context: RuntimeExecutionContext) -> Path:
    safe_instance_id = str(context.instance.instance_id or context.instance.benchmark_id or "instance")
    safe_instance_id = safe_instance_id.replace("/", "_").replace("\\", "_")
    return (context.output_dir / f"copilot-trajectory-{safe_instance_id}.jsonl").resolve()


def run_copilot_instance(context: RuntimeExecutionContext) -> RuntimeExecutionResult:
    if context.workspace != "docker":
        raise RuntimeError(COPILOT_WORKSPACE_ERROR.format(workspace=context.workspace))

    compose_file = _resolve_compose_file(context)
    service = _resolve_compose_service(context)
    extra_args = _resolve_extra_args(context)
    output_format = _resolve_output_format(context)
    if not _contains_output_format_arg(extra_args):
        extra_args = [*extra_args, "--output-format", output_format]
    if not _contains_stream_arg(extra_args):
        extra_args = [*extra_args, "--stream", "off"]
    network_isolation = _resolve_network_isolation_enabled(context)
    keep_stack = _resolve_keep_compose_stack(context)
    cleanup_repo = _resolve_cleanup_repo_enabled(context)
    agent_timeout_seconds = _resolve_agent_timeout_seconds(context)
    proxy_service = _resolve_proxy_service(context)
    proxy_port = _resolve_proxy_port(context, proxy_service=proxy_service)
    compose_project = _resolve_compose_project_name(context)
    _maybe_cleanup_stale_copilot_docker_resources(context, keep_project=compose_project)
    swe_sandbox_enabled = _resolve_swe_sandbox_enabled(context)
    swe_sandbox_service = _resolve_swe_sandbox_service(context)
    swe_sandbox_repo_path = _resolve_swe_sandbox_repo_path(context)
    swe_sandbox_image = _resolve_swe_sandbox_image(context)
    dind_service = _resolve_dind_service(context)
    dind_image = _resolve_dind_image(context)
    dind_tls_enabled = _resolve_dind_tls_enabled(context)
    dind_port = _resolve_dind_port(context)
    sandbox_gateway_mode = _resolve_sandbox_gateway_mode(context)
    swe_exec_broker_enabled = _resolve_swe_exec_broker_enabled(context)
    swe_exec_broker_port = _resolve_swe_exec_broker_port(context)
    swe_exec_timeout_seconds = _resolve_swe_exec_timeout_seconds(context)
    use_host_docker_socket = _resolve_use_host_docker_socket(context)
    shared_proxy_mode = _resolve_shared_proxy_batch_enabled(context)
    compose_swe_sandbox_enabled = (
        swe_sandbox_enabled
        and swe_sandbox_image is not None
        and not shared_proxy_mode
    )
    # The broker only has a role in the dind path: the legacy host-socket bridge hands the
    # agent the host daemon outright, and without a nested sandbox there is nothing to exec.
    use_swe_exec_broker = (
        swe_exec_broker_enabled
        and compose_swe_sandbox_enabled
        and not use_host_docker_socket
    )
    compression_enabled = True
    use_compose_compression_service = _resolve_use_compose_compression_service(context)
    compression_image = _resolve_compression_service_image(context)
    compression_autobuild = _resolve_compression_service_autobuild(context)
    compression_script_path: Path | None = None
    compression_build_context: Path | None = None
    if compression_enabled:
        compression_script_path = _validate_compression_script_path(context)
        if compression_autobuild:
            compression_build_context = _resolve_compression_service_build_context(context)

    stack_services = _resolve_stack_services(
        context,
        proxy_service=proxy_service,
        compression_enabled=use_compose_compression_service,
        include_proxy=network_isolation,
    )
    if compose_swe_sandbox_enabled:
        stack_services.append(swe_sandbox_service if use_host_docker_socket else dind_service)
    prompt = _build_prompt(context)

    # New checkouts live outside output artifacts; remove empty legacy folder if present.
    legacy_workspace_root = context.instance_dir / "copilot-workspace"
    if legacy_workspace_root.is_dir() and not any(legacy_workspace_root.iterdir()):
        legacy_workspace_root.rmdir()

    repo_root, clone_url, base_commit = _prepare_instance_repo_checkout(context)
    tmp_run_dir = _resolve_tmp_run_root(context)
    tmp_run_dir.mkdir(parents=True, exist_ok=True)
    emit_progress(
        f"[{context.instance.instance_id}] Copilot checkout path: {repo_root}",
        component="copilot",
    )
    emit_progress(
        f"[{context.instance.instance_id}] Copilot tmp run path: {tmp_run_dir}",
        component="copilot",
    )
    run_dir = context.instance_dir / "copilot"

    compose_prefix = [
        "docker",
        "compose",
        "-f",
        str(compose_file),
        "--project-name",
        compose_project,
    ]

    run_container_args: list[str] = []

    command = [
        *compose_prefix,
        "run",
        "--rm",
        *run_container_args,
        "-e",
        "COPILOT_PROVIDER_BASE_URL",
        "-e",
        "COPILOT_MODEL",
        "-e",
        "COPILOT_PROVIDER_API_KEY",
        service,
        "-p",
        prompt,
        *extra_args,
    ]

    env = os.environ.copy()
    env["COPILOT_COMPRESSION_SERVICE_IMAGE"] = compression_image
    env["COPILOT_PROXY_IMAGE"] = compression_image
    # Consumed by the driver_opts of copilot-sandbox / copilot-dind. Set unconditionally so
    # the compose default is never the thing deciding whether the host is reachable.
    env["COPILOT_SANDBOX_GATEWAY_MODE"] = sandbox_gateway_mode
    if compression_script_path is not None:
        env["COPILOT_COMPRESSION_SCRIPT_PATH"] = str(compression_script_path)
        env["MINER_MODULE_PATH"] = COPILOT_DEFAULT_COMPRESSION_SCRIPT_CONTAINER_PATH

    compression_build_stdout = ""
    compression_build_stderr = ""
    if compression_enabled and compression_autobuild:
        emit_progress(
            f"[{context.instance.instance_id}] Building compression-service image {compression_image}",
            component="copilot",
        )
        compression_build_stdout, compression_build_stderr = _build_compression_service_image(
            image_name=compression_image,
            build_context=compression_build_context or _resolve_compression_service_build_context(context),
        )

    should_start_stack = network_isolation or compose_swe_sandbox_enabled
    if should_start_stack:
        # Ensure compose also manages profile-gated sidecars during ps/down.
        existing_profiles = str(env.get("COMPOSE_PROFILES", "")).strip()
        if existing_profiles:
            profiles = {item.strip() for item in existing_profiles.split(",") if item.strip()}
            profiles.add("copilot-sidecars")
            env["COMPOSE_PROFILES"] = ",".join(sorted(profiles))
        else:
            env["COMPOSE_PROFILES"] = "copilot-sidecars"

    agent_image = _resolve_compose_service_image(
        compose_file=compose_file,
        compose_project=compose_project,
        service=service,
        env=env,
    )
    workspace_owner = _resolve_image_user(agent_image)
    if workspace_owner is not None:
        emit_progress(
            f"[{context.instance.instance_id}] seeding workspace owned by "
            f"{workspace_owner[0]}:{workspace_owner[1]} (agent user in {agent_image})",
            component="copilot",
        )
    # Seeding from the task image itself (rather than only ever from a bare external clone)
    # preserves whatever that image's own dependency-install step already built at the repo
    # path - compiled extensions, editable-install metadata, generated version files - none
    # of which a fresh `git clone` can ever have. Only applies on the dind path: the
    # host-docker-socket fallback talks to the sandbox container directly rather than
    # through this shared volume, and with no sandbox image at all there is nothing to seed
    # from but the clone.
    workspace_volume = _seed_workspace_volume(
        compose_project=compose_project,
        repo_root=repo_root,
        owner=workspace_owner,
        sandbox_image=(
            swe_sandbox_image
            if compose_swe_sandbox_enabled and not use_host_docker_socket
            else None
        ),
        sandbox_repo_path=swe_sandbox_repo_path,
        base_commit=base_commit,
    )
    env["COPILOT_WORKSPACE_VOLUME_NAME"] = workspace_volume
    if compose_swe_sandbox_enabled:
        if use_host_docker_socket:
            env["COPILOT_SWE_SANDBOX_IMAGE"] = swe_sandbox_image
        else:
            dind_image = _resolve_effective_dind_image(
                context,
                swe_sandbox_image=swe_sandbox_image,
                default_dind_image=dind_image,
            )
            env["COPILOT_DIND_IMAGE"] = dind_image
            env["COPILOT_DIND_SERVICE"] = dind_service
            env["COPILOT_DIND_PORT"] = str(dind_port)
            env["COPILOT_DIND_HOSTNAME"] = COPILOT_DIND_TLS_HOSTNAME
            # Always set explicitly, in both directions: this reaches the daemon through the
            # dind service's own `environment:` entry, and leaving it to whatever the host
            # environment happens to hold would let an unrelated DOCKER_TLS_CERTDIR decide
            # whether the daemon is authenticated.
            env["COPILOT_DIND_TLS_CERTDIR"] = COPILOT_DIND_TLS_CERTDIR if dind_tls_enabled else ""
            if not dind_tls_enabled:
                # Reverts the daemon to a plaintext, unauthenticated listener. The iptables
                # layer becomes the only remaining control, which is why
                # _apply_proxy_compression_isolation is made mandatory below.
                emit_progress(
                    "[copilot] WARNING: dind mutual TLS is disabled "
                    "(SOMA_COPILOT_DIND_TLS=false); its API is unauthenticated",
                    component="copilot",
                )
    swe_exec_broker_token = ""
    if use_swe_exec_broker:
        broker_script = _swe_exec_broker_script_path()
        swe_exec_client = _swe_exec_client_path()
        for required_path, label in (
            (broker_script, "exec broker"),
            (swe_exec_client, "swe-exec client"),
        ):
            if not required_path.is_file():
                raise RuntimeError(
                    f"Copilot SWE {label} script is missing at {required_path}. "
                    "Set SOMA_COPILOT_SWE_EXEC_BROKER=false to fall back to direct "
                    "DOCKER_HOST access (which grants the agent host-root-equivalent power)."
                )
        swe_exec_broker_token = secrets.token_urlsafe(32)
        env["COPILOT_SWE_EXEC_BROKER_IMAGE"] = compression_image
        env["COPILOT_SWE_EXEC_BROKER_SCRIPT_PATH"] = str(broker_script)
        env["COPILOT_SWE_EXEC_BROKER_TOKEN"] = swe_exec_broker_token
        env["COPILOT_SWE_EXEC_BROKER_PORT"] = str(swe_exec_broker_port)
        env["COPILOT_SWE_SANDBOX_REPO_PATH"] = swe_sandbox_repo_path
        env["COPILOT_SWE_EXEC_TIMEOUT_SECONDS"] = str(swe_exec_timeout_seconds)
    if network_isolation:
        base_url_raw = str(context.llm_config.get("base_url", "")).strip()
        if not base_url_raw:
            raise RuntimeError(
                "Copilot network isolation requires llm.base_url to be set. "
                "Set LLM_BASE_URL (or OPENAI_BASE_URL / OPENROUTER_BASE_URL)."
            )
        env[COPILOT_PROXY_UPSTREAM_ENV] = _normalize_proxy_upstream_base_url(base_url_raw)
        env["PROXY_COMPRESSION_ENABLED"] = "true" if compression_enabled else "false"
        env["PROXY_COMPRESSION_BASE_URL"] = (
            f"http://{COPILOT_COMPRESSION_SERVICE_NAME}:{COPILOT_COMPRESSION_SERVICE_PORT}/"
        )
        env["COPILOT_PROVIDER_BASE_URL"] = f"http://{proxy_service}:{proxy_port}/"
    elif isinstance(context.llm_config.get("base_url"), str):
        env["COPILOT_PROVIDER_BASE_URL"] = str(context.llm_config["base_url"])
    if isinstance(context.llm_config.get("model"), str):
        env["COPILOT_MODEL"] = str(context.llm_config["model"])
    run_id_header_value = _resolve_copilot_run_id_header_value(context)
    if run_id_header_value:
        # For gateway mode, provider-facing auth is replaced with run_id and resolved server-side.
        # Keep this on Copilot run container only; avoid mutating shared proxy container env per run.
        env["COPILOT_PROVIDER_API_KEY"] = run_id_header_value
    elif isinstance(context.llm_config.get("api_key"), str):
        env["COPILOT_PROVIDER_API_KEY"] = str(context.llm_config["api_key"])
        env["PROXY_PROVIDER_API_KEY"] = str(context.llm_config["api_key"])

    stack_up_stdout = ""
    stack_up_stderr = ""
    stack_down_stdout = ""
    stack_down_stderr = ""
    swe_sandbox_container_id = ""
    swe_sandbox_container_name = ""
    sidecar_log_paths: dict[str, str] = {}
    message_request_log_paths: dict[str, str] = {}
    proxy_token_usage: dict[str, int] = {}
    patch_capture: dict[str, Any] = {
        "status": "not-captured",
        "repo_dir": "",
        "patch_path": "",
        "has_changes": False,
        "line_count": 0,
        "size_bytes": 0,
    }

    isolation_rules: list[tuple[str, str, str]] = []
    stack_up_attempted = False
    try:
        if should_start_stack:
            stack_up_attempted = True
            stack_up = _run_command(
                [*compose_prefix, "up", "-d", *stack_services],
                cwd=repo_root,
                env=env,
            )
            stack_up_stdout = stack_up.stdout or ""
            stack_up_stderr = stack_up.stderr or ""
            if stack_up.returncode != 0:
                raise RuntimeError(
                    "Failed to start Copilot proxy sidecar. "
                    f"{(stack_up.stderr or stack_up.stdout or '').strip()}"
                )
            if sandbox_gateway_mode == COPILOT_SANDBOX_GATEWAY_MODE_ISOLATED:
                _warn_if_networks_reach_host(
                    compose_project=compose_project,
                    network_suffixes=("copilot-sandbox", "copilot-dind"),
                )
            if network_isolation and use_compose_compression_service:
                isolation_rules = _apply_proxy_compression_isolation(
                    compose_project=compose_project,
                    proxy_service=proxy_service,
                    dind_service=(
                        dind_service
                        if compose_swe_sandbox_enabled and not use_host_docker_socket
                        else None
                    ),
                    # With TLS off, iptables is the only thing standing between the miner's
                    # code and the privileged daemon, so a failure to insert it must abort.
                    require_dind_isolation=(
                        not dind_tls_enabled
                        and compose_swe_sandbox_enabled
                        and not use_host_docker_socket
                    ),
                )

        if compose_swe_sandbox_enabled and use_host_docker_socket:
            sandbox_ps = _run_command(
                [*compose_prefix, "ps", "-q", swe_sandbox_service],
                cwd=repo_root,
                env=env,
            )
            swe_sandbox_container_id = (sandbox_ps.stdout or "").strip().splitlines()[0] if (sandbox_ps.stdout or "").strip() else ""
            if not swe_sandbox_container_id:
                raise RuntimeError(
                    "Copilot SWE sandbox container is not running after compose up. "
                    f"{(sandbox_ps.stderr or sandbox_ps.stdout or '').strip()}"
                )
            inspect_name = _run_command(["docker", "inspect", "-f", "{{.Name}}", swe_sandbox_container_id])
            if inspect_name.returncode == 0:
                swe_sandbox_container_name = (inspect_name.stdout or "").strip().lstrip("/")
            if not swe_sandbox_container_name:
                swe_sandbox_container_name = swe_sandbox_container_id

            # Legacy bridge: the agent gets the HOST docker.sock directly, so it can see
            # and control every container on the host, not just this SWE sandbox.
            env["SWE_BENCH_SANDBOX_CONTAINER_ID"] = swe_sandbox_container_id
            env["SWE_BENCH_SANDBOX_CONTAINER_NAME"] = swe_sandbox_container_name
            env["SWE_BENCH_SANDBOX_REPO_PATH"] = swe_sandbox_repo_path
            run_container_args.extend(_docker_socket_mount_args())
            run_container_args.extend(["--user", "root"])
            run_container_args.extend([
                "-e",
                "SWE_BENCH_SANDBOX_CONTAINER_ID",
                "-e",
                "SWE_BENCH_SANDBOX_CONTAINER_NAME",
                "-e",
                "SWE_BENCH_SANDBOX_REPO_PATH",
            ])
        elif compose_swe_sandbox_enabled:
            dind_ps = _run_command(
                [*compose_prefix, "ps", "-q", dind_service],
                cwd=repo_root,
                env=env,
            )
            dind_container_id = (dind_ps.stdout or "").strip().splitlines()[0] if (dind_ps.stdout or "").strip() else ""
            if not dind_container_id:
                raise RuntimeError(
                    "Copilot isolated Docker daemon (dind) is not running after compose up. "
                    f"{(dind_ps.stderr or dind_ps.stdout or '').strip()}"
                )
            dind_utils.wait_for_dind_ready(dind_container_id=dind_container_id)
            dind_utils.ensure_image_in_isolated_daemon(
                dind_container_id=dind_container_id,
                image=swe_sandbox_image,
                role="Copilot SWE sandbox",
            )
            swe_sandbox_container_name = swe_sandbox_service
            swe_sandbox_container_id = _start_nested_swe_sandbox(
                dind_container_id=dind_container_id,
                image=swe_sandbox_image,
                container_name=swe_sandbox_container_name,
                workspace_container_path=COPILOT_DIND_WORKSPACE_MOUNT,
                repo_path=swe_sandbox_repo_path,
            )

            env["SWE_BENCH_SANDBOX_CONTAINER_ID"] = swe_sandbox_container_id
            env["SWE_BENCH_SANDBOX_CONTAINER_NAME"] = swe_sandbox_container_name
            env["SWE_BENCH_SANDBOX_REPO_PATH"] = swe_sandbox_repo_path
            run_container_args.extend([
                "-e",
                "SWE_BENCH_SANDBOX_CONTAINER_ID",
                "-e",
                "SWE_BENCH_SANDBOX_CONTAINER_NAME",
                "-e",
                "SWE_BENCH_SANDBOX_REPO_PATH",
            ])

            if use_swe_exec_broker:
                env["COPILOT_SWE_EXEC_SANDBOX_CONTAINER"] = swe_sandbox_container_id
                env_bin_dir = _discover_swe_sandbox_env_bin_dir(
                    dind_container_id=dind_container_id,
                    sandbox_container_id=swe_sandbox_container_id,
                )
                if env_bin_dir:
                    emit_progress(
                        f"[{context.instance.instance_id}] swe-exec will default PATH to "
                        f"{env_bin_dir} (discovered task env interpreter)",
                        component="copilot",
                    )
                    env["COPILOT_SWE_EXEC_SANDBOX_ENV_BIN_DIR"] = env_bin_dir
                else:
                    emit_progress(
                        f"[{context.instance.instance_id}] could not discover a task-specific "
                        "conda env for swe-exec; PATH will default to whatever the sandbox "
                        "image's own shell resolves",
                        component="copilot",
                    )
                _start_swe_exec_broker(
                    compose_prefix=compose_prefix,
                    cwd=repo_root,
                    env=env,
                    broker_port=swe_exec_broker_port,
                )
                if network_isolation and use_compose_compression_service:
                    isolation_rules.extend(
                        _apply_proxy_compression_isolation(
                            compose_project=compose_project,
                            exec_broker_service=COPILOT_SWE_EXEC_BROKER_SERVICE,
                            purge_stale_rules=False,
                        )
                    )
                broker_url = (
                    f"http://{COPILOT_SWE_EXEC_BROKER_SERVICE}:{swe_exec_broker_port}"
                )
                env["SWE_EXEC_BROKER_URL"] = broker_url
                env["SWE_EXEC_BROKER_TOKEN"] = swe_exec_broker_token
                env["SWE_EXEC_TIMEOUT_SECONDS"] = str(swe_exec_timeout_seconds)
                run_container_args.extend([
                    "-v",
                    f"{_swe_exec_client_path()}:{COPILOT_SWE_EXEC_CLIENT_CONTAINER_PATH}:ro",
                    "-e",
                    "SWE_EXEC_BROKER_URL",
                    "-e",
                    "SWE_EXEC_BROKER_TOKEN",
                    "-e",
                    "SWE_EXEC_TIMEOUT_SECONDS",
                ])
                emit_progress(
                    f"[{context.instance.instance_id}] SWE sandbox reachable via exec broker "
                    f"at {broker_url} (agent holds no Docker API access)",
                    component="copilot",
                )
            else:
                sandbox_network = f"{compose_project}_copilot-sandbox"
                connect_result = _run_command([
                    "docker", "network", "connect",
                    "--alias", dind_service,
                    "--alias", COPILOT_DIND_TLS_HOSTNAME,
                    sandbox_network, dind_container_id,
                ])
                if connect_result.returncode != 0:
                    connect_details = (
                        connect_result.stderr or connect_result.stdout or ""
                    ).strip()
                    # Compose only creates copilot-sandbox once a service that uses it comes
                    # up, so with a dind-only stack the network may not exist yet. Failing
                    # loudly beats handing the agent a DOCKER_HOST it cannot route to.
                    if "already exists" not in connect_details:
                        raise RuntimeError(
                            "Could not attach the isolated Docker daemon to the agent "
                            "network, which the SOMA_COPILOT_SWE_EXEC_BROKER=false fallback "
                            f"requires: {connect_details}"
                        )
                emit_progress(
                    "[copilot] WARNING: exec broker disabled",
                    component="copilot",
                )
                dind_host = _isolated_docker_host(
                    dind_service=(
                        COPILOT_DIND_TLS_HOSTNAME if dind_tls_enabled else dind_service
                    ),
                    port=dind_port,
                )
                run_container_args.extend(["-e", f"DOCKER_HOST={dind_host}"])
                if dind_tls_enabled:
                    run_container_args.extend([
                        "-v",
                        f"{compose_project}_copilot-dind-certs:{COPILOT_DIND_CLIENT_CERT_DIR}:ro",
                        "-e",
                        "DOCKER_TLS_VERIFY=1",
                        "-e",
                        f"DOCKER_CERT_PATH={COPILOT_DIND_CLIENT_CERT_DIR}",
                    ])

        if compose_swe_sandbox_enabled:
            command = [
                *compose_prefix,
                "run",
                "--rm",
                *run_container_args,
                "-e",
                "COPILOT_PROVIDER_BASE_URL",
                "-e",
                "COPILOT_MODEL",
                "-e",
                "COPILOT_PROVIDER_API_KEY",
                service,
                "-p",
                prompt,
                *extra_args,
            ]

        benchmark_trajectory_path: Path | None = None
        live_trajectory_path: Path | None = None
        live_stream_log_path: Path | None = None
        if output_format == "json":
            benchmark_trajectory_path = _resolve_benchmark_trajectory_path(context)
            live_trajectory_path = benchmark_trajectory_path
        live_stream_log_path = tmp_run_dir / "copilot-stream.log"

        result, agent_timed_out = _run_copilot_command_streaming(
            command=command,
            cwd=repo_root,
            env=env,
            trajectory_path=live_trajectory_path,
            stream_log_path=live_stream_log_path,
            timeout_seconds=agent_timeout_seconds,
        )
        if agent_timed_out:
            emit_progress(
                f"[{context.instance.instance_id}] Copilot execution timed out after "
                f"{agent_timeout_seconds:.0f}s; collecting token usage, skipping patch capture",
                component="copilot",
            )

        trajectory_path = live_trajectory_path
        if trajectory_path is None:
            if benchmark_trajectory_path is None:
                benchmark_trajectory_path = _resolve_benchmark_trajectory_path(context)
            trajectory_path = _write_copilot_trajectory(
                stdout=result.stdout or "",
                output_format=output_format,
                destination_path=benchmark_trajectory_path,
            )
        agent_steps: int | None = None
        if trajectory_path is not None:
            emit_progress(
                f"[{context.instance.instance_id}] Copilot trajectory saved to {trajectory_path}",
                component="copilot",
            )
            agent_steps = _count_trajectory_steps(trajectory_path)
        if not agent_timed_out:
            emit_progress(
                f"[{context.instance.instance_id}] capturing repository patch from workspace volume",
                component="copilot",
            )
            patch_capture = _capture_workspace_patch(
                volume_name=workspace_volume,
                tmp_run_dir=tmp_run_dir,
                base_commit=base_commit,
            )
            emit_progress(
                f"[{context.instance.instance_id}] patch capture status: {patch_capture.get('status')} changes={patch_capture.get('has_changes')}",
                component="copilot",
            )
    finally:
        if stack_up_attempted:
            sidecar_service_names = {proxy_service, COPILOT_DEFAULT_PROXY_SERVICE, "compression-service"}
            if compose_swe_sandbox_enabled:
                sidecar_service_names.add(swe_sandbox_service if use_host_docker_socket else dind_service)
            if use_swe_exec_broker:
                sidecar_service_names.add(COPILOT_SWE_EXEC_BROKER_SERVICE)
            sidecar_services = [service_name for service_name in sidecar_service_names]
            sidecar_log_paths = _collect_compose_service_logs(
                compose_prefix=compose_prefix,
                cwd=repo_root,
                env=env,
                services=sidecar_services,
                destination_dir=tmp_run_dir,
            )
            message_request_log_paths = _extract_message_request_logs(
                sidecar_log_paths=sidecar_log_paths,
                destination_dir=tmp_run_dir,
            )
            proxy_token_usage = _extract_proxy_token_usage(sidecar_log_paths=sidecar_log_paths)
        if isolation_rules:
            _remove_proxy_compression_isolation(isolation_rules)
        if stack_up_attempted and not keep_stack:
            stack_down = _run_command(
                [*compose_prefix, "down", "--remove-orphans", "--volumes"],
                cwd=repo_root,
                env=env,
            )
            stack_down_stdout = stack_down.stdout or ""
            stack_down_stderr = stack_down.stderr or ""
            for _net_suffix in ("copilot-sandbox", "copilot-egress", "copilot-dind"):
                _run_command(["docker", "network", "rm", f"{compose_project}_{_net_suffix}"])
        if not keep_stack:
            _remove_workspace_volume(volume_name=workspace_volume)

    if cleanup_repo:
        if repo_root.is_dir():
            shutil.rmtree(repo_root, ignore_errors=True)

    log_paths = _write_logs(
        run_dir,
        stdout=result.stdout or "",
        stderr=result.stderr or "",
        command=command,
        prompt=prompt,
    )
    metadata = {
        "compose_file": str(compose_file),
        "service": service,
        "cwd": str(repo_root),
        "repo": context.instance.repo,
        "repo_clone_url": clone_url,
        "repo_commit": base_commit,
        "repo_workspace": str(repo_root),
        "tmp_run_dir": str(tmp_run_dir),
        "workspace_volume": workspace_volume,
        "command": "docker compose --project-name <name> run copilot -p ...",
        "args": extra_args,
        "exit_code": result.returncode,
        "network_isolation": network_isolation,
        "keep_stack": keep_stack,
        "cleanup_repo": cleanup_repo,
        "compose_project": compose_project,
        "proxy_service": proxy_service,
        "proxy_port": proxy_port,
        "stack_services": stack_services,
        "swe_sandbox_enabled": swe_sandbox_enabled,
        "swe_sandbox_service": swe_sandbox_service,
        "swe_sandbox_image": swe_sandbox_image,
        "swe_sandbox_container_id": swe_sandbox_container_id,
        "swe_sandbox_container_name": swe_sandbox_container_name,
        "swe_sandbox_repo_path": swe_sandbox_repo_path,
        "dind_service": dind_service,
        "dind_image": dind_image,
        "dind_tls": dind_tls_enabled,
        "dind_port": dind_port,
        "swe_exec_broker": use_swe_exec_broker,
        "docker_bridge_mode": (
            "host-socket"
            if use_host_docker_socket
            else ("dind-broker" if use_swe_exec_broker else "dind-direct")
        ),
        "stack_up_stdout": stack_up_stdout,
        "stack_up_stderr": stack_up_stderr,
        "stack_down_stdout": stack_down_stdout,
        "stack_down_stderr": stack_down_stderr,
        "run_id": _resolve_execution_run_id(context),
        "compression_enabled": compression_enabled,
        "compression_script_path": str(compression_script_path) if compression_script_path is not None else "",
        "compression_service_image": compression_image,
        "compression_service_autobuild": compression_autobuild,
        "compression_service_build_context": str(compression_build_context) if compression_build_context is not None else "",
        "compression_service_build_stdout": compression_build_stdout,
        "compression_service_build_stderr": compression_build_stderr,
        "run_root": str(_resolve_run_root(context)),
        "runtime_options": _resolve_runtime_options(context),
        "copilot_output_format": output_format,
        "trajectory_path": str(trajectory_path) if trajectory_path is not None else "",
        "stream_log_path": str((tmp_run_dir / "copilot-stream.log").resolve()),
        "agent_steps": agent_steps,
        "agent_timeout_seconds": agent_timeout_seconds,
        "agent_timed_out": agent_timed_out,
        "patch_capture": patch_capture,
        "token_usage": proxy_token_usage,
        **sidecar_log_paths,
        **message_request_log_paths,
        **log_paths,
    }

    if agent_timed_out:
        return RuntimeExecutionResult(
            benchmark_id=context.instance.benchmark_id,
            instance_id=context.instance.instance_id,
            backend_name=context.backend_name,
            status="timeout",
            error=f"Copilot execution timed out after {agent_timeout_seconds:.0f} seconds",
            metadata=metadata,
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
        error=_error_summary(result.stderr or "", result.stdout or ""),
        metadata=metadata,
    )


def run_copilot_batch(
    contexts: list[RuntimeExecutionContext],
    concurrency: int,
) -> list[RuntimeExecutionResult]:
    if not contexts:
        return []

    shared_proxy_enabled = _resolve_shared_proxy_batch_enabled(contexts[0])
    shared_proxy_teardown_enabled = _resolve_shared_proxy_teardown_enabled(contexts[0])
    if shared_proxy_enabled:
        run_id = _resolve_execution_run_id(contexts[0])
        digest = hashlib.sha256(
            f"{contexts[0].output_dir.resolve()}::{run_id}".encode("utf-8")
        ).hexdigest()[:10]
        shared_compose_project = f"soma-copilot-{digest}"
        contexts = [
            _with_runtime_option_overrides(
                context,
                {
                    "copilot_compose_project": shared_compose_project,
                    "copilot_keep_stack": True,
                },
            )
            for context in contexts
        ]
        if concurrency > 1:
            emit_progress(
                "[copilot] shared proxy mode requires serial execution; forcing concurrency=1 "
                "to avoid workspace volume races.",
                component="copilot",
            )
            concurrency = 1

    worker_count = max(1, min(concurrency, len(contexts)))
    try:
        if worker_count == 1:
            return [run_copilot_instance(context) for context in contexts]

        results: list[RuntimeExecutionResult | None] = [None] * len(contexts)
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            future_to_index = {
                executor.submit(run_copilot_instance, context): index
                for index, context in enumerate(contexts)
            }
            for future in as_completed(future_to_index):
                results[future_to_index[future]] = future.result()

        return [result for result in results if result is not None]
    finally:
        if shared_proxy_enabled and shared_proxy_teardown_enabled and contexts:
            compose_project = str(
                _resolve_runtime_option(contexts[0], "copilot_compose_project")
                or _resolve_compose_project_name(contexts[0])
            )
            _teardown_shared_compose_stack(contexts[0], compose_project=compose_project)


COPILOT_BACKEND = RuntimeBackend(
    name="copilot",
    default_image=COPILOT_DEFAULT_IMAGE,
    default_build_target="container",
    execution_hint=(
        "Use --execute to run benchmark instances through GitHub Copilot CLI in Docker Compose. "
        "By default this backend runs docker compose -f "
        "src/soma_bench/benchmark/backends/copilot/copilot-cli-container/docker-compose.yml "
        "run --rm copilot -p <prompt> --allow-all --no-ask-user."
    ),
    execute=run_copilot_instance,
    execute_batch=run_copilot_batch,
)
