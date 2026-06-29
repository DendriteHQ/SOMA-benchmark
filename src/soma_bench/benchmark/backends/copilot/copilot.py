from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import hashlib
import tempfile
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from ..base import RuntimeBackend, RuntimeExecutionContext, RuntimeExecutionResult
from ...swerebench_eval import capture_repo_patch
from ...progress import emit_progress

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
COPILOT_DEFAULT_SHARED_PROXY_BATCH = True
COPILOT_DEFAULT_SHARED_PROXY_TEARDOWN = True


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


def _host_docker_binary() -> str | None:
    docker_binary = shutil.which("docker")
    if docker_binary and os.path.isabs(docker_binary):
        return docker_binary
    return None


def _docker_socket_mount_args() -> list[str]:
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


def _seed_workspace_volume(*, compose_project: str, repo_root: Path) -> str:
    volume_name = _workspace_volume_name(compose_project=compose_project)
    _run_command(["docker", "volume", "rm", "-f", volume_name])
    create_result = _run_command(["docker", "volume", "create", volume_name])
    if create_result.returncode != 0:
        raise RuntimeError(
            "Failed to create Copilot workspace Docker volume. "
            f"{(create_result.stderr or create_result.stdout or '').strip()}"
        )

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
            "rm -rf /workspace/* /workspace/.[!.]* /workspace/..?* 2>/dev/null || true; cp -a /src/. /workspace/",
        ]
    )
    if copy_result.returncode != 0:
        _remove_workspace_volume(volume_name=volume_name)
        raise RuntimeError(
            "Failed to seed Copilot workspace volume from repository checkout. "
            f"{(copy_result.stderr or copy_result.stdout or '').strip()}"
        )
    return volume_name


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


def _capture_workspace_patch(*, volume_name: str, tmp_run_dir: Path) -> dict[str, Any]:
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

    return capture_repo_patch(repo_dir=snapshot_dir, output_dir=patch_eval_dir)


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
    proxy_service: str,
) -> list[tuple[str, str, str]]:
    """Block TCP from compression-service to proxy (one-way: proxy may still call compression-service).

    Returns inserted rules as (src_ip, dst_ip, comment) tuples for later removal.
    """
    if not _ensure_bridge_netfilter():
        return []

    _iptables_purge_stale_soma_rules()

    comp_name = f"{compose_project}-{COPILOT_COMPRESSION_SERVICE_NAME}-1"
    proxy_name = f"{compose_project}-{proxy_service}-1"
    comp_ips = _get_container_network_ips(comp_name)
    proxy_ips = _get_container_network_ips(proxy_name)

    sandbox_key = next((k for k in comp_ips if "sandbox" in k), None)
    comp_ip = comp_ips.get(sandbox_key, "") if sandbox_key else next(iter(comp_ips.values()), "")
    if sandbox_key and sandbox_key in proxy_ips:
        proxy_ip = proxy_ips[sandbox_key]
    else:
        proxy_ip = next(iter(proxy_ips.values()), "")

    if not comp_ip or not proxy_ip:
        emit_progress(
            f"[copilot] iptables isolation skipped: could not resolve container IPs "
            f"(comp={comp_ip!r} proxy={proxy_ip!r})",
            component="copilot",
        )
        return []

    comment = f"soma-iso-{compose_project[-16:]}"
    if _iptables_insert_drop(src_ip=comp_ip, dst_ip=proxy_ip, comment=comment):
        emit_progress(
            f"[copilot] iptables DROP: compression ({comp_ip}) -> proxy ({proxy_ip}) tcp",
            component="copilot",
        )
        return [(comp_ip, proxy_ip, comment)]
    emit_progress(
        "[copilot] iptables DROP rule insert failed (insufficient privileges?)",
        component="copilot",
    )
    return []


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

    Uses the last assistant.turn_end event's turnId (0-indexed) + 1.
    Falls back to counting assistant.turn_start events if no turn_end is found.
    """
    try:
        if not trajectory_path.is_file():
            return None
        last_turn_end: dict | None = None
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
                elif entry_type == "assistant.turn_end":
                    last_turn_end = entry
        if last_turn_end is not None:
            turn_id_raw = (last_turn_end.get("data") or {}).get("turnId")
            if turn_id_raw is not None:
                try:
                    return int(turn_id_raw) + 1
                except (TypeError, ValueError):
                    pass
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
) -> subprocess.CompletedProcess[str]:
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
        )
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
        prompt = (
            f"{prompt}\n\n"
            "SWE sandbox is available for command execution. "
            "Run task-environment commands using docker exec against "
            "$SWE_BENCH_SANDBOX_CONTAINER_NAME (or $SWE_BENCH_SANDBOX_CONTAINER_ID). "
            "The repository path inside sandbox is $SWE_BENCH_SANDBOX_REPO_PATH."
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


def _error_summary(stderr: str, stdout: str) -> str:
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
    proxy_service = _resolve_proxy_service(context)
    proxy_port = _resolve_proxy_port(context, proxy_service=proxy_service)
    compose_project = _resolve_compose_project_name(context)
    swe_sandbox_enabled = _resolve_swe_sandbox_enabled(context)
    swe_sandbox_service = _resolve_swe_sandbox_service(context)
    swe_sandbox_repo_path = _resolve_swe_sandbox_repo_path(context)
    swe_sandbox_image = _resolve_swe_sandbox_image(context)
    shared_proxy_mode = _resolve_shared_proxy_batch_enabled(context)
    compose_swe_sandbox_enabled = (
        swe_sandbox_enabled
        and swe_sandbox_image is not None
        and not shared_proxy_mode
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
        stack_services.append(swe_sandbox_service)
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

    workspace_volume = _seed_workspace_volume(compose_project=compose_project, repo_root=repo_root)
    env["COPILOT_WORKSPACE_VOLUME_NAME"] = workspace_volume
    if compose_swe_sandbox_enabled:
        env["COPILOT_SWE_SANDBOX_IMAGE"] = swe_sandbox_image
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
            if network_isolation and use_compose_compression_service:
                isolation_rules = _apply_proxy_compression_isolation(
                    compose_project=compose_project,
                    proxy_service=proxy_service,
                )

        if compose_swe_sandbox_enabled:
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

        result = _run_copilot_command_streaming(
            command=command,
            cwd=repo_root,
            env=env,
            trajectory_path=live_trajectory_path,
            stream_log_path=live_stream_log_path,
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
        emit_progress(
            f"[{context.instance.instance_id}] capturing repository patch from workspace volume",
            component="copilot",
        )
        patch_capture = _capture_workspace_patch(
            volume_name=workspace_volume,
            tmp_run_dir=tmp_run_dir,
        )
        emit_progress(
            f"[{context.instance.instance_id}] patch capture status: {patch_capture.get('status')} changes={patch_capture.get('has_changes')}",
            component="copilot",
        )
    finally:
        if stack_up_attempted:
            sidecar_services = [service_name for service_name in {proxy_service, COPILOT_DEFAULT_PROXY_SERVICE, "compression-service"}]
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
            for _net_suffix in ("copilot-sandbox", "copilot-egress"):
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
        "patch_capture": patch_capture,
        "token_usage": proxy_token_usage,
        **sidecar_log_paths,
        **message_request_log_paths,
        **log_paths,
    }

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
