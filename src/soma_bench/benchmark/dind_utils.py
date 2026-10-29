"""Shared Docker-in-Docker (dind) helpers.

Used by both the Copilot runtime backend (backends/copilot/copilot.py) and the offline
swebench_dind_prebake build tool, so a pre-baked image and the runtime's save/load
fallback are guaranteed to produce byte-identical results - both paths load a target
image into an isolated dind daemon using this exact code.
"""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

from .registry_auth import docker_env_for_image

# A pre-baked dind image unpacks its whole docker storage (several GB) before dockerd
# starts, so readiness is bounded by disk throughput, not by dockerd's own start time.
DIND_READY_TIMEOUT_ENV = "SOMA_DIND_READY_TIMEOUT_SECONDS"
DEFAULT_DIND_READY_TIMEOUT_SECONDS = 600.0
DIND_START_SLEEP_SECONDS = 0.5
DIND_FAILURE_LOG_TAIL_LINES = 20


def run_command(
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


def docker_image_exists(image_ref: str) -> bool:
    return run_command(["docker", "image", "inspect", image_ref]).returncode == 0


def ensure_docker_image_available(image_ref: str, *, role: str) -> None:
    if docker_image_exists(image_ref):
        return

    pull_result = run_command(["docker", "pull", image_ref], env=docker_env_for_image(image_ref))
    if pull_result.returncode == 0 and docker_image_exists(image_ref):
        return

    message = (pull_result.stderr or pull_result.stdout or "").strip()
    raise RuntimeError(
        f"{role} Docker image is not available locally and automatic pull failed for {image_ref}: {message}"
    )


def dind_ready_timeout_seconds() -> float:
    raw = (os.getenv(DIND_READY_TIMEOUT_ENV) or "").strip()
    if not raw:
        return DEFAULT_DIND_READY_TIMEOUT_SECONDS
    try:
        value = float(raw)
    except ValueError:
        return DEFAULT_DIND_READY_TIMEOUT_SECONDS
    return value if value > 0 else DEFAULT_DIND_READY_TIMEOUT_SECONDS


def _dind_container_running(dind_container_id: str) -> bool:
    probe = run_command(["docker", "inspect", "--format", "{{.State.Running}}", dind_container_id])
    return probe.returncode == 0 and probe.stdout.strip() == "true"


def _dind_log_tail(dind_container_id: str) -> str:
    logs = run_command(["docker", "logs", "--tail", str(DIND_FAILURE_LOG_TAIL_LINES), dind_container_id])
    # dockerd and the pre-bake entrypoint both log to stderr.
    return "\n".join(part.strip() for part in (logs.stdout, logs.stderr) if part and part.strip())


def wait_for_dind_ready(*, dind_container_id: str) -> None:
    timeout_seconds = dind_ready_timeout_seconds()
    deadline = time.monotonic() + timeout_seconds
    last_error = ""
    while True:
        probe = run_command(["docker", "exec", dind_container_id, "docker", "info"])
        if probe.returncode == 0:
            return
        last_error = (probe.stderr or probe.stdout or "").strip()
        if not _dind_container_running(dind_container_id):
            reason = "exited before its daemon became ready"
            break
        if time.monotonic() >= deadline:
            reason = f"did not become ready within {timeout_seconds:g}s"
            break
        time.sleep(DIND_START_SLEEP_SECONDS)
    message = f"Isolated Docker daemon (dind) {reason}: {last_error}"
    log_tail = _dind_log_tail(dind_container_id)
    if log_tail:
        message += f"\ndind container log (last {DIND_FAILURE_LOG_TAIL_LINES} lines):\n{log_tail}"
    raise RuntimeError(message)


def ensure_image_in_isolated_daemon(*, dind_container_id: str, image: str, role: str) -> None:
    if run_command(["docker", "exec", dind_container_id, "docker", "image", "inspect", image]).returncode == 0:
        return
    if run_command(["docker", "exec", dind_container_id, "docker", "pull", image]).returncode == 0:
        if run_command(["docker", "exec", dind_container_id, "docker", "image", "inspect", image]).returncode == 0:
            return

    ensure_docker_image_available(image, role=role)
    save_proc = subprocess.Popen(["docker", "save", image], stdout=subprocess.PIPE)
    try:
        load_proc = subprocess.run(
            ["docker", "exec", "-i", dind_container_id, "docker", "load"],
            stdin=save_proc.stdout,
            capture_output=True,
            text=True,
            check=False,
        )
    finally:
        if save_proc.stdout is not None:
            save_proc.stdout.close()
        save_proc.wait()

    if load_proc.returncode != 0 or save_proc.returncode != 0:
        raise RuntimeError(
            f"Failed to load {role} image {image!r} into the isolated Docker daemon: "
            f"{(load_proc.stderr or load_proc.stdout or '').strip()}"
        )
