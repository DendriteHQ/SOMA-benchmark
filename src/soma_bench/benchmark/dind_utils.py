"""Shared Docker-in-Docker (dind) helpers.

Used by both the Copilot runtime backend (backends/copilot/copilot.py) and the offline
swebench_dind_prebake build tool, so a pre-baked image and the runtime's save/load
fallback are guaranteed to produce byte-identical results - both paths load a target
image into an isolated dind daemon using this exact code.
"""

from __future__ import annotations

import subprocess
import time
from pathlib import Path

DIND_START_RETRIES = 60
DIND_START_SLEEP_SECONDS = 0.5


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

    pull_result = run_command(["docker", "pull", image_ref])
    if pull_result.returncode == 0 and docker_image_exists(image_ref):
        return

    message = (pull_result.stderr or pull_result.stdout or "").strip()
    raise RuntimeError(
        f"{role} Docker image is not available locally and automatic pull failed for {image_ref}: {message}"
    )


def wait_for_dind_ready(*, dind_container_id: str) -> None:
    last_error = ""
    for _ in range(DIND_START_RETRIES):
        probe = run_command(["docker", "exec", dind_container_id, "docker", "info"])
        if probe.returncode == 0:
            return
        last_error = (probe.stderr or probe.stdout or "").strip()
        time.sleep(DIND_START_SLEEP_SECONDS)
    raise RuntimeError(f"Isolated Docker daemon (dind) did not become ready: {last_error}")


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
