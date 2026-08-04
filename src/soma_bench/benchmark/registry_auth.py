"""Token-based Docker registry auth for pulls/pushes of pre-baked dind images.

Both the Copilot runtime (pulling a pre-baked dind image) and the offline
swebench_dind_prebake build tool need Docker Hub credentials for a private repo. Relying
on an ambient `docker login` in ~/.docker/config.json makes every run depend on host
state nothing in this repo controls: the entry silently expires, a concurrent
`docker login`/`logout` elsewhere on the box rewrites the same file, and a freshly
provisioned machine simply doesn't have it - in which case the runtime quietly degrades
to the slow on-demand save/load path.

Instead, DOCKERHUB_USERNAME/DOCKERHUB_TOKEN are materialized into a private,
process-scoped DOCKER_CONFIG directory, and every auth'd docker call is passed that
directory. Writing the base64 `auths` entry is exactly what `docker login` does for the
default plaintext store, minus the network round-trip and minus delegation to whatever
global credsStore/credHelper the host happens to have configured. ~/.docker/config.json
is neither read nor written on this path.
"""

from __future__ import annotations

import atexit
import base64
import json
import os
import shutil
import tempfile
import threading
from pathlib import Path

DOCKERHUB_USERNAME_ENV = "DOCKERHUB_USERNAME"
DOCKERHUB_TOKEN_ENV = "DOCKERHUB_TOKEN"

# The registry key `docker login` itself uses for Docker Hub - the CLI looks up exactly
# this string for any docker.io reference, so the auth entry must be filed under it.
DOCKERHUB_REGISTRY_KEY = "https://index.docker.io/v1/"
DOCKERHUB_HOSTS = frozenset({"docker.io", "index.docker.io", "registry-1.docker.io"})

_CONFIG_LOCK = threading.Lock()
_CONFIG_DIR: Path | None = None


def dockerhub_credentials() -> tuple[str, str] | None:
    username = (os.getenv(DOCKERHUB_USERNAME_ENV) or "").strip()
    token = (os.getenv(DOCKERHUB_TOKEN_ENV) or "").strip()
    if not username or not token:
        return None
    return username, token


def targets_dockerhub(image_ref: str) -> bool:
    """True if `image_ref` resolves to Docker Hub (so Docker Hub creds apply to it)."""
    reference = image_ref.strip()
    if not reference:
        return False
    if "/" not in reference:
        # Single-component reference is always a Docker Hub library image; the only colon
        # it can carry is the tag separator ("ubuntu:24.04"), never a registry port.
        return True
    # Docker's own reference grammar: the part before the first "/" is a registry host only
    # if it contains "." or ":" or is exactly "localhost". Otherwise it's a Hub namespace
    # ("dendritehq/soma-swebench-dind:tag").
    first_segment = reference.split("/", 1)[0]
    is_registry_host = "." in first_segment or ":" in first_segment or first_segment == "localhost"
    if not is_registry_host:
        return True
    return first_segment.split(":", 1)[0].lower() in DOCKERHUB_HOSTS


def _ensure_config_dir(username: str, token: str) -> Path:
    global _CONFIG_DIR
    with _CONFIG_LOCK:
        if _CONFIG_DIR is not None and (_CONFIG_DIR / "config.json").is_file():
            return _CONFIG_DIR

        # mkdtemp creates the directory 0700, so the token never lands in a world-readable
        # path; the file itself is tightened to 0600 for the same reason.
        directory = Path(tempfile.mkdtemp(prefix="soma-docker-auth-"))
        encoded = base64.b64encode(f"{username}:{token}".encode()).decode("ascii")
        config_path = directory / "config.json"
        config_path.write_text(
            json.dumps({"auths": {DOCKERHUB_REGISTRY_KEY: {"auth": encoded}}}),
            encoding="utf-8",
        )
        os.chmod(config_path, 0o600)
        atexit.register(shutil.rmtree, directory, ignore_errors=True)
        _CONFIG_DIR = directory
        return directory


def docker_env_for_image(image_ref: str) -> dict[str, str]:
    """Environment for a docker CLI call that may need Docker Hub auth for `image_ref`.

    Always returns a full environment (docker helpers pass it straight to subprocess).
    When DOCKERHUB_USERNAME/DOCKERHUB_TOKEN are set and the reference points at Docker
    Hub, DOCKER_CONFIG is redirected at the token-derived config dir; otherwise the
    ambient environment is returned unchanged and docker falls back to its usual
    ~/.docker/config.json lookup.
    """
    env = dict(os.environ)
    credentials = dockerhub_credentials()
    if credentials is None or not targets_dockerhub(image_ref):
        return env
    env["DOCKER_CONFIG"] = str(_ensure_config_dir(*credentials))
    return env


def uses_token_auth(image_ref: str) -> bool:
    """Whether docker_env_for_image() would authenticate `image_ref` from the env token."""
    return dockerhub_credentials() is not None and targets_dockerhub(image_ref)
