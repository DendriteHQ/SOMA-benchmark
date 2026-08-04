"""Offline build tool: bake SWE-bench sandbox images into pre-loaded dind images.

For each target instance, this starts a throwaway `docker:dind` container, loads the
instance's SWE-bench sandbox image into it (reusing the exact same code path the Copilot
runtime backend uses as its save/load fallback - see dind_utils.py), commits the stopped
container to a new image, verifies it from a fresh container, and pushes it to a Docker
Hub repo. The benchmark RUNTIME never builds anything - it only ever pulls a pre-baked
tag if one exists (backends/copilot/copilot.py's `_resolve_effective_dind_image`), and
falls back to today's on-demand save/load otherwise.

Safely re-runnable: Docker Hub itself is the source of truth for "already built" (via
`docker manifest inspect`), so a crashed/interrupted run just picks up where it left off.

Local-only mode: if DOCKERHUB_USERNAME/DOCKERHUB_TOKEN are not set (and --no-push is not
passed), the tool still builds, commits, and verifies each image, but skips login/push and
leaves the result as a local image tagged `<repo>:<instance-tag>` - useful for validating
the whole mechanism before wiring up real Docker Hub credentials.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import tempfile
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from . import dind_utils
from .cache_paths import benchmark_cache_root
from .manifest import load_selected_instance_ids
from .registry_auth import docker_env_for_image
from .runner_settings import load_repo_env
from .swebench_images import (
    derive_prebaked_dind_image,
    derive_swebench_instance_image,
    resolve_swebench_namespace,
)

DEFAULT_DATASET = "SWE-bench/SWE-bench_Verified"
DEFAULT_SPLIT = "test"
DEFAULT_BASE_DIND_IMAGE = "docker:27-dind"
DEFAULT_CONCURRENCY = 2
CONTAINER_NAME_PREFIX = "soma-dind-prebake-"

# Instance-id prefixes that are never pre-baked. SWE-bench instance ids are
# `<org>__<repo>-<pr>`, so a prefix match on `<org>__` drops a whole repo's instances.
EXCLUDED_INSTANCE_PREFIXES = ("matplotlib__",)

_LOG_LOCK = threading.Lock()


def _log(message: str) -> None:
    # Locked so interleaved output from concurrent build_one() workers doesn't get
    # scrambled mid-line.
    with _LOG_LOCK:
        print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def _first_non_empty(*values: Any) -> str:
    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _sanitize_for_container_name(instance_id: str) -> str:
    return "".join(char if char.isalnum() or char in "-_." else "-" for char in instance_id.lower())


def _cache_slug(value: str) -> str:
    # Mirrors runner_settings.py's private _slug() exactly, so this resolves to the same
    # on-disk path the normal benchmark-solve/run-infer flow already populates.
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-._")
    return normalized or "default"


def _cached_rows_path(dataset: str, split: str, *, dataset_config: str = "default") -> Path:
    return (
        benchmark_cache_root()
        / _cache_slug(dataset)
        / "splits"
        / _cache_slug(dataset_config)
        / _cache_slug(split)
        / "rows.jsonl"
    )


def load_all_instance_ids(dataset: str, split: str, *, dataset_config: str = "default") -> list[str]:
    """Enumerate every instance_id for a dataset split.

    Prefers the plain JSONL row cache this repo's own benchmark-solve/run-infer flow
    already populates offline at `<benchmark_cache_root()>/.../rows.jsonl` (no network,
    no `datasets` import needed) - only falls back to `datasets.load_dataset` if that
    cache doesn't exist yet (e.g. this dataset/split has never been resolved before).
    """
    cache_path = _cached_rows_path(dataset, split, dataset_config=dataset_config)
    if cache_path.is_file():
        instance_ids: set[str] = set()
        with cache_path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                row = payload.get("row") if isinstance(payload, dict) else None
                instance_id = row.get("instance_id") if isinstance(row, dict) else None
                if instance_id:
                    instance_ids.add(str(instance_id))
        if instance_ids:
            return sorted(instance_ids)

    from datasets import DownloadConfig, load_dataset

    rows = load_dataset(dataset, split=split, download_config=DownloadConfig(local_files_only=True))
    return sorted({str(row["instance_id"]) for row in rows})


def is_excluded_instance(instance_id: str) -> bool:
    return instance_id.startswith(EXCLUDED_INSTANCE_PREFIXES)


def image_exists_on_registry(image_ref: str) -> bool:
    return dind_utils.run_command(
        ["docker", "manifest", "inspect", image_ref],
        env=docker_env_for_image(image_ref),
    ).returncode == 0


def verify_dockerhub_credentials(*, username: str, token: str, image_ref: str) -> None:
    """Fail fast on a bad/expired token instead of 500 manifest-inspect misses later.

    Runs against the same process-private DOCKER_CONFIG every push/manifest call uses, so
    validating credentials here does not write an auth entry into ~/.docker/config.json.
    `--password-stdin` reads the password from stdin; dind_utils.run_command doesn't
    support passing stdin input, so this call is made directly here instead.
    """
    result = subprocess.run(
        ["docker", "login", "--username", username, "--password-stdin"],
        input=token,
        env=docker_env_for_image(image_ref),
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "Docker Hub credentials from DOCKERHUB_USERNAME/DOCKERHUB_TOKEN were rejected: "
            f"{(result.stderr or result.stdout or '').strip()}"
        )


def cleanup_orphaned_containers() -> None:
    listing = dind_utils.run_command(["docker", "ps", "-a", "--format", "{{.Names}}"])
    if listing.returncode != 0:
        return
    for name in (listing.stdout or "").splitlines():
        name = name.strip()
        if name.startswith(CONTAINER_NAME_PREFIX):
            dind_utils.run_command(["docker", "rm", "-f", "-v", name])


class BuildResult:
    def __init__(self, instance_id: str, *, status: str, detail: str = "") -> None:
        self.instance_id = instance_id
        self.status = status  # "built", "skipped", "failed"
        self.detail = detail


def build_one(
    instance_id: str,
    *,
    source_image: str,
    target_image: str,
    base_dind_image: str,
    push: bool,
    keep_source_image: bool,
) -> BuildResult:
    container_name = f"{CONTAINER_NAME_PREFIX}{_sanitize_for_container_name(instance_id)}-{uuid.uuid4().hex[:8]}"
    verify_container_name = f"{container_name}-verify"
    _log(f"[{instance_id}] starting dind ({base_dind_image}) as {container_name}")
    try:
        run_result = dind_utils.run_command([
            "docker", "run", "-d", "--privileged",
            "--name", container_name,
            "-e", "DOCKER_TLS_CERTDIR=",
            base_dind_image,
        ])
        if run_result.returncode != 0:
            return BuildResult(instance_id, status="failed", detail=f"docker run failed: {run_result.stderr}")

        dind_utils.wait_for_dind_ready(dind_container_id=container_name)
        _log(f"[{instance_id}] dind ready, loading {source_image} (large images can take a few minutes)")
        dind_utils.ensure_image_in_isolated_daemon(
            dind_container_id=container_name,
            image=source_image,
            role="prebake",
        )
        _log(f"[{instance_id}] image loaded, stopping dind cleanly before baking")

        dind_utils.run_command(["docker", "exec", container_name, "sync"])
        stop_result = dind_utils.run_command(["docker", "stop", "--timeout", "60", container_name])
        if stop_result.returncode != 0:
            return BuildResult(instance_id, status="failed", detail=f"docker stop failed: {stop_result.stderr}")
        exit_code = dind_utils.run_command(
            ["docker", "inspect", "-f", "{{.State.ExitCode}}", container_name]
        ).stdout.strip()
        if exit_code not in ("0", ""):
            return BuildResult(
                instance_id, status="failed",
                detail=f"dind did not shut down cleanly (exit code {exit_code}); not baking",
            )

        # NOTE: `docker:dind` declares /var/lib/docker as a VOLUME, and `docker commit`
        # never captures volume data - only the container's own writable layer. A plain
        # commit here silently produces an image with an EMPTY /var/lib/docker (confirmed
        # by hand: commit -> fresh container -> `docker images` inside comes back empty).
        # The correct approach: `docker cp` the volume's actual contents out to the host,
        # then `docker build` a new image with a Dockerfile that COPies that data back into
        # /var/lib/docker. Because that path is still a declared VOLUME (inherited from the
        # base image), Docker auto-creates a fresh anonymous volume for it on the next
        # container start - and since the image itself has content there, Docker seeds the
        # new anonymous volume from that image content (documented Docker volume behavior).
        # Confirmed working end-to-end by hand before wiring this into the script.
        with tempfile.TemporaryDirectory(prefix="soma-dind-prebake-") as build_context_dir:
            volume_copy_dir = Path(build_context_dir) / "varlibdocker"
            volume_copy_dir.mkdir()
            _log(f"[{instance_id}] extracting dind volume data (docker cp, several GB, can take a minute)")
            cp_result = dind_utils.run_command([
                "docker", "cp", f"{container_name}:/var/lib/docker/.", str(volume_copy_dir),
            ])
            if cp_result.returncode != 0:
                return BuildResult(instance_id, status="failed", detail=f"docker cp failed: {cp_result.stderr}")

            dockerfile_path = Path(build_context_dir) / "Dockerfile"
            dockerfile_path.write_text(
                f"FROM {base_dind_image}\nCOPY varlibdocker /var/lib/docker\n",
                encoding="utf-8",
            )
            _log(f"[{instance_id}] building baked image {target_image}")
            # DOCKER_BUILDKIT=0 forces the classic builder deliberately. BuildKit keeps the
            # COPY layer *and* the whole build context in its own content store, which the
            # `docker rmi target_image` below does NOT reclaim - only `docker builder prune`
            # does, and nothing here calls it. Measured leak: 358GB of reclaimable cache
            # after ~22 builds while only 4 baked images existed locally (~48GB/h, would have
            # filled the disk ~13h into a 500-instance run). The classic builder writes plain
            # image layers instead, so the existing rmi frees everything. Verified by A/B on
            # docker 29.1.4: classic build leaves the cache byte-identical, BuildKit grows it
            # and keeps the growth after rmi.
            build_result = dind_utils.run_command(
                ["docker", "build", "-t", target_image, build_context_dir],
                env={**os.environ, "DOCKER_BUILDKIT": "0"},
            )
            if build_result.returncode != 0:
                return BuildResult(instance_id, status="failed", detail=f"docker build failed: {build_result.stderr}")

        # Verification pass: fresh container from the baked image, no load step.
        _log(f"[{instance_id}] verifying baked image from a fresh container")
        verify_run = dind_utils.run_command([
            "docker", "run", "-d", "--privileged",
            "--name", verify_container_name,
            target_image,
        ])
        if verify_run.returncode != 0:
            return BuildResult(instance_id, status="failed", detail=f"verify container failed to start: {verify_run.stderr}")
        try:
            dind_utils.wait_for_dind_ready(dind_container_id=verify_container_name)
            inspect_result = dind_utils.run_command(
                ["docker", "exec", verify_container_name, "docker", "image", "inspect", source_image]
            )
            if inspect_result.returncode != 0:
                return BuildResult(
                    instance_id, status="failed",
                    detail="verification failed: baked image does not contain the source image",
                )
        finally:
            dind_utils.run_command(["docker", "rm", "-f", "-v", verify_container_name])

        if push:
            _log(f"[{instance_id}] pushing {target_image} to registry")
            push_result = dind_utils.run_command(
                ["docker", "push", target_image],
                env=docker_env_for_image(target_image),
            )
            if push_result.returncode != 0:
                return BuildResult(instance_id, status="failed", detail=f"docker push failed: {push_result.stderr}")
        else:
            _log(f"[{instance_id}] local-only mode, skipping push - keeping {target_image} on this machine")

        # Only drop the local copy of the target image once it's safely on the registry -
        # in local-only mode (push=False) it's the actual deliverable, keep it.
        if push:
            dind_utils.run_command(["docker", "rmi", target_image])
        return BuildResult(instance_id, status="built")
    finally:
        # -v matters here: docker:dind declares /var/lib/docker as a VOLUME, so every
        # container started from it gets an anonymous volume even without an explicit -v
        # flag. Without -v on rm, that volume (several GB, holding the loaded swebench
        # image) is orphaned and never reclaimed - confirmed leaking ~1GB/instance in
        # practice before this fix (docker volume prune recovered 10.8GB from ~14 builds).
        dind_utils.run_command(["docker", "rm", "-f", "-v", container_name])
        if not keep_source_image:
            dind_utils.run_command(["docker", "rmi", source_image])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m soma_bench.benchmark.swebench_dind_prebake")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument("--repo", required=True, help="Target Docker Hub repo, e.g. myorg/soma-swebench-dind.")
    parser.add_argument("--namespace", default=None, help="Source swebench image namespace override.")
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--select", default=None, help="Optional file with one instance_id per line.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true", help="Rebuild even if the target tag already exists.")
    parser.add_argument("--no-push", action="store_true", help="Build/commit/verify locally, skip login and push.")
    parser.add_argument("--keep-source-images", action="store_true")
    parser.add_argument("--base-dind-image", default=DEFAULT_BASE_DIND_IMAGE)
    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parents[3]
    load_repo_env(repo_root)

    username = _first_non_empty(os.getenv("DOCKERHUB_USERNAME"))
    token = _first_non_empty(os.getenv("DOCKERHUB_TOKEN"))
    push = not args.no_push and bool(username and token)
    if not args.no_push and not push:
        _log("[prebake] DOCKERHUB_USERNAME/DOCKERHUB_TOKEN not set - running in local-only mode (no login/push).")

    if push and not args.dry_run:
        _log("[prebake] validating Docker Hub credentials (token-scoped config, ~/.docker untouched)")
        verify_dockerhub_credentials(username=username, token=token, image_ref=args.repo)

    cleanup_orphaned_containers()

    instance_ids = load_all_instance_ids(args.dataset, args.split)
    excluded = [i for i in instance_ids if is_excluded_instance(i)]
    if excluded:
        instance_ids = [i for i in instance_ids if not is_excluded_instance(i)]
        _log(
            f"[prebake] excluding {len(excluded)} instance(s) matching "
            f"{', '.join(EXCLUDED_INSTANCE_PREFIXES)} (not pre-baked)"
        )
    if args.select:
        selected = load_selected_instance_ids(args.select)
        instance_ids = [i for i in instance_ids if i in selected]
    if args.limit > 0:
        instance_ids = instance_ids[: args.limit]

    namespace = args.namespace if args.namespace is not None else resolve_swebench_namespace(None)

    _log(f"[prebake] checking registry state for {len(instance_ids)} instance(s)...")
    plan: list[tuple[str, str, str]] = []
    skipped = 0
    for scan_index, instance_id in enumerate(instance_ids, start=1):
        source_image = derive_swebench_instance_image(instance_id, namespace=namespace)
        target_image = derive_prebaked_dind_image(instance_id, repo=args.repo)
        if not args.force and push and image_exists_on_registry(target_image):
            skipped += 1
        else:
            plan.append((instance_id, source_image, target_image))
        if scan_index % 25 == 0 or scan_index == len(instance_ids):
            _log(f"[prebake] registry check progress: {scan_index}/{len(instance_ids)} "
                 f"({skipped} already built, {len(plan)} pending so far)")

    _log(f"[prebake] {len(instance_ids)} instance(s) selected, {skipped} already on registry, {len(plan)} to build")
    if args.dry_run:
        for instance_id, source_image, target_image in plan:
            _log(f"[dry-run] {instance_id}: {source_image} -> {target_image}")
        return 0

    total = len(plan)
    completed = 0
    built_count = 0
    failed_count = 0
    results: list[BuildResult] = []
    with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as executor:
        futures = {
            executor.submit(
                build_one,
                instance_id,
                source_image=source_image,
                target_image=target_image,
                base_dind_image=args.base_dind_image,
                push=push,
                keep_source_image=args.keep_source_images,
            ): instance_id
            for instance_id, source_image, target_image in plan
        }
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            completed += 1
            remaining = total - completed
            if result.status == "built":
                built_count += 1
            elif result.status == "failed":
                failed_count += 1
            detail = f" ({result.detail})" if result.detail else ""
            _log(
                f"[prebake] {result.instance_id}: {result.status}{detail} | "
                f"progress: {completed}/{total} done "
                f"(built={built_count}, failed={failed_count}) - {remaining} remaining"
            )

    failed = [r for r in results if r.status == "failed"]
    built = [r for r in results if r.status == "built"]
    _log(f"[prebake] done: built={len(built)} failed={len(failed)} skipped={skipped}")
    if failed:
        _log("[prebake] failed instances: " + ", ".join(r.instance_id for r in failed))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
