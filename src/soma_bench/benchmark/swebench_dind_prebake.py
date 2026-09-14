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
import shutil
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

OVERLAY_XATTR_PATTERN = "trusted.overlay.*"

PREBAKE_ENTRYPOINT = """#!/bin/sh
set -e

PREBAKED_TAR=/prebaked.tar
MARKER=/var/lib/docker/.soma-prebaked
TAR=/usr/bin/tar

if [ -f "$PREBAKED_TAR" ] && [ ! -e "$MARKER" ]; then
    if ! "$TAR" --version 2>/dev/null | head -1 | grep -qi 'gnu tar'; then
        echo "soma-prebake: $TAR is not GNU tar; refusing to unpack without xattr support" >&2
        exit 1
    fi
    echo "soma-prebake: unpacking baked docker storage into /var/lib/docker" >&2
    "$TAR" --xattrs --xattrs-include='OVERLAY_XATTR_PATTERN_PLACEHOLDER' --numeric-owner \\
        -xf "$PREBAKED_TAR" -C /var/lib/docker
    touch "$MARKER"
    echo "soma-prebake: unpack done" >&2
fi

exec dockerd-entrypoint.sh "$@"
"""
PREBAKE_ENTRYPOINT = PREBAKE_ENTRYPOINT.replace(
    "OVERLAY_XATTR_PATTERN_PLACEHOLDER", OVERLAY_XATTR_PATTERN
)
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


def _count_opaque_xattrs(tar_path: Path) -> int:
    """How many directories in the archive carry `trusted.overlay.opaque`.

    Read back off the finished archive rather than counted while walking the tree, so
    what is reported is what a consumer will actually unpack.
    """
    import tarfile

    key = "SCHILY.xattr.trusted.overlay.opaque"
    with tarfile.open(tar_path) as archive:
        return sum(1 for member in archive if key in member.pax_headers)


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
    # An explicit bind mount for the dind's storage, rather than the anonymous volume the
    # base image's `VOLUME /var/lib/docker` would otherwise create. Two reasons: the tar
    # step below needs to read this tree from the host with GNU tar (docker cp cannot -
    # it drops xattrs, see OVERLAY_XATTR_PATTERN), and knowing the path outright beats
    # inspecting Docker's own volume bookkeeping for it. It also removes the multi-GB
    # `docker cp` this function used to do: the inner dockerd now writes the only copy
    # that is ever made.
    storage_dir = Path(tempfile.mkdtemp(prefix="soma-dind-prebake-storage-"))
    _log(f"[{instance_id}] starting dind ({base_dind_image}) as {container_name}")
    try:
        run_result = dind_utils.run_command([
            "docker", "run", "-d", "--privileged",
            "--name", container_name,
            "-e", "DOCKER_TLS_CERTDIR=",
            "-v", f"{storage_dir}:/var/lib/docker",
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

        # `docker commit` is not an option here: `docker:dind` declares /var/lib/docker as
        # a VOLUME and a commit only captures the container's own writable layer, so it
        # produces an image with an empty storage tree (confirmed by hand: commit ->
        # fresh container -> `docker images` inside comes back empty). Nor can the tree be
        # COPYed into an image layer - see PREBAKE_ENTRYPOINT for why that is not a
        # tunable failure but a property of overlayfs. What does work: carry the tree as
        # a single tar *file* in the image (inert regular file, no special files in the
        # layer at all) and unpack it into the mounted volume on first start.
        with tempfile.TemporaryDirectory(prefix="soma-dind-prebake-") as build_context_dir:
            context = Path(build_context_dir)
            tar_path = context / "prebaked.tar"
            _log(f"[{instance_id}] archiving dind storage (several GB, can take a minute)")
            # GNU tar on the host, not `docker cp`/`docker export`: those go through
            # Docker's own archive code, which does not carry trusted.* xattrs, and losing
            # them is silent content corruption rather than an error (see
            # OVERLAY_XATTR_PATTERN). --numeric-owner because the ids in this tree are the
            # inner daemon's, and must not be remapped through the build host's passwd.
            tar_result = dind_utils.run_command([
                "tar",
                "--xattrs",
                f"--xattrs-include={OVERLAY_XATTR_PATTERN}",
                "--numeric-owner",
                "-cf", str(tar_path),
                "-C", str(storage_dir),
                ".",
            ])
            if tar_result.returncode != 0:
                return BuildResult(instance_id, status="failed", detail=f"tar failed: {tar_result.stderr}")

            # Counted and logged because losing these is the one failure mode of this
            # whole approach that produces a working-looking image with wrong content -
            # a run that reports 0 here for a task whose build deletes directories is the
            # signal that the xattr path broke, and there is nothing else to notice it by.
            opaque = _count_opaque_xattrs(tar_path)
            _log(f"[{instance_id}] archived {tar_path.stat().st_size} bytes, {opaque} opaque-dir marker(s)")

            # Free the storage tree before the build rather than in the `finally` below:
            # the tar is a full second copy, and holding both across a build that also
            # streams the context into BuildKit is the peak disk figure for a whole run.
            shutil.rmtree(storage_dir, ignore_errors=True)

            entrypoint_path = context / "prebake-entrypoint.sh"
            entrypoint_path.write_text(PREBAKE_ENTRYPOINT, encoding="utf-8")
            # COPY carries the mode straight from the build context, and a 0644 entrypoint
            # fails at container start with a bare "executable file not found in $PATH".
            entrypoint_path.chmod(0o755)
            # `apk add tar` assumes an Alpine-based dind (which every `docker:*-dind` tag
            # is). A non-Alpine --base-dind-image fails loudly right here, which is the
            # intent: the alternative is busybox tar silently dropping xattrs at runtime.
            (context / "Dockerfile").write_text(
                f"FROM {base_dind_image}\n"
                "RUN apk add --no-cache tar\n"
                "COPY prebaked.tar /prebaked.tar\n"
                "COPY prebake-entrypoint.sh /usr/local/bin/soma-prebake-entrypoint.sh\n"
                'ENTRYPOINT ["soma-prebake-entrypoint.sh"]\n',
                encoding="utf-8",
            )
            _log(f"[{instance_id}] building baked image {target_image}")
            # BuildKit keeps the exported layer *and* the build context in its own content
            # store, and `docker rmi` does not reclaim either - only `docker builder prune`
            # does. Measured leak before that call existed: 358GB reclaimable after ~22
            # builds with only 4 baked images on disk (~48GB/h, enough to fill the disk
            # part-way into a 500-instance run).
            build_result = dind_utils.run_command(
                ["docker", "build", "-t", target_image, str(context)],
            )
            if build_result.returncode != 0:
                return BuildResult(instance_id, status="failed", detail=f"docker build failed: {build_result.stderr}")
            prune_result = dind_utils.run_command(["docker", "builder", "prune", "-f"])
            if prune_result.returncode != 0:
                # Non-fatal: the bake itself succeeded, this is just cache hygiene.
                _log(f"[{instance_id}] warning: docker builder prune failed: {prune_result.stderr}")

        # Verification pass: fresh container from the baked image, no load step.
        _log(f"[{instance_id}] verifying baked image from a fresh container")
        verify_run = dind_utils.run_command([
            "docker", "run", "-d", "--privileged",
            "--name", verify_container_name,
            target_image,
        ])
        if verify_run.returncode != 0:
            # `docker run -d` can create the container and still fail to start it (a bad
            # entrypoint is the usual way), which leaves it behind in Created state -
            # observed in practice, so the removal cannot live only in the try/finally
            # below that this early return skips.
            dind_utils.run_command(["docker", "rm", "-f", "-v", verify_container_name])
            return BuildResult(instance_id, status="failed", detail=f"verify container failed to start: {verify_run.stderr}")
        try:
            dind_utils.wait_for_dind_ready(dind_container_id=verify_container_name)
            # The unpack marker, checked before the image itself: if the entrypoint
            # skipped or half-did the unpack, `docker image inspect` failing below would
            # read as "the bake lost the image" when the real fault is one step earlier.
            marker_result = dind_utils.run_command([
                "docker", "exec", verify_container_name,
                "test", "-e", "/var/lib/docker/.soma-prebaked",
            ])
            if marker_result.returncode != 0:
                return BuildResult(
                    instance_id, status="failed",
                    detail="verification failed: baked image did not unpack its storage tree",
                )
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
        # The bind-mounted storage tree is ours, not Docker's, so `docker rm -v` does not
        # touch it - several GB per instance if left behind.
        shutil.rmtree(storage_dir, ignore_errors=True)
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
    parser.add_argument(
        "--instance-id",
        default=None,
        help=(
            "Bake exactly this one instance instead of a dataset split -- pairs with "
            "--source-image, and skips the dataset load, --select/--limit filtering and "
            "the orphaned-container sweep entirely (that sweep removes every "
            f"'{CONTAINER_NAME_PREFIX}*' container on the host, which is only safe for a "
            "lone offline batch run, not for several single-instance calls in flight at "
            "once -- see quality_worker.prebake, the caller this mode exists for)."
        ),
    )
    parser.add_argument(
        "--source-image",
        default=None,
        help=(
            "Explicit source image for --instance-id, bypassing --namespace/"
            "derive_swebench_instance_image entirely. Required with --instance-id: a "
            "SOMA task's env image is never named by that convention (see "
            "swebench_images.resolve_benchmark_runtime_image)."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true", help="Rebuild even if the target tag already exists.")
    parser.add_argument("--no-push", action="store_true", help="Build/commit/verify locally, skip login and push.")
    parser.add_argument("--keep-source-images", action="store_true")
    parser.add_argument("--base-dind-image", default=DEFAULT_BASE_DIND_IMAGE)
    args = parser.parse_args(argv)

    if bool(args.instance_id) != bool(args.source_image):
        parser.error("--instance-id and --source-image must be given together")

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

    if args.instance_id:
        # Single-instance mode: no dataset, no --select/--limit, and deliberately no
        # cleanup_orphaned_containers() -- see the --instance-id help text above for why.
        instance_ids = [args.instance_id]
    else:
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
        source_image = (
            args.source_image
            if args.instance_id
            else derive_swebench_instance_image(instance_id, namespace=namespace)
        )
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
