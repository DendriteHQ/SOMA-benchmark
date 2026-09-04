from __future__ import annotations

import os
from typing import Any, Mapping

from .soma_tasks import ROLE_ENV, task_image

DEFAULT_SWEREBENCH_NAMESPACE = "swebench"
DEFAULT_INSTANCE_IMAGE_TAG = "latest"


def _first_non_empty_image(hidden_eval: Mapping[str, Any]) -> str | None:
    for key in ("docker_image", "image_name"):
        value = hidden_eval.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def is_swebench_benchmark(benchmark_name: str) -> bool:
    return "swe-bench" in benchmark_name.strip().lower()


def resolve_swebench_namespace(runtime_options: Mapping[str, Any] | None = None) -> str | None:
    raw_value: Any = None
    if runtime_options and "swerebench_namespace" in runtime_options:
        raw_value = runtime_options["swerebench_namespace"]
    elif "SOMA_SWEREBENCH_NAMESPACE" in os.environ:
        raw_value = os.getenv("SOMA_SWEREBENCH_NAMESPACE")

    if isinstance(raw_value, str):
        namespace = raw_value.strip()
        if not namespace:
            return None
        return namespace

    return DEFAULT_SWEREBENCH_NAMESPACE


def derive_swebench_instance_image(
    instance_id: str,
    *,
    namespace: str | None,
    instance_image_tag: str = DEFAULT_INSTANCE_IMAGE_TAG,
) -> str:
    image_key = f"sweb.eval.x86_64.{instance_id.lower()}:{instance_image_tag}"
    if namespace:
        return f"{namespace}/{image_key}".replace("__", "_1776_")
    return image_key


def resolve_benchmark_runtime_image(
    *,
    instance_id: str,
    hidden_eval: Mapping[str, Any],
    runtime_options: Mapping[str, Any] | None = None,
) -> str | None:
    explicit_image = _first_non_empty_image(hidden_eval)
    if explicit_image:
        return explicit_image

    # SOMA task rows name their agent-workspace image directly, so no name has to be derived
    # from the instance id and no SWE-bench namespace convention applies to them.
    soma_env_image = task_image(hidden_eval, ROLE_ENV)
    if soma_env_image:
        return soma_env_image

    benchmark_name = str(hidden_eval.get("benchmark", "")).strip()
    if not benchmark_name or not is_swebench_benchmark(benchmark_name):
        return None

    return derive_swebench_instance_image(
        instance_id,
        namespace=resolve_swebench_namespace(runtime_options),
    )


DEFAULT_PREBAKED_DIND_REPO_ENV = "SOMA_SWEBENCH_DIND_PREBAKED_REPO"


def derive_prebaked_dind_tag(instance_id: str) -> str:
    # SWE-bench instance IDs (e.g. "django__django-11551") are already legal Docker tag
    # characters ([a-zA-Z0-9_][a-zA-Z0-9._-]{0,127}); this is a tag, not a repo path
    # segment, so - unlike derive_swebench_instance_image - no "__" -> "_1776_" mangling.
    return instance_id.lower()


def derive_prebaked_dind_image(instance_id: str, *, repo: str) -> str:
    return f"{repo}:{derive_prebaked_dind_tag(instance_id)}"


#: A SOMA task's own env image lives under a namespace this worker owns (see
#: soma_tasks.task_image), never the public "swebench" one above -- so it is baked
#: into, and resolved from, a repo of its own. The tag scheme is unchanged
#: (derive_prebaked_dind_tag/derive_prebaked_dind_image take any instance id, SOMA
#: task ids included).
DEFAULT_SOMA_TASK_DIND_PREBAKED_REPO_ENV = "SOMA_TASK_DIND_PREBAKED_REPO"

#: Repo name a SOMA task's baked dind image defaults to, alongside the repo its own
#: env image lives in. Must stay equal to quality_worker.config's
#: DEFAULT_PREBAKE_REPOSITORY -- that is the producer of these images and this is the
#: consumer, and nothing checks the two agree at runtime (a mismatch just looks like
#: "no baked image exists", i.e. a silent fall back to the slow path).
DEFAULT_SOMA_TASK_DIND_REPOSITORY = "soma-is-task-dind"


def _strip_tag_and_digest(image_ref: str) -> str:
    """``ns/repo:tag`` or ``host:5000/ns/repo@sha256:...`` -> ``ns/repo`` / ``host:5000/ns/repo``.

    The tag separator is only the last ``:`` that comes *after* the last ``/`` -- a
    registry host's port colon looks identical otherwise.
    """
    ref = image_ref.split("@", 1)[0]
    tail_start = ref.rfind("/") + 1
    colon = ref.find(":", tail_start)
    return ref[:colon] if colon != -1 else ref


def default_soma_task_dind_repo(env_image_ref: str | None) -> str | None:
    """The repo a SOMA task's baked dind image is expected in, derived from its env image.

    A SOMA task's env image reference is carried in the task row itself (see
    :func:`soma_tasks.task_image`), and quality-worker bakes that task's dind image
    into a sibling repo of it -- same registry, same namespace, a different repo name.
    Deriving the default from the reference rather than hardcoding a namespace is what
    lets a plain ``benchmark-solve`` (no issue-scout, no worker, no configuration) still
    find a baked image, while keeping this checkout free of any one deployment's
    account name. ``SOMA_TASK_DIND_PREBAKED_REPO`` still overrides it outright.

    Returns ``None`` for a reference with no namespace component: replacing the sole
    path segment of a bare ``soma-is-tasks:tag`` would name a Docker Hub *library*
    repo this project does not own, so there is nothing sensible to guess.
    """
    if not env_image_ref or not env_image_ref.strip():
        return None
    path = _strip_tag_and_digest(env_image_ref.strip())
    prefix, separator, _repo = path.rpartition("/")
    if not separator:
        return None
    return f"{prefix}/{DEFAULT_SOMA_TASK_DIND_REPOSITORY}"


def resolve_soma_task_dind_repo(runtime_options: Mapping[str, Any] | None = None) -> str | None:
    raw_value: Any = None
    if runtime_options and "soma_task_dind_prebaked_repo" in runtime_options:
        raw_value = runtime_options["soma_task_dind_prebaked_repo"]
    elif DEFAULT_SOMA_TASK_DIND_PREBAKED_REPO_ENV in os.environ:
        raw_value = os.getenv(DEFAULT_SOMA_TASK_DIND_PREBAKED_REPO_ENV)

    if isinstance(raw_value, str):
        repo = raw_value.strip()
        return repo or None

    return None


def resolve_prebaked_dind_repo(runtime_options: Mapping[str, Any] | None = None) -> str | None:
    raw_value: Any = None
    if runtime_options and "swebench_dind_prebaked_repo" in runtime_options:
        raw_value = runtime_options["swebench_dind_prebaked_repo"]
    elif DEFAULT_PREBAKED_DIND_REPO_ENV in os.environ:
        raw_value = os.getenv(DEFAULT_PREBAKED_DIND_REPO_ENV)

    if isinstance(raw_value, str):
        repo = raw_value.strip()
        return repo or None

    return None


def enrich_hidden_eval_with_runtime_image(
    *,
    instance_id: str,
    hidden_eval: Mapping[str, Any],
    runtime_options: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload = dict(hidden_eval)
    runtime_image = resolve_benchmark_runtime_image(
        instance_id=instance_id,
        hidden_eval=payload,
        runtime_options=runtime_options,
    )
    if runtime_image:
        payload.setdefault("docker_image", runtime_image)
        payload.setdefault("image_name", runtime_image)
    return payload