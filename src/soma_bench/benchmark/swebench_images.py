from __future__ import annotations

import os
from typing import Any, Mapping

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

    benchmark_name = str(hidden_eval.get("benchmark", "")).strip()
    if not benchmark_name or not is_swebench_benchmark(benchmark_name):
        return None

    return derive_swebench_instance_image(
        instance_id,
        namespace=resolve_swebench_namespace(runtime_options),
    )


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