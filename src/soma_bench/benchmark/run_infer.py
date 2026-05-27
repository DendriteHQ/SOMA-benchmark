from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .backends import get_runtime_backend, list_runtime_backends
from .backends.base import RuntimeExecutionContext
from .config import DEFAULTS, SUPPORTED_WORKSPACES, build_agent_image_ref, default_output_path
from .execution import execute_runtime_contexts
from .manifest import BenchmarkInstance, load_manifest, load_selected_instance_ids
from .runner_settings import build_llm_config, load_repo_env, resolve_benchmark_concurrency
from .swebench_images import resolve_benchmark_runtime_image


def _append_jsonl_row(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def _build_runtime_options(
    *,
    openclaw_command: str | None,
    openclaw_container_image: str | None,
    openclaw_current_user: bool,
    openclaw_run_id_header_value: str | None,
    openclaw_ignore_api_key: bool,
    openclaw_plugin_path: str | None,
    openclaw_disable_plugin: bool,
    openclaw_plugin_reinstall_on_run_start: bool,
    swerebench_eval: bool,
    swerebench_split: str | None,
    swerebench_harness_root: str | None,
    swerebench_harness_python: str | None,
    swerebench_namespace: str | None,
    swerebench_cache_level: str | None,
    swerebench_timeout: int | None,
    swerebench_max_workers: int | None,
) -> dict[str, Any]:
    runtime_options: dict[str, Any] = {}
    if openclaw_command:
        runtime_options["openclaw_command"] = openclaw_command
    if openclaw_container_image:
        runtime_options["openclaw_container_image"] = openclaw_container_image
    if openclaw_current_user:
        runtime_options["openclaw_user"] = "current"
    if openclaw_run_id_header_value:
        runtime_options["openclaw_run_id_header_value"] = openclaw_run_id_header_value
    if openclaw_ignore_api_key:
        runtime_options["openclaw_ignore_api_key"] = True
    if openclaw_plugin_path:
        runtime_options["openclaw_plugin_path"] = openclaw_plugin_path
    if openclaw_disable_plugin:
        runtime_options["openclaw_plugin_enabled"] = False
    if openclaw_plugin_reinstall_on_run_start:
        runtime_options["openclaw_plugin_reinstall_on_run_start"] = True
    if swerebench_eval:
        runtime_options["swerebench_eval_enabled"] = True
    if swerebench_split:
        runtime_options["swerebench_split"] = swerebench_split
    if swerebench_harness_root:
        runtime_options["swerebench_harness_root"] = swerebench_harness_root
    if swerebench_harness_python:
        runtime_options["swerebench_harness_python"] = swerebench_harness_python
    if swerebench_namespace is not None:
        runtime_options["swerebench_namespace"] = swerebench_namespace
    if swerebench_cache_level:
        runtime_options["swerebench_cache_level"] = swerebench_cache_level
    if swerebench_timeout is not None:
        runtime_options["swerebench_timeout"] = swerebench_timeout
    if swerebench_max_workers is not None:
        runtime_options["swerebench_max_workers"] = swerebench_max_workers
    return runtime_options


def _resolve_runtime_container_image(
    instance: BenchmarkInstance,
    runtime_options: dict[str, Any] | None = None,
) -> str | None:
    return resolve_benchmark_runtime_image(
        instance_id=instance.instance_id,
        hidden_eval=instance.hidden_eval,
        runtime_options=runtime_options,
    )


def build_run_plan(
    instances: list[BenchmarkInstance],
    workspace: str,
    agent_image: str,
    build_target: str,
    runtime_options: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    plan: list[dict[str, Any]] = []
    for instance in instances:
        stage_label = (
            instance.expected_next_action.strip()
            or instance.expected_next_stage.strip()
            or instance.stage_boundary.strip()
            or "resume"
        )
        plan.append(
            {
                "benchmark_id": instance.benchmark_id,
                "trajectory_id": instance.trajectory_id,
                "instance_id": instance.instance_id,
                "repo": instance.repo,
                "workspace": workspace,
                "user_prompt": instance.user_prompt,
                "stage_count": 1,
                "stage_names": [stage_label],
                "source_stage": instance.stage_boundary,
                "resume_stage": instance.expected_next_stage,
                "expected_next_action": instance.expected_next_action,
                "resume_prompt": instance.resume_prompt,
                "agent_image": build_agent_image_ref(
                    instance.instance_id,
                    image=agent_image,
                    target=build_target,
                ),
                "runtime_container_image": _resolve_runtime_container_image(instance, runtime_options),
                "runtime_options": dict(runtime_options or {}),
            }
        )
    return plan


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m soma_bench.benchmark.run_infer")
    parser.add_argument("--dataset", required=True, help="Path to the SOMA benchmark manifest JSONL.")
    parser.add_argument("--split", default="test", help="Logical split label for reporting.")
    parser.add_argument(
        "--runtime-backend",
        choices=list_runtime_backends(),
        default=DEFAULTS.runtime_backend,
        help="Runtime backend whose execution defaults should be used.",
    )
    parser.add_argument(
        "--workspace",
        choices=SUPPORTED_WORKSPACES,
        default=DEFAULTS.workspace,
        help="Workspace backend to plan for.",
    )
    parser.add_argument("--max-iterations", type=int, default=DEFAULTS.max_iterations)
    parser.add_argument(
        "--concurrency",
        type=int,
        default=None,
        help="Maximum number of benchmark instances to execute concurrently. Defaults to SOMA_BENCHMARK_CONCURRENCY or 1.",
    )
    parser.add_argument("--output-dir", default=DEFAULTS.output_dir)
    parser.add_argument("--n-limit", type=int, default=0)
    parser.add_argument("--select", default=None, help="Optional file with one instance_id per line.")
    parser.add_argument("--image", default=None, help="Target agent image repository.")
    parser.add_argument("--target", default=None, help="Build target suffix.")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Execute planned instances with the selected runtime backend after scaffold generation.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the inferred run plan without creating any scaffold directories.",
    )
    parser.add_argument(
        "--openclaw-command",
        default=None,
        help="Optional extra arguments appended to `openclaw agent` inside the OpenClaw CLI sidecar.",
    )
    parser.add_argument(
        "--openclaw-container-image",
        default=None,
        help="Optional Docker image override for the shared OpenClaw gateway and CLI sidecars.",
    )
    parser.add_argument(
        "--openclaw-current-user",
        action="store_true",
        help=(
            "Run OpenClaw gateway and CLI containers as the invoking host uid:gid. "
            "Required unless provided via SOMA_OPENCLAW_USER or SOMA_OPENCLAW_CONTAINER_USER."
        ),
    )
    parser.add_argument(
        "--openclaw-ignore-api-key",
        action="store_true",
        help="Skip local LLM API key validation and do not forward API key credentials to OpenClaw. Use when the upstream gateway resolves auth from X-Run-Id.",
    )
    parser.add_argument(
        "--openclaw-plugin-path",
        default=None,
        help="Optional local path to the SOMA OpenClaw plugin repository.",
    )
    parser.add_argument(
        "--openclaw-disable-plugin",
        action="store_true",
        help="Disable automatic SOMA context-engine plugin setup for OpenClaw.",
    )
    parser.add_argument(
        "--openclaw-plugin-reinstall-on-run-start",
        action="store_true",
        help="Recreate the SOMA plugin Python environment once at the start of this benchmark process.",
    )
    parser.add_argument(
        "--swerebench-eval",
        action="store_true",
        help="After agent execution, export the resulting patch and run SWE-rebench evaluation through swebench.harness.",
    )
    parser.add_argument(
        "--swerebench-harness-root",
        default=None,
        help="Optional path to a checkout of the SWE-rebench SWE-bench fork used for patch evaluation.",
    )
    parser.add_argument(
        "--swerebench-harness-python",
        default=None,
        help="Optional Python executable used to launch swebench.harness.run_evaluation.",
    )
    parser.add_argument(
        "--swerebench-namespace",
        default=None,
        help="Optional swebench image namespace override. Use an empty string to force local image builds.",
    )
    parser.add_argument(
        "--swerebench-cache-level",
        choices=["none", "base", "env", "instance"],
        default=None,
        help="Optional cache level forwarded to swebench.harness.run_evaluation.",
    )
    parser.add_argument(
        "--swerebench-timeout",
        type=int,
        default=None,
        help="Optional per-instance evaluation timeout in seconds for swebench.harness.run_evaluation.",
    )
    parser.add_argument(
        "--swerebench-max-workers",
        type=int,
        default=None,
        help="Optional swebench evaluation worker count. Defaults to 1 when enabled.",
    )
    args = parser.parse_args(argv)

    if args.dry_run and args.execute:
        parser.error("--execute cannot be combined with --dry-run")

    repo_root = Path(__file__).resolve().parents[3]
    load_repo_env(repo_root)

    concurrency = resolve_benchmark_concurrency(args.concurrency)
    runtime_backend = get_runtime_backend(args.runtime_backend)
    image = args.image or runtime_backend.default_image
    target = args.target or runtime_backend.default_build_target
    llm_config = build_llm_config(None, ignore_api_key=args.openclaw_ignore_api_key)
    selected_ids = load_selected_instance_ids(args.select) if args.select else None
    instances = load_manifest(args.dataset, limit=args.n_limit, selected_instance_ids=selected_ids)
    runtime_options = _build_runtime_options(
        openclaw_command=args.openclaw_command,
        openclaw_container_image=args.openclaw_container_image,
        openclaw_current_user=args.openclaw_current_user,
        openclaw_run_id_header_value=None,
        openclaw_ignore_api_key=args.openclaw_ignore_api_key,
        openclaw_plugin_path=args.openclaw_plugin_path,
        openclaw_disable_plugin=args.openclaw_disable_plugin,
        openclaw_plugin_reinstall_on_run_start=args.openclaw_plugin_reinstall_on_run_start,
        swerebench_eval=args.swerebench_eval,
        swerebench_split=args.split,
        swerebench_harness_root=args.swerebench_harness_root,
        swerebench_harness_python=args.swerebench_harness_python,
        swerebench_namespace=args.swerebench_namespace,
        swerebench_cache_level=args.swerebench_cache_level,
        swerebench_timeout=args.swerebench_timeout,
        swerebench_max_workers=args.swerebench_max_workers,
    )
    run_plan = build_run_plan(
        instances,
        workspace=args.workspace,
        agent_image=image,
        build_target=target,
        runtime_options=runtime_options,
    )
    payload = {
        "dataset": str(Path(args.dataset)),
        "split": args.split,
        "runtime_backend": runtime_backend.name,
        "workspace": args.workspace,
        "max_iterations": args.max_iterations,
        "concurrency": concurrency,
        "instance_count": len(run_plan),
        "llm": {
            "model": llm_config.get("model"),
            "base_url": llm_config.get("base_url"),
        },
        "execution_mode": (
            "dry-run" if args.dry_run else "backend-execution" if args.execute else "scaffold-only"
        ),
        "runtime_options": runtime_options,
        "run_plan": run_plan,
    }
    if args.dry_run:
        print(json.dumps(payload, indent=2))
        return 0

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    instance_index = {instance.benchmark_id: instance for instance in instances}
    for item in run_plan:
        instance_dir = output_dir / "instances" / item["benchmark_id"]
        instance_dir.mkdir(parents=True, exist_ok=True)
        (instance_dir / "run-plan.json").write_text(json.dumps(item, indent=2) + "\n", encoding="utf-8")

    run_plan_path = output_dir / "run-plan.json"
    run_plan_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    output_json = default_output_path(output_dir)
    output_json.touch(exist_ok=True)

    if not args.execute:
        print(
            json.dumps(
                {
                    "status": "scaffolded",
                    "run_plan": str(run_plan_path),
                    "output_json": str(output_json),
                    "next_step": runtime_backend.execution_hint,
                },
                indent=2,
            )
        )
        return 0

    contexts: list[RuntimeExecutionContext] = []
    for item in run_plan:
        instance = instance_index[item["benchmark_id"]]
        instance_dir = output_dir / "instances" / item["benchmark_id"]
        contexts.append(
            RuntimeExecutionContext(
                backend_name=runtime_backend.name,
                instance=instance,
                run_payload=item,
                llm_config=llm_config,
                workspace=args.workspace,
                max_iterations=args.max_iterations,
                output_dir=output_dir,
                instance_dir=instance_dir,
            )
        )

    result_rows = [
        result.to_dict()
        for result in execute_runtime_contexts(
            runtime_backend,
            contexts,
            concurrency=concurrency,
        )
    ]
    for row in result_rows:
        _append_jsonl_row(output_json, row)

    print(
        json.dumps(
            {
                "status": (
                    "completed"
                    if all(row.get("status") != "runtime-error" for row in result_rows)
                    else "partial-failure"
                ),
                "run_plan": str(run_plan_path),
                "output_json": str(output_json),
                "result_count": len(result_rows),
                "next_step": runtime_backend.execution_hint,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
