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


def add_harness_arguments(parser: argparse.ArgumentParser) -> None:
    """Register CLI flags shared by every agent harness (run-infer and solve).

    Flag names are harness-neutral: every harness runs on the same runner and reads them the same way.
    """
    parser.add_argument(
        "--run-id-header-value",
        dest="run_id_header_value",
        default=None,
        help=(
            "Optional run id forwarded to the upstream LLM gateway in place of the provider API key. "
            "The gateway resolves authorization for the run server-side."
        ),
    )
    parser.add_argument(
        "--ignore-api-key",
        dest="ignore_api_key",
        action="store_true",
        help=(
            "Skip local LLM API key validation. Use with --run-id-header-value when the upstream "
            "gateway resolves authorization server-side."
        ),
    )
    parser.add_argument(
        "--compression-script-path",
        dest="compression_script_path",
        default=None,
        help=(
            "Optional path to compressor script mounted into compression-service as /app/miner/base_miner.py. "
            "If omitted, the harness falls back to src/compression_service/app/base_miner.py."
        ),
    )
    parser.add_argument(
        "--compression-service-autobuild",
        dest="compression_service_autobuild",
        action="store_true",
        help=(
            "Build compression-service image before each run when compression-service is enabled. "
            "Uses src/compression_service by default."
        ),
    )
    parser.add_argument(
        "--compression-service-build-context",
        dest="compression_service_build_context",
        default=None,
        help=(
            "Optional docker build context for compression-service autobuild. "
            "Defaults to src/compression_service."
        ),
    )
    parser.add_argument(
        "--use-host-docker-socket",
        dest="use_host_docker_socket",
        action="store_true",
        help=(
            "Legacy/insecure fallback: bridge the agent to the SWE sandbox by mounting "
            "the HOST Docker socket directly, instead of the default isolated dind daemon. "
            "The agent can then see and control every container on the host, not just its own "
            "sandbox - only use this if the isolated dind daemon does not work in your environment "
            "(e.g. nested privileged containers are unsupported)."
        ),
    )
    parser.add_argument(
        "--claude-code-permission-mode",
        default=None,
        help=(
            "Claude Code permission mode (default bypassPermissions - a benchmark run has "
            "nobody to answer a prompt)."
        ),
    )
    parser.add_argument(
        "--claude-code-max-turns",
        type=int,
        default=None,
        help="Turn ceiling for Claude Code. Unlimited by default, like Copilot.",
    )
    parser.add_argument(
        "--claude-code-disallowed-tools",
        default=None,
        help=(
            "Comma-separated tools to deny Claude Code, replacing the backend's default "
            "deny list (Workflow, Cron*, and other session-management tools). Pass an "
            "empty string to allow all of them."
        ),
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


def build_runtime_options(args: argparse.Namespace, *, swerebench_split: str | None) -> dict[str, Any]:
    """Translate the flags registered by `add_harness_arguments` into run-plan runtime options."""
    runtime_options: dict[str, Any] = {}
    if args.run_id_header_value:
        runtime_options["run_id_header_value"] = args.run_id_header_value
    if args.swerebench_eval:
        runtime_options["swerebench_eval_enabled"] = True
    if swerebench_split:
        runtime_options["swerebench_split"] = swerebench_split
    if args.swerebench_harness_root:
        runtime_options["swerebench_harness_root"] = args.swerebench_harness_root
    if args.swerebench_harness_python:
        runtime_options["swerebench_harness_python"] = args.swerebench_harness_python
    if args.swerebench_namespace is not None:
        runtime_options["swerebench_namespace"] = args.swerebench_namespace
    if args.swerebench_cache_level:
        runtime_options["swerebench_cache_level"] = args.swerebench_cache_level
    if args.swerebench_timeout is not None:
        runtime_options["swerebench_timeout"] = args.swerebench_timeout
    if args.swerebench_max_workers is not None:
        runtime_options["swerebench_max_workers"] = args.swerebench_max_workers

    if args.compression_script_path:
        runtime_options["compression_script_path"] = args.compression_script_path
    if args.compression_service_autobuild:
        runtime_options["compression_service_autobuild"] = True
    if args.compression_service_build_context:
        runtime_options["compression_service_build_context"] = args.compression_service_build_context
    if args.use_host_docker_socket:
        runtime_options["use_host_docker_socket"] = True

    if args.claude_code_permission_mode:
        runtime_options["claude_code_permission_mode"] = args.claude_code_permission_mode
    if args.claude_code_max_turns is not None:
        runtime_options["claude_code_max_turns"] = args.claude_code_max_turns
    # Not `if ...:` - an empty string is the documented way to clear the backend's own
    # default deny list, and is distinct from the flag being absent.
    if args.claude_code_disallowed_tools is not None:
        runtime_options["claude_code_disallowed_tools"] = args.claude_code_disallowed_tools
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
    add_harness_arguments(parser)
    args = parser.parse_args(argv)

    if args.dry_run and args.execute:
        parser.error("--execute cannot be combined with --dry-run")

    repo_root = Path(__file__).resolve().parents[3]
    load_repo_env(repo_root)

    concurrency = resolve_benchmark_concurrency(args.concurrency)
    runtime_backend = get_runtime_backend(args.runtime_backend)
    image = args.image or runtime_backend.default_image
    target = args.target or runtime_backend.default_build_target
    llm_config = build_llm_config(None, ignore_api_key=args.ignore_api_key)
    selected_ids = load_selected_instance_ids(args.select) if args.select else None
    instances = load_manifest(args.dataset, limit=args.n_limit, selected_instance_ids=selected_ids)
    runtime_options = build_runtime_options(args, swerebench_split=args.split)
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
