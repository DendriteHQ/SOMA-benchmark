from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from .backends import get_runtime_backend, list_runtime_backends
from .backends.base import RuntimeExecutionContext
from .build_images import build_image_plan
from .config import default_output_path
from .execution import execute_runtime_contexts
from .manifest import load_manifest, write_jsonl
from .progress import emit_progress
from .run_infer import _build_runtime_options, build_run_plan
from .runner_settings import resolve_benchmark_concurrency, resolve_runner_config
from .swebench_images import enrich_hidden_eval_with_runtime_image


def _write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return output_path


def _append_jsonl_row(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def _load_batch_input(path: str | Path) -> tuple[str, list[dict[str, Any]]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Batch input must be a JSON object")
    batch_id = str(payload.get("batch_id") or "batch").strip() or "batch"
    tasks_raw = payload.get("tasks")
    if not isinstance(tasks_raw, list) or not tasks_raw:
        raise ValueError("Batch input must contain a non-empty 'tasks' list")
    tasks: list[dict[str, Any]] = []
    for raw in tasks_raw:
        if not isinstance(raw, dict):
            raise ValueError("Each batch task must be a JSON object")
        run_id = raw.get("run_id")
        benchmark = raw.get("benchmark")
        instance_id = raw.get("instance_id")
        if run_id is None or not str(benchmark or "").strip() or not str(instance_id or "").strip():
            raise ValueError("Each batch task requires run_id, benchmark, and instance_id")
        task = dict(raw)
        task["run_id"] = int(run_id)
        task["benchmark"] = str(benchmark).strip()
        task["instance_id"] = str(instance_id).strip()
        if not str(task.get("benchmark_id") or "").strip():
            task["benchmark_id"] = f"{task['instance_id']}-run-{task['run_id']}"
        tasks.append(task)
    return batch_id, tasks


def _build_problem_statement(runtime_setup_entry: dict[str, Any]) -> str:
    hidden_eval = runtime_setup_entry.get("hidden_eval")
    if not isinstance(hidden_eval, dict):
        hidden_eval = {}

    parts: list[str] = []
    problem_statement = str(hidden_eval.get("problem_statement", "")).strip()
    if problem_statement:
        parts.append(problem_statement)

    hints_text = str(hidden_eval.get("hints_text", "")).strip()
    if hints_text:
        parts.extend(["", "Hints:", hints_text])

    if parts:
        return "\n".join(parts)

    benchmark_name = str(hidden_eval.get("benchmark", "benchmark")).strip() or "benchmark"
    instance_id = str(runtime_setup_entry.get("instance_id", "")).strip() or "unknown-instance"
    return f"Solve benchmark instance {instance_id} from {benchmark_name}."


def build_direct_manifest_row(
    *,
    benchmark_name: str,
    instance_id: str,
    runtime_setup_entry: dict[str, Any],
    runtime_options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    prompt = _build_problem_statement(runtime_setup_entry)
    hidden_eval = runtime_setup_entry.get("hidden_eval")
    if not isinstance(hidden_eval, dict):
        hidden_eval = {}
    hidden_eval = enrich_hidden_eval_with_runtime_image(
        instance_id=instance_id,
        hidden_eval=hidden_eval,
        runtime_options=runtime_options,
    )

    return {
        "benchmark_id": instance_id,
        "trajectory_id": f"direct-{instance_id}",
        "instance_id": instance_id,
        "repo": str(runtime_setup_entry.get("repo", "")).strip(),
        "base_commit": str(runtime_setup_entry.get("base_commit", "")).strip(),
        "environment_setup_commit": str(runtime_setup_entry.get("environment_setup_commit", "")).strip(),
        "install_config": runtime_setup_entry.get("install_config")
        if isinstance(runtime_setup_entry.get("install_config"), dict)
        else {},
        "requirements": runtime_setup_entry.get("requirements")
        if isinstance(runtime_setup_entry.get("requirements"), list)
        else [],
        "environment": runtime_setup_entry.get("environment")
        if isinstance(runtime_setup_entry.get("environment"), dict)
        else {},
        "fail_to_pass": runtime_setup_entry.get("fail_to_pass")
        if isinstance(runtime_setup_entry.get("fail_to_pass"), list)
        else [],
        "pass_to_pass": runtime_setup_entry.get("pass_to_pass")
        if isinstance(runtime_setup_entry.get("pass_to_pass"), list)
        else [],
        "user_prompt": prompt,
        "stage_boundary": "problem-statement",
        "expected_next_stage": "solution",
        "expected_next_action": "Solve the benchmark task from the provided problem statement.",
        "resume_prompt": "",
        "hidden_eval": hidden_eval,
        "metadata": {
            "source_type": "benchmark-problem-statement",
            "benchmark": benchmark_name,
            "instance_id": instance_id,
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m soma_bench.benchmark.solve")
    parser.add_argument("--agent-name", required=True, help="Runtime backend name, for example 'openclaw'.")
    parser.add_argument(
        "--model",
        default=None,
        help="Optional LLM model override. Defaults to LLM_MODEL, OPENAI_MODEL, or OPENROUTER_MODEL from .env.",
    )
    parser.add_argument(
        "--benchmark",
        required=True,
        help="Benchmark dataset identifier, for example 'nebius/SWE-rebench' or 'SWE-bench/SWE-bench_Verified'.",
    )
    parser.add_argument("--instance-id", required=False, help="Concrete benchmark instance identifier.")
    parser.add_argument(
        "--batch-input-json",
        default=None,
        help="Optional path to batch input JSON produced by sandbox_service.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional explicit output directory. Defaults to the benchmark output root derived from .env and benchmark metadata.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Execute the selected runtime backend immediately instead of stopping after scaffold generation.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=None,
        help="Maximum number of benchmark instances to execute concurrently. Defaults to SOMA_BENCHMARK_CONCURRENCY or 1.",
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
        "--openclaw-run-id-header-value",
        default=None,
        help="Optional value sent as X-Run-Id on OpenClaw upstream model calls.",
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
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.batch_input_json:
        return _run_batch_mode(args)
    if not args.instance_id:
        parser.error("--instance-id is required when --batch-input-json is not provided")

    repo_root = Path(__file__).resolve().parents[3]
    emit_progress(
        f"resolving benchmark metadata for {args.benchmark} instance {args.instance_id}",
        component="benchmark-solve",
    )
    resolved = resolve_runner_config(
        repo_root=repo_root,
        agent_name=args.agent_name,
        model=args.model,
        benchmark_name=args.benchmark,
        selection_id=args.instance_id,
        ignore_api_key=args.openclaw_ignore_api_key,
    )

    if resolved.agent.runtime_backend not in list_runtime_backends():
        available = ", ".join(sorted(list_runtime_backends()))
        raise ValueError(
            f"Unsupported runtime backend resolved from agent '{args.agent_name}': "
            f"{resolved.agent.runtime_backend}. Available backends: {available}"
        )

    runtime_backend = get_runtime_backend(resolved.agent.runtime_backend)
    image = runtime_backend.default_image
    target = runtime_backend.default_build_target
    output_dir = Path(args.output_dir) if args.output_dir else resolved.benchmark.output_dir
    execute = args.execute or resolved.benchmark.execute
    concurrency = (
        resolve_benchmark_concurrency(args.concurrency)
        if args.concurrency is not None
        else resolved.benchmark.concurrency
    )
    benchmark_config = replace(
        resolved.benchmark,
        output_dir=output_dir,
        execute=execute,
        concurrency=concurrency,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    emit_progress(
        f"writing benchmark scaffold under {output_dir}",
        component="benchmark-solve",
    )
    manifest_output_file = output_dir / "benchmark-manifest.jsonl"
    image_plan_output_file = output_dir / "image-plan.json"
    eval_output_file = output_dir / "evaluation-summary.json"

    runtime_options = _build_runtime_options(
        openclaw_command=args.openclaw_command,
        openclaw_container_image=args.openclaw_container_image,
        openclaw_current_user=args.openclaw_current_user,
        openclaw_run_id_header_value=args.openclaw_run_id_header_value,
        openclaw_ignore_api_key=args.openclaw_ignore_api_key,
        openclaw_plugin_path=args.openclaw_plugin_path,
        openclaw_disable_plugin=args.openclaw_disable_plugin,
        openclaw_plugin_reinstall_on_run_start=args.openclaw_plugin_reinstall_on_run_start,
        swerebench_eval=args.swerebench_eval,
        swerebench_split=benchmark_config.split,
        swerebench_harness_root=args.swerebench_harness_root,
        swerebench_harness_python=args.swerebench_harness_python,
        swerebench_namespace=args.swerebench_namespace,
        swerebench_cache_level=args.swerebench_cache_level,
        swerebench_timeout=args.swerebench_timeout,
        swerebench_max_workers=args.swerebench_max_workers,
    )

    manifest_row = build_direct_manifest_row(
        benchmark_name=benchmark_config.benchmark_name,
        instance_id=args.instance_id,
        runtime_setup_entry=benchmark_config.runtime_setup_entry,
        runtime_options=runtime_options,
    )
    write_jsonl(manifest_output_file, [manifest_row])

    instances = load_manifest(manifest_output_file)
    emit_progress(
        f"loaded manifest with {len(instances)} instance row(s)",
        component="benchmark-solve",
    )
    image_plan = build_image_plan(instances, image=image, target=target)
    image_plan_payload = {
        "dataset": str(manifest_output_file),
        "split": benchmark_config.split,
        "dataset_config": benchmark_config.dataset_config,
        "runtime_backend": runtime_backend.name,
        "instance_count": len(image_plan),
        "image": image,
        "target": target,
        "planned_images": image_plan,
    }
    _write_json(image_plan_output_file, image_plan_payload)

    run_plan = build_run_plan(
        instances,
        workspace=benchmark_config.workspace,
        agent_image=image,
        build_target=target,
        runtime_options=runtime_options,
    )
    run_plan_payload = {
        "dataset": str(manifest_output_file),
        "split": benchmark_config.split,
        "dataset_config": benchmark_config.dataset_config,
        "runtime_backend": runtime_backend.name,
        "workspace": benchmark_config.workspace,
        "max_iterations": benchmark_config.max_iterations,
        "concurrency": benchmark_config.concurrency,
        "instance_count": len(run_plan),
        "llm": {
            "model": resolved.llm_config.get("model"),
            "base_url": resolved.llm_config.get("base_url"),
        },
        "execution_mode": "backend-execution" if benchmark_config.execute else "scaffold-only",
        "source": {
            "type": "benchmark-problem-statement",
            "benchmark": benchmark_config.benchmark_name,
            "instance_id": args.instance_id,
        },
        "runtime_options": runtime_options,
        "run_plan": run_plan,
    }

    instance_index = {instance.benchmark_id: instance for instance in instances}
    for item in run_plan:
        instance_dir = output_dir / "instances" / item["benchmark_id"]
        instance_dir.mkdir(parents=True, exist_ok=True)
        (instance_dir / "run-plan.json").write_text(json.dumps(item, indent=2) + "\n", encoding="utf-8")

    run_plan_path = output_dir / "run-plan.json"
    run_plan_path.write_text(json.dumps(run_plan_payload, indent=2) + "\n", encoding="utf-8")
    output_json = default_output_path(output_dir)
    output_json.write_text("", encoding="utf-8")

    result_rows: list[dict[str, Any]] = []
    evaluation_summary: dict[str, Any] | None = None
    if benchmark_config.execute:
        emit_progress(
            f"starting runtime execution with backend {runtime_backend.name}",
            component="benchmark-solve",
        )
        contexts: list[RuntimeExecutionContext] = []
        for item in run_plan:
            instance = instance_index[item["benchmark_id"]]
            instance_dir = output_dir / "instances" / item["benchmark_id"]
            emit_progress(
                f"preparing instance {instance.instance_id}",
                component="benchmark-solve",
            )
            contexts.append(
                RuntimeExecutionContext(
                    backend_name=runtime_backend.name,
                    instance=instance,
                    run_payload=item,
                    llm_config=resolved.llm_config,
                    workspace=benchmark_config.workspace,
                    max_iterations=benchmark_config.max_iterations,
                    output_dir=output_dir,
                    instance_dir=instance_dir,
                )
            )

        result_rows = [
            result.to_dict()
            for result in execute_runtime_contexts(
                runtime_backend,
                contexts,
                concurrency=benchmark_config.concurrency,
            )
        ]
        for row in result_rows:
            _append_jsonl_row(output_json, row)
            emit_progress(
                f"instance {row.get('instance_id', 'unknown')} finished with status {row.get('status', 'unknown')}",
                component="benchmark-solve",
            )

        from .eval_infer import summarize_results

        evaluation_summary = summarize_results(result_rows)
        _write_json(eval_output_file, evaluation_summary)
        emit_progress(
            f"wrote evaluation summary to {eval_output_file}",
            component="benchmark-solve",
        )

    payload = {
        "agent_name": args.agent_name,
        "model": resolved.llm_config.get("model"),
        "benchmark": benchmark_config.benchmark_name,
        "instance_id": args.instance_id,
        "dataset_config": benchmark_config.dataset_config,
        "split": benchmark_config.split,
        "runtime_backend": runtime_backend.name,
        "execution_mode": "backend-execution" if benchmark_config.execute else "scaffold-only",
        "concurrency": benchmark_config.concurrency,
        "source": {
            "type": "benchmark-problem-statement",
            "benchmark": benchmark_config.benchmark_name,
            "instance_id": args.instance_id,
        },
        "manifest_row_count": len(instances),
        "planned_image_count": len(image_plan),
        "run_plan_count": len(run_plan),
        "outputs": {
            "manifest": str(manifest_output_file),
            "image_plan": str(image_plan_output_file),
            "run_plan": str(run_plan_path),
            "output_json": str(output_json),
            "evaluation_summary": str(eval_output_file) if evaluation_summary is not None else None,
        },
    }
    print(json.dumps(payload, indent=2))
    return 0


def _run_batch_mode(args: argparse.Namespace) -> int:
    batch_id, batch_tasks = _load_batch_input(args.batch_input_json)
    repo_root = Path(__file__).resolve().parents[3]

    first_task = batch_tasks[0]
    emit_progress(
        f"resolving benchmark metadata for batch {batch_id} with {len(batch_tasks)} task(s)",
        component="benchmark-solve",
    )
    resolved_primary = resolve_runner_config(
        repo_root=repo_root,
        agent_name=args.agent_name,
        model=args.model,
        benchmark_name=str(first_task["benchmark"]),
        selection_id=str(first_task["instance_id"]),
        ignore_api_key=args.openclaw_ignore_api_key,
    )
    if resolved_primary.agent.runtime_backend not in list_runtime_backends():
        available = ", ".join(sorted(list_runtime_backends()))
        raise ValueError(
            f"Unsupported runtime backend resolved from agent '{args.agent_name}': "
            f"{resolved_primary.agent.runtime_backend}. Available backends: {available}"
        )

    runtime_backend = get_runtime_backend(resolved_primary.agent.runtime_backend)
    image = runtime_backend.default_image
    target = runtime_backend.default_build_target
    output_dir = Path(args.output_dir) if args.output_dir else resolved_primary.benchmark.output_dir
    execute = args.execute or resolved_primary.benchmark.execute
    concurrency = (
        resolve_benchmark_concurrency(args.concurrency)
        if args.concurrency is not None
        else resolved_primary.benchmark.concurrency
    )
    benchmark_config = replace(
        resolved_primary.benchmark,
        output_dir=output_dir,
        execute=execute,
        concurrency=concurrency,
    )

    runtime_options_common = _build_runtime_options(
        openclaw_command=args.openclaw_command,
        openclaw_container_image=args.openclaw_container_image,
        openclaw_current_user=args.openclaw_current_user,
        openclaw_run_id_header_value=args.openclaw_run_id_header_value,
        openclaw_ignore_api_key=args.openclaw_ignore_api_key,
        openclaw_plugin_path=args.openclaw_plugin_path,
        openclaw_disable_plugin=args.openclaw_disable_plugin,
        openclaw_plugin_reinstall_on_run_start=args.openclaw_plugin_reinstall_on_run_start,
        swerebench_eval=args.swerebench_eval,
        swerebench_split=benchmark_config.split,
        swerebench_harness_root=args.swerebench_harness_root,
        swerebench_harness_python=args.swerebench_harness_python,
        swerebench_namespace=args.swerebench_namespace,
        swerebench_cache_level=args.swerebench_cache_level,
        swerebench_timeout=args.swerebench_timeout,
        swerebench_max_workers=args.swerebench_max_workers,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_output_file = output_dir / "benchmark-manifest.jsonl"
    image_plan_output_file = output_dir / "image-plan.json"
    eval_output_file = output_dir / "evaluation-summary.json"

    manifest_rows: list[dict[str, Any]] = []
    runtime_options_by_benchmark_id: dict[str, dict[str, Any]] = {}
    benchmark_to_run_id: dict[str, int] = {}
    for task in batch_tasks:
        resolved_task = resolve_runner_config(
            repo_root=repo_root,
            agent_name=args.agent_name,
            model=args.model,
            benchmark_name=str(task["benchmark"]),
            selection_id=str(task["instance_id"]),
            ignore_api_key=args.openclaw_ignore_api_key,
        )
        benchmark_id = str(task["benchmark_id"])
        task_runtime_options = dict(runtime_options_common)
        runtime_override = task.get("runtime_options")
        if isinstance(runtime_override, dict):
            task_runtime_options.update(runtime_override)

        row = build_direct_manifest_row(
            benchmark_name=str(task["benchmark"]),
            instance_id=str(task["instance_id"]),
            runtime_setup_entry=resolved_task.benchmark.runtime_setup_entry,
            runtime_options=task_runtime_options,
        )
        row["benchmark_id"] = benchmark_id
        metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        metadata.update(
            {
                "batch_id": batch_id,
                "batch_run_id": int(task["run_id"]),
            }
        )
        row["metadata"] = metadata
        manifest_rows.append(row)
        runtime_options_by_benchmark_id[benchmark_id] = task_runtime_options
        benchmark_to_run_id[benchmark_id] = int(task["run_id"])

    write_jsonl(manifest_output_file, manifest_rows)
    instances = load_manifest(manifest_output_file)
    image_plan = build_image_plan(instances, image=image, target=target)
    _write_json(
        image_plan_output_file,
        {
            "dataset": str(manifest_output_file),
            "split": benchmark_config.split,
            "dataset_config": benchmark_config.dataset_config,
            "runtime_backend": runtime_backend.name,
            "instance_count": len(image_plan),
            "image": image,
            "target": target,
            "planned_images": image_plan,
            "batch_id": batch_id,
        },
    )

    run_plan = build_run_plan(
        instances,
        workspace=benchmark_config.workspace,
        agent_image=image,
        build_target=target,
        runtime_options=runtime_options_common,
    )
    for item in run_plan:
        benchmark_id = str(item["benchmark_id"])
        item["runtime_options"] = dict(runtime_options_by_benchmark_id.get(benchmark_id, runtime_options_common))

    run_plan_path = output_dir / "run-plan.json"
    run_plan_path.write_text(
        json.dumps(
            {
                "dataset": str(manifest_output_file),
                "split": benchmark_config.split,
                "dataset_config": benchmark_config.dataset_config,
                "runtime_backend": runtime_backend.name,
                "workspace": benchmark_config.workspace,
                "max_iterations": benchmark_config.max_iterations,
                "concurrency": benchmark_config.concurrency,
                "instance_count": len(run_plan),
                "llm": {
                    "model": resolved_primary.llm_config.get("model"),
                    "base_url": resolved_primary.llm_config.get("base_url"),
                },
                "execution_mode": "backend-execution" if benchmark_config.execute else "scaffold-only",
                "source": {
                    "type": "batch-problem-statement",
                    "batch_id": batch_id,
                },
                "runtime_options": runtime_options_common,
                "run_plan": run_plan,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    output_json = default_output_path(output_dir)
    output_json.write_text("", encoding="utf-8")

    result_rows: list[dict[str, Any]] = []
    evaluation_summary: dict[str, Any] | None = None
    if benchmark_config.execute:
        contexts: list[RuntimeExecutionContext] = []
        instance_index = {instance.benchmark_id: instance for instance in instances}
        for item in run_plan:
            instance = instance_index[item["benchmark_id"]]
            instance_dir = output_dir / "instances" / item["benchmark_id"]
            instance_dir.mkdir(parents=True, exist_ok=True)
            contexts.append(
                RuntimeExecutionContext(
                    backend_name=runtime_backend.name,
                    instance=instance,
                    run_payload=item,
                    llm_config=resolved_primary.llm_config,
                    workspace=benchmark_config.workspace,
                    max_iterations=benchmark_config.max_iterations,
                    output_dir=output_dir,
                    instance_dir=instance_dir,
                )
            )

        result_rows = [
            result.to_dict()
            for result in execute_runtime_contexts(
                runtime_backend,
                contexts,
                concurrency=benchmark_config.concurrency,
            )
        ]
        for row in result_rows:
            benchmark_id = str(row.get("benchmark_id") or "")
            run_id = benchmark_to_run_id.get(benchmark_id)
            if run_id is not None:
                metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
                metadata["batch_run_id"] = run_id
                row["metadata"] = metadata
            _append_jsonl_row(output_json, row)

        from .eval_infer import summarize_results

        evaluation_summary = summarize_results(result_rows)
        _write_json(eval_output_file, evaluation_summary)

    print(
        json.dumps(
            {
                "agent_name": args.agent_name,
                "model": resolved_primary.llm_config.get("model"),
                "batch_id": batch_id,
                "task_count": len(batch_tasks),
                "runtime_backend": runtime_backend.name,
                "execution_mode": "backend-execution" if benchmark_config.execute else "scaffold-only",
                "concurrency": benchmark_config.concurrency,
                "outputs": {
                    "manifest": str(manifest_output_file),
                    "image_plan": str(image_plan_output_file),
                    "run_plan": str(run_plan_path),
                    "output_json": str(output_json),
                    "evaluation_summary": str(eval_output_file) if evaluation_summary is not None else None,
                },
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
