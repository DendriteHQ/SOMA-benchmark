from __future__ import annotations

import copy
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

from .cache_paths import benchmark_cache_root
from .progress import emit_progress

DEFAULT_CACHE_LEVEL = "instance"
DEFAULT_MAX_WORKERS = 1
DEFAULT_TIMEOUT_SECONDS = 1_800
OPENCLAW_PATCH_EXCLUDE_PATHS = (
    ".openclaw",
    "AGENTS.md",
    "BOOTSTRAP.md",
    "HEARTBEAT.md",
    "IDENTITY.md",
    "SOUL.md",
    "TOOLS.md",
    "USER.md",
    "OPENCLAW_BENCHMARK_ENV.sh",
    "issue.md",
)


def _run_command(
    args: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        text=True,
        capture_output=True,
        check=False,
        cwd=str(cwd) if cwd is not None else None,
        env=env,
    )


def _run_command_streaming(
    args: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    component: str,
) -> subprocess.CompletedProcess[str]:
    process = subprocess.Popen(
        args,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        cwd=str(cwd) if cwd is not None else None,
        env=env,
    )
    captured_lines: list[str] = []
    assert process.stdout is not None
    for raw_line in process.stdout:
        line = raw_line.rstrip("\n")
        captured_lines.append(raw_line)
        if line.strip():
            emit_progress(line, component=component)
    returncode = process.wait()
    return subprocess.CompletedProcess(
        args=args,
        returncode=returncode,
        stdout="".join(captured_lines),
        stderr="",
    )


def _slug(value: str, *, default: str) -> str:
    normalized = re.sub(r"[^a-z0-9_.-]+", "-", value.lower()).strip("-._")
    return normalized or default


def _coerce_bool(raw_value: Any) -> bool | None:
    if isinstance(raw_value, bool):
        return raw_value
    if isinstance(raw_value, str):
        normalized = raw_value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    return None


def _coerce_positive_int(raw_value: Any, default: int) -> int:
    if raw_value is None:
        return default
    try:
        value = int(raw_value)
    except (TypeError, ValueError):
        return default
    return value if value > 0 else default


def _resolve_option(runtime_options: dict[str, Any], name: str, env_name: str) -> Any:
    if name in runtime_options:
        return runtime_options[name]
    return os.getenv(env_name)


def _model_label(model_name: str) -> str:
    return model_name.replace("/", "__") or "unknown-model"


def _benchmark_cache_root() -> Path:
    return benchmark_cache_root()


def _cache_slug(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-._")
    return normalized or "default"


def _extract_cached_row(row_wrapper: Any) -> dict[str, Any] | None:
    if not isinstance(row_wrapper, dict):
        return None
    row = row_wrapper.get("row")
    if isinstance(row, dict):
        return row
    if isinstance(row_wrapper.get("instance_id"), str):
        return row_wrapper
    return None


def _test_command_uses_pytest(test_cmd: Any) -> bool:
    if isinstance(test_cmd, str):
        return re.search(r"(^|[;&|]\s*)pytest(?=\s|$)", test_cmd.strip()) is not None
    if isinstance(test_cmd, list):
        return any(_test_command_uses_pytest(item) for item in test_cmd)
    return False


def _with_testbed_pythonpath_for_pytest(row: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    normalized_row = copy.deepcopy(row)
    install_config = normalized_row.get("install_config")
    if not isinstance(install_config, dict):
        return normalized_row, False

    if not _test_command_uses_pytest(install_config.get("test_cmd")):
        return normalized_row, False

    export_command = 'export PYTHONPATH="/testbed${PYTHONPATH:+:$PYTHONPATH}"'
    eval_commands = install_config.get("eval_commands")
    if eval_commands is None:
        eval_commands_list: list[str] = []
    elif isinstance(eval_commands, list):
        eval_commands_list = [str(command) for command in eval_commands]
    else:
        eval_commands_list = [str(eval_commands)]

    if not any("PYTHONPATH" in command and "/testbed" in command for command in eval_commands_list):
        eval_commands_list.append(export_command)
        install_config["eval_commands"] = eval_commands_list
        return normalized_row, True

    install_config["eval_commands"] = eval_commands_list
    return normalized_row, False


def _write_local_eval_dataset_from_cache(
    *,
    instance_id: str,
    benchmark_name: str,
    split: str,
    output_dir: Path,
) -> Path | None:
    splits_root = _benchmark_cache_root() / _cache_slug(benchmark_name) / "splits"
    if not splits_root.is_dir():
        return None

    split_dir_name = _cache_slug(split)
    for config_dir in sorted(path for path in splits_root.iterdir() if path.is_dir()):
        rows_path = config_dir / split_dir_name / "rows.jsonl"
        if not rows_path.is_file():
            continue

        with rows_path.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    row_wrapper = json.loads(line)
                except json.JSONDecodeError:
                    continue
                row = _extract_cached_row(row_wrapper)
                if not isinstance(row, dict):
                    continue
                candidate = row.get("instance_id")
                if isinstance(candidate, str) and candidate.strip() == instance_id:
                    dataset_path = output_dir / "local-eval-dataset.jsonl"
                    eval_row, _ = _with_testbed_pythonpath_for_pytest(row)
                    dataset_path.write_text(json.dumps(eval_row, ensure_ascii=True) + "\n", encoding="utf-8")
                    return dataset_path

    return None


def _report_summary(report_payload: dict[str, Any]) -> dict[str, Any]:
    tests_status = report_payload.get("tests_status")
    summary: dict[str, Any] = {
        "resolved": bool(report_payload.get("resolved", False)),
        "patch_exists": bool(report_payload.get("patch_exists", False)),
        "patch_successfully_applied": bool(report_payload.get("patch_successfully_applied", False)),
    }
    if not isinstance(tests_status, dict):
        return summary

    for key in ("FAIL_TO_PASS", "PASS_TO_PASS", "FAIL_TO_FAIL", "PASS_TO_FAIL"):
        bucket = tests_status.get(key)
        if not isinstance(bucket, dict):
            continue
        success = bucket.get("success") if isinstance(bucket.get("success"), list) else []
        failure = bucket.get("failure") if isinstance(bucket.get("failure"), list) else []
        summary[key.lower()] = {
            "success": len(success),
            "failure": len(failure),
            "total": len(success) + len(failure),
            "successful_tests": success,
            "failed_tests": failure,
        }
    return summary


def capture_repo_patch(
    *,
    repo_dir: Path,
    output_dir: Path,
    base_ref: str | None = None,
) -> dict[str, Any]:
    """Diff `repo_dir` against `base_ref` (defaults to HEAD).

    Passing the instance's actual base_commit (instead of relying on HEAD) is
    required whenever the agent might run `git commit` itself — a coding CLI
    that commits its own work leaves the working tree clean relative to HEAD,
    which would otherwise make a real, committed change look like no change
    at all.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    patch_path = output_dir / "agent.patch"
    diff_ref = base_ref or "HEAD"
    payload: dict[str, Any] = {
        "status": "skipped-no-git",
        "repo_dir": str(repo_dir),
        "patch_path": str(patch_path),
        "has_changes": False,
        "line_count": 0,
        "size_bytes": 0,
    }
    if not (repo_dir / ".git").is_dir():
        patch_path.write_text("", encoding="utf-8")
        return payload

    add_result = _run_command(["git", "-C", str(repo_dir), "add", "-N", "--all"])
    if add_result.returncode != 0:
        payload["status"] = "error"
        payload["error"] = (add_result.stderr or add_result.stdout or "git add -N failed").strip()
        patch_path.write_text("", encoding="utf-8")
        return payload

    diff_result = _run_command(
        [
            "git",
            "-C",
            str(repo_dir),
            "-c",
            "core.fileMode=false",
            "diff",
            "--binary",
            "--full-index",
            "--no-ext-diff",
            diff_ref,
            "--",
            ".",
            *(f":(exclude){path}" for path in OPENCLAW_PATCH_EXCLUDE_PATHS),
        ]
    )
    patch_text = diff_result.stdout or ""
    patch_path.write_text(patch_text, encoding="utf-8")
    payload.update(
        {
            "status": "captured" if diff_result.returncode == 0 else "error",
            "has_changes": bool(patch_text.strip()),
            "line_count": len(patch_text.splitlines()),
            "size_bytes": len(patch_text.encode("utf-8")),
        }
    )
    if diff_result.returncode != 0:
        payload["error"] = (diff_result.stderr or diff_result.stdout or "git diff failed").strip()
    return payload


def _load_cached_row(instance_id: str, benchmark_name: str, split: str) -> dict[str, Any] | None:
    """Return the raw dataset row for ``instance_id`` from the local benchmark cache."""
    splits_root = _benchmark_cache_root() / _cache_slug(benchmark_name) / "splits"
    if not splits_root.is_dir():
        return None
    split_dir_name = _cache_slug(split)
    for config_dir in sorted(path for path in splits_root.iterdir() if path.is_dir()):
        rows_path = config_dir / split_dir_name / "rows.jsonl"
        if not rows_path.is_file():
            continue
        with rows_path.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    wrapper = json.loads(line)
                except json.JSONDecodeError:
                    continue
                row = _extract_cached_row(wrapper)
                if isinstance(row, dict) and str(row.get("instance_id", "")).strip() == instance_id:
                    return row
    return None


def _v2_report_summary(spec: dict[str, Any], item: dict[str, Any]) -> dict[str, Any]:
    """Normalize an official SWE-rebench-V2 report item into the SOMA summary shape."""
    fail_to_pass_expected = spec.get("FAIL_TO_PASS") or []
    pass_to_pass_expected = spec.get("PASS_TO_PASS") or []
    from_fail_to_pass = list(item.get("from_fail_to_pass") or [])
    failed_from_pass_to_pass = list(item.get("failed_from_pass_to_pass") or [])
    p2p_total = len(pass_to_pass_expected)
    p2p_failure = len(failed_from_pass_to_pass)
    return {
        "resolved": bool(item.get("passed_match", False)),
        "fail_to_pass": {
            "success": len(from_fail_to_pass),
            "total": len(fail_to_pass_expected),
            "success_tests": sorted(from_fail_to_pass),
        },
        "pass_to_pass": {
            "success": max(p2p_total - p2p_failure, 0),
            "total": p2p_total,
            "failure_tests": sorted(failed_from_pass_to_pass),
        },
        "exit_code": item.get("exit_code"),
        "log_path": item.get("log_path"),
    }


def run_swerebench_v2_evaluation(
    *,
    instance_id: str,
    benchmark_name: str,
    split: str,
    model_name: str,
    patch_capture: dict[str, Any],
    output_dir: Path,
    v2_root: Path,
    runtime_options: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate a captured patch with the official SWE-rebench-V2 ``scripts/eval.py``.

    Unlike the SWE-bench harness path, this evaluator natively understands the
    V2 instance schema (prebuilt ``image_name``, ``install_config.log_parser``,
    no conda ``python`` spec) and ships all V2 log parsers.
    """
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    base_result: dict[str, Any] = {
        "evaluator": "swe-rebench-v2",
        "dataset_name": benchmark_name,
        "split": split,
        "v2_root": str(v2_root),
    }

    eval_script = v2_root / "scripts" / "eval.py"
    if not eval_script.is_file():
        return {**base_result, "status": "unavailable", "reason": f"eval.py not found under {v2_root}"}

    spec = _load_cached_row(instance_id, benchmark_name, split)
    if spec is None:
        return {
            **base_result,
            "status": "unavailable",
            "reason": f"instance {instance_id} not found in local benchmark cache",
        }

    try:
        patch_text = Path(str(patch_capture["patch_path"])).read_text(encoding="utf-8")
    except (KeyError, OSError) as exc:
        return {**base_result, "status": "error", "reason": f"could not read captured patch: {exc}"}

    specs_path = output_dir / "v2-instances.json"
    patches_path = output_dir / "v2-patches.json"
    report_path = output_dir / "v2-eval-report.json"
    specs_path.write_text(json.dumps([spec], ensure_ascii=True), encoding="utf-8")
    patches_path.write_text(
        json.dumps([{"instance_id": instance_id, "patch": patch_text}], ensure_ascii=True),
        encoding="utf-8",
    )

    harness_python_value = _resolve_option(
        runtime_options, "swerebench_harness_python", "SOMA_SWEREBENCH_HARNESS_PYTHON"
    )
    harness_python = str(harness_python_value).strip() if harness_python_value else sys.executable
    max_workers = _coerce_positive_int(
        _resolve_option(runtime_options, "swerebench_max_workers", "SOMA_SWEREBENCH_MAX_WORKERS"),
        DEFAULT_MAX_WORKERS,
    )

    command = [
        harness_python,
        str(eval_script),
        "--json",
        str(specs_path),
        "--patches",
        str(patches_path),
        "--instance-ids",
        instance_id,
        "--report-json",
        str(report_path),
        "--max-workers",
        str(max_workers),
    ]
    emit_progress(
        f"[{instance_id}] launching SWE-rebench-V2 evaluator ({eval_script})",
        component="swerebench-eval",
    )
    result = _run_command_streaming(command, cwd=output_dir, env=os.environ.copy(), component="swerebench-eval")
    (output_dir / "v2-eval.stdout.log").write_text(result.stdout or "", encoding="utf-8")

    payload: dict[str, Any] = {
        **base_result,
        "status": "error",
        "work_dir": str(output_dir),
        "command": command,
        "exit_code": result.returncode,
        "report_path": str(report_path),
        "instances_path": str(specs_path),
        "patches_path": str(patches_path),
    }

    item: dict[str, Any] | None = None
    if report_path.is_file():
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            report = None
        items = report.get("items") if isinstance(report, dict) else None
        if isinstance(items, list):
            for entry in items:
                if isinstance(entry, dict) and entry.get("instance_id") == instance_id:
                    item = entry
                    break

    if item is not None:
        payload["report"] = item
        item_error = str(item.get("error") or "").strip()
        if item_error:
            payload["status"] = "error"
            payload["error"] = item_error
        else:
            # A non-resolving patch is a *completed* eval, not a failure; eval.py
            # only exits non-zero to signal "not all_ok", which we read from the report.
            payload["status"] = "completed"
            payload["summary"] = _v2_report_summary(spec, item)
            payload["resolved"] = payload["summary"]["resolved"]
    else:
        payload["error"] = (result.stdout or "swe-rebench-v2 eval produced no report").strip()[-2000:]

    outcome = "resolved" if payload.get("resolved") else payload["status"]
    emit_progress(
        f"[{instance_id}] SWE-rebench-V2 evaluation: {outcome}",
        component="swerebench-eval",
    )
    return payload


def maybe_run_swerebench_evaluation(
    *,
    instance_id: str,
    benchmark_name: str,
    split: str,
    model_name: str,
    patch_capture: dict[str, Any],
    output_dir: Path,
    runtime_options: dict[str, Any],
) -> dict[str, Any]:
    output_dir = output_dir.expanduser().resolve()

    enabled = _coerce_bool(
        _resolve_option(runtime_options, "swerebench_eval_enabled", "SOMA_SWEREBENCH_EVAL")
    )
    if enabled is not True:
        return {
            "status": "disabled",
            "dataset_name": benchmark_name,
            "split": split,
            "reason": "swerebench evaluation is disabled",
        }

    if not patch_capture.get("has_changes"):
        return {
            "status": "skipped-empty-patch",
            "dataset_name": benchmark_name,
            "split": split,
            "reason": "agent patch is empty",
            "patch_path": patch_capture.get("patch_path"),
        }

    # Route SWE-rebench-V2 instances to the official evaluator (scripts/eval.py),
    # which natively supports the V2 schema and all V2 log parsers. The SWE-bench
    # harness path below cannot evaluate V2 (it assumes the V1 install_config and
    # is missing most V2 parsers). Selection is automatic by dataset name, or
    # forced via swerebench_evaluator / SOMA_SWEREBENCH_EVALUATOR = "v2" | "harness".
    evaluator_value = _resolve_option(runtime_options, "swerebench_evaluator", "SOMA_SWEREBENCH_EVALUATOR")
    evaluator = str(evaluator_value).strip().lower() if isinstance(evaluator_value, str) else ""
    is_v2_dataset = "swe-rebench-v2" in str(benchmark_name).strip().lower()
    if evaluator == "v2" or (evaluator != "harness" and is_v2_dataset):
        v2_root_value = _resolve_option(runtime_options, "swerebench_v2_root", "SOMA_SWEREBENCH_V2_ROOT")
        if not (isinstance(v2_root_value, str) and v2_root_value.strip()):
            return {
                "status": "unavailable",
                "evaluator": "swe-rebench-v2",
                "dataset_name": benchmark_name,
                "split": split,
                "reason": "SWE-rebench-V2 evaluator selected but swerebench_v2_root / SOMA_SWEREBENCH_V2_ROOT is not set",
            }
        v2_root = Path(v2_root_value).expanduser().resolve()
        if not v2_root.is_dir():
            return {
                "status": "unavailable",
                "evaluator": "swe-rebench-v2",
                "dataset_name": benchmark_name,
                "split": split,
                "reason": f"configured swerebench_v2_root does not exist: {v2_root}",
            }
        return run_swerebench_v2_evaluation(
            instance_id=instance_id,
            benchmark_name=benchmark_name,
            split=split,
            model_name=model_name,
            patch_capture=patch_capture,
            output_dir=output_dir,
            v2_root=v2_root,
            runtime_options=runtime_options,
        )

    harness_root_value = _resolve_option(runtime_options, "swerebench_harness_root", "SOMA_SWEREBENCH_HARNESS_ROOT")
    harness_root = None
    if isinstance(harness_root_value, str) and harness_root_value.strip():
        harness_root = Path(harness_root_value).expanduser().resolve()
        if not harness_root.is_dir():
            return {
                "status": "unavailable",
                "dataset_name": benchmark_name,
                "split": split,
                "reason": f"configured harness root does not exist: {harness_root}",
            }

    harness_python_value = _resolve_option(
        runtime_options,
        "swerebench_harness_python",
        "SOMA_SWEREBENCH_HARNESS_PYTHON",
    )
    harness_python = str(harness_python_value).strip() if harness_python_value else sys.executable

    namespace_value = _resolve_option(runtime_options, "swerebench_namespace", "SOMA_SWEREBENCH_NAMESPACE")
    if isinstance(namespace_value, str):
        namespace = namespace_value
    elif str(benchmark_name).strip().lower().endswith("swe-rebench-leaderboard"):
        namespace = "swerebench"
    else:
        namespace = ""

    cache_level_value = _resolve_option(runtime_options, "swerebench_cache_level", "SOMA_SWEREBENCH_CACHE_LEVEL")
    cache_level = str(cache_level_value).strip() if cache_level_value else DEFAULT_CACHE_LEVEL

    timeout = _coerce_positive_int(
        _resolve_option(runtime_options, "swerebench_timeout", "SOMA_SWEREBENCH_TIMEOUT"),
        DEFAULT_TIMEOUT_SECONDS,
    )
    max_workers = _coerce_positive_int(
        _resolve_option(runtime_options, "swerebench_max_workers", "SOMA_SWEREBENCH_MAX_WORKERS"),
        DEFAULT_MAX_WORKERS,
    )
    run_id_prefix_value = _resolve_option(
        runtime_options,
        "swerebench_run_id_prefix",
        "SOMA_SWEREBENCH_RUN_ID_PREFIX",
    )
    run_id_prefix = str(run_id_prefix_value).strip() if run_id_prefix_value else "soma-openclaw"
    run_id = f"{_slug(run_id_prefix, default='soma-openclaw')}-{_slug(instance_id, default='instance')}"

    output_dir.mkdir(parents=True, exist_ok=True)
    emit_progress(
        f"[{instance_id}] preparing SWE-rebench evaluation in {output_dir}",
        component="swerebench-eval",
    )
    local_dataset_path = _write_local_eval_dataset_from_cache(
        instance_id=instance_id,
        benchmark_name=benchmark_name,
        split=split,
        output_dir=output_dir,
    )
    dataset_name_for_harness = str(local_dataset_path) if local_dataset_path is not None else benchmark_name
    if local_dataset_path is not None:
        emit_progress(
            f"[{instance_id}] using local benchmark cache {local_dataset_path} for harness dataset input",
            component="swerebench-eval",
        )
    else:
        emit_progress(
            f"[{instance_id}] local benchmark cache entry not found; falling back to dataset name {benchmark_name}",
            component="swerebench-eval",
        )

    predictions_path = output_dir / "predictions.jsonl"
    prediction_row = {
        "instance_id": instance_id,
        "model_name_or_path": model_name,
        "model_patch": Path(str(patch_capture["patch_path"])) .read_text(encoding="utf-8"),
    }
    predictions_path.write_text(json.dumps(prediction_row, ensure_ascii=True) + "\n", encoding="utf-8")

    stdout_path = output_dir / "harness.stdout.log"
    stderr_path = output_dir / "harness.stderr.log"
    env = os.environ.copy()
    if harness_root is not None:
        existing_pythonpath = env.get("PYTHONPATH", "")
        pythonpath_parts = [str(harness_root)]
        if existing_pythonpath:
            pythonpath_parts.append(existing_pythonpath)
        env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)

    command = [
        harness_python,
        "-m",
        "swebench.harness.run_evaluation",
        "--dataset_name",
        dataset_name_for_harness,
        "--split",
        split,
        "--predictions_path",
        str(predictions_path),
        "--instance_ids",
        instance_id,
        "--max_workers",
        str(max_workers),
        "--cache_level",
        cache_level,
        "--run_id",
        run_id,
        "--timeout",
        str(timeout),
        "--namespace",
        namespace,
        "--report_dir",
        str(output_dir),
    ]
    emit_progress(
        f"[{instance_id}] launching swebench harness run_id={run_id}",
        component="swerebench-eval",
    )
    result = _run_command_streaming(command, cwd=output_dir, env=env, component="swerebench-eval")
    stdout_path.write_text(result.stdout or "", encoding="utf-8")
    stderr_path.write_text(result.stderr or "", encoding="utf-8")

    model_dir = _model_label(model_name)
    report_path = output_dir / "logs" / "run_evaluation" / run_id / model_dir / instance_id / "report.json"
    test_output_path = output_dir / "logs" / "run_evaluation" / run_id / model_dir / instance_id / "test_output.txt"
    results_path = output_dir / "evaluation_results" / run_id / "results.json"
    instance_results_path = output_dir / "evaluation_results" / run_id / "instance_results.jsonl"

    payload: dict[str, Any] = {
        "status": "completed" if result.returncode == 0 else "error",
        "dataset_name": dataset_name_for_harness,
        "requested_dataset_name": benchmark_name,
        "split": split,
        "run_id": run_id,
        "work_dir": str(output_dir),
        "predictions_path": str(predictions_path),
        "local_dataset_path": str(local_dataset_path) if local_dataset_path is not None else "",
        "stdout_log": str(stdout_path),
        "stderr_log": str(stderr_path),
        "report_path": str(report_path),
        "test_output_path": str(test_output_path),
        "results_path": str(results_path),
        "instance_results_path": str(instance_results_path),
        "command": command,
        "exit_code": result.returncode,
        "namespace": namespace,
        "cache_level": cache_level,
        "max_workers": max_workers,
        "timeout": timeout,
        "harness_root": str(harness_root) if harness_root is not None else "",
        "harness_python": harness_python,
    }

    if report_path.is_file():
        report_payload = json.loads(report_path.read_text(encoding="utf-8"))
        instance_report = report_payload.get(instance_id) if isinstance(report_payload, dict) else None
        if isinstance(instance_report, dict):
            payload["report"] = instance_report
            payload["summary"] = _report_summary(instance_report)
            payload["resolved"] = bool(instance_report.get("resolved", False))

    if results_path.is_file():
        payload["results"] = json.loads(results_path.read_text(encoding="utf-8"))

    if result.returncode != 0:
        payload["error"] = (result.stderr or result.stdout or "swebench harness failed").strip()
        emit_progress(
            f"[{instance_id}] swebench harness failed with exit code {result.returncode}",
            component="swerebench-eval",
        )
    else:
        emit_progress(
            f"[{instance_id}] swebench harness completed successfully",
            component="swerebench-eval",
        )
    return payload
