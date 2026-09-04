"""Grading for SOMA task instances, replacing the SWE-bench harness for these task lists.

`swerebench_eval.maybe_run_swerebench_evaluation` shells out to `swebench.harness.run_evaluation`,
which resolves a test environment from the dataset row's `install_config`/`version` plus SWE-bench's
own per-repo spec maps. SOMA task rows carry none of that, and none of their repos appear in those
maps - but each row ships a `test` image that already has the repo at `base_commit` with the test
patch applied and a `run_tests` entrypoint that reproduces the exact command validation used.

So grading here is: start the test image, apply the agent's patch to the repo, run the task's own
`run_tests` script, read back the pytest JSON report it writes, and check the FAIL_TO_PASS /
PASS_TO_PASS node ids against it. The returned payload deliberately mirrors the shape
`swerebench_eval` produces (`resolved` plus a `summary` with `fail_to_pass`/`pass_to_pass` buckets)
so `eval_infer.summarize_results` aggregates both kinds of run without knowing the difference.

A graded test id that the report does not mention at all is counted as a failure and listed under
`missing_tests`: the run_tests script only executes the task's own test selection, so a silently
absent id means the selection and the graded id list disagree - which must not read as a pass.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import uuid
from pathlib import Path
from typing import Any, Iterable, Mapping

from .progress import emit_progress
from .registry_auth import docker_env_for_image
from .soma_tasks import (
    DEFAULT_TASK_WORKDIR,
    ROLE_TEST,
    is_soma_task,
    task_image,
    task_image_workdir,
    task_report_path,
    task_run_tests_command,
)

DEFAULT_TEST_TIMEOUT_SECONDS = 1_800
DEFAULT_EVAL_NETWORK = "none"
CONTAINER_NAME_PREFIX = "soma-task-eval-"

# Only an outright pass counts. pytest reports setup failures as "error" and deselected or
# environment-gated tests as "skipped"; neither demonstrates the behaviour a graded id asserts.
PASSING_OUTCOME = "passed"


def _run_command(
    args: list[str],
    *,
    env: dict[str, str] | None = None,
    timeout: int | None = None,
) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            args,
            text=True,
            capture_output=True,
            check=False,
            env=env,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        return subprocess.CompletedProcess(
            args=args,
            returncode=124,
            stdout=exc.stdout if isinstance(exc.stdout, str) else "",
            stderr=f"timed out after {timeout} seconds",
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


def _resolve_option(runtime_options: Mapping[str, Any], name: str, env_name: str) -> Any:
    if name in runtime_options:
        return runtime_options[name]
    return os.getenv(env_name)


def _normalize_test_ids(values: Iterable[Any]) -> list[str]:
    normalized: list[str] = []
    for value in values:
        if not isinstance(value, str):
            continue
        candidate = value.strip()
        if candidate.startswith("./"):
            candidate = candidate[2:]
        if candidate:
            normalized.append(candidate)
    return normalized


def _outcomes_from_report(report_payload: Any) -> dict[str, str]:
    outcomes: dict[str, str] = {}
    if not isinstance(report_payload, dict):
        return outcomes
    tests = report_payload.get("tests")
    if not isinstance(tests, list):
        return outcomes
    for entry in tests:
        if not isinstance(entry, dict):
            continue
        node_id = entry.get("nodeid")
        outcome = entry.get("outcome")
        if isinstance(node_id, str) and isinstance(outcome, str):
            node_id = node_id[2:] if node_id.startswith("./") else node_id
            outcomes[node_id] = outcome
    return outcomes


def _bucket(test_ids: list[str], outcomes: Mapping[str, str]) -> tuple[dict[str, Any], list[str]]:
    successful: list[str] = []
    failed: list[str] = []
    missing: list[str] = []
    for test_id in test_ids:
        outcome = outcomes.get(test_id)
        if outcome is None:
            missing.append(test_id)
            failed.append(test_id)
        elif outcome == PASSING_OUTCOME:
            successful.append(test_id)
        else:
            failed.append(test_id)
    bucket = {
        "success": len(successful),
        "failure": len(failed),
        "total": len(test_ids),
        "successful_tests": successful,
        "failed_tests": failed,
    }
    return bucket, missing


def _apply_patch_in_container(
    *,
    container_id: str,
    workdir: str,
    container_patch_path: str,
) -> tuple[bool, str]:
    """Apply the agent patch, falling back through progressively looser appliers.

    `git apply` is tried first because it is the strictest and keeps rename/mode information;
    `--3way` recovers patches whose context drifted, and `patch -p1` handles diffs git rejects
    outright. The test image's repo is at the same `base_commit` the patch was captured against,
    so the first attempt is expected to win - the fallbacks exist so a formatting quirk in one
    agent's diff does not silently score as an empty patch.
    """
    attempts = (
        ["git", "apply", "--verbose", "--whitespace=nowarn", container_patch_path],
        ["git", "apply", "--3way", "--whitespace=nowarn", container_patch_path],
        ["patch", "--batch", "--fuzz=5", "-p1", "-i", container_patch_path],
    )
    failures: list[str] = []
    for attempt in attempts:
        command = " ".join(attempt)
        result = _run_command([
            "docker", "exec", "-w", workdir, container_id,
            "sh", "-lc", f"git config --global --add safe.directory {workdir} >/dev/null 2>&1; {command}",
        ])
        if result.returncode == 0:
            return True, command
        failures.append(f"$ {command}\n{(result.stderr or result.stdout or '').strip()}")
    return False, "\n\n".join(failures)


def maybe_run_soma_task_evaluation(
    *,
    instance_id: str,
    hidden_eval: Mapping[str, Any],
    fail_to_pass: Iterable[Any],
    pass_to_pass: Iterable[Any],
    patch_capture: Mapping[str, Any],
    output_dir: Path,
    runtime_options: Mapping[str, Any],
) -> dict[str, Any]:
    output_dir = output_dir.expanduser().resolve()

    enabled = _coerce_bool(_resolve_option(runtime_options, "swerebench_eval_enabled", "SOMA_SWEREBENCH_EVAL"))
    if enabled is not True:
        return {"status": "disabled", "reason": "patch evaluation is disabled"}

    if not is_soma_task(hidden_eval):
        return {
            "status": "unavailable",
            "reason": "instance carries no SOMA test image (images.test.ref)",
        }

    test_image = task_image(hidden_eval, ROLE_TEST) or ""
    workdir = task_image_workdir(hidden_eval, ROLE_TEST) or DEFAULT_TASK_WORKDIR
    run_tests = task_run_tests_command(hidden_eval)
    report_in_container = task_report_path(hidden_eval)

    base_payload: dict[str, Any] = {
        "evaluator": "soma-task",
        "test_image": test_image,
        "workdir": workdir,
        "run_tests": run_tests,
        "work_dir": str(output_dir),
    }

    if not patch_capture.get("has_changes"):
        return {
            **base_payload,
            "status": "skipped-empty-patch",
            "reason": "agent patch is empty",
            "patch_path": patch_capture.get("patch_path"),
        }

    patch_path = Path(str(patch_capture.get("patch_path", ""))).expanduser()
    if not patch_path.is_file():
        return {
            **base_payload,
            "status": "error",
            "error": f"agent patch file is missing: {patch_path}",
        }

    fail_to_pass_ids = _normalize_test_ids(fail_to_pass)
    pass_to_pass_ids = _normalize_test_ids(pass_to_pass)

    timeout = _coerce_positive_int(
        _resolve_option(runtime_options, "swerebench_timeout", "SOMA_SWEREBENCH_TIMEOUT"),
        DEFAULT_TEST_TIMEOUT_SECONDS,
    )
    network = str(
        _resolve_option(runtime_options, "soma_task_eval_network", "SOMA_TASK_EVAL_NETWORK")
        or DEFAULT_EVAL_NETWORK
    ).strip() or DEFAULT_EVAL_NETWORK

    output_dir.mkdir(parents=True, exist_ok=True)
    docker_env = docker_env_for_image(test_image)

    emit_progress(
        f"[{instance_id}] grading against SOMA test image {test_image}",
        component="soma-task-eval",
    )

    inspect_result = _run_command(["docker", "image", "inspect", test_image], env=docker_env)
    if inspect_result.returncode != 0:
        pull_result = _run_command(["docker", "pull", test_image], env=docker_env)
        if pull_result.returncode != 0:
            return {
                **base_payload,
                "status": "error",
                "error": (
                    f"could not pull SOMA test image {test_image}: "
                    f"{(pull_result.stderr or pull_result.stdout or '').strip()}"
                ),
            }

    container_name = f"{CONTAINER_NAME_PREFIX}{_slug(instance_id, default='instance')}-{uuid.uuid4().hex[:8]}"
    container_patch_path = "/tmp/soma-agent.patch"
    report_path = output_dir / "report.json"
    test_log_path = output_dir / "run-tests.log"

    payload: dict[str, Any] = {
        **base_payload,
        "container_name": container_name,
        "network": network,
        "timeout": timeout,
        "patch_path": str(patch_path),
        "report_path": str(report_path),
        "test_log_path": str(test_log_path),
        "fail_to_pass_total": len(fail_to_pass_ids),
        "pass_to_pass_total": len(pass_to_pass_ids),
    }

    # The graded container never needs the network (the image already has every dependency
    # installed) and must not be able to reach the run's LLM proxy, so it is started detached
    # on `none` and kept alive by a sleep while patch/test/report steps exec into it.
    run_result = _run_command(
        [
            "docker", "run", "-d",
            "--name", container_name,
            "--network", network,
            "--entrypoint", "sh",
            test_image,
            "-lc", f"sleep {timeout + 120}",
        ],
        env=docker_env,
    )
    if run_result.returncode != 0:
        return {
            **payload,
            "status": "error",
            "error": (
                "could not start SOMA test container: "
                f"{(run_result.stderr or run_result.stdout or '').strip()}"
            ),
        }
    container_id = (run_result.stdout or "").strip() or container_name

    try:
        copy_result = _run_command(
            ["docker", "cp", str(patch_path), f"{container_id}:{container_patch_path}"]
        )
        if copy_result.returncode != 0:
            return {
                **payload,
                "status": "error",
                "error": (
                    "could not copy the agent patch into the test container: "
                    f"{(copy_result.stderr or copy_result.stdout or '').strip()}"
                ),
            }

        applied, apply_detail = _apply_patch_in_container(
            container_id=container_id,
            workdir=workdir,
            container_patch_path=container_patch_path,
        )
        payload["patch_apply_detail"] = apply_detail
        if not applied:
            emit_progress(
                f"[{instance_id}] agent patch did not apply to the test image",
                component="soma-task-eval",
            )
            return {
                **payload,
                "status": "completed",
                "resolved": False,
                "summary": {
                    "resolved": False,
                    "patch_exists": True,
                    "patch_successfully_applied": False,
                    "fail_to_pass": _bucket(fail_to_pass_ids, {})[0],
                    "pass_to_pass": _bucket(pass_to_pass_ids, {})[0],
                },
            }

        emit_progress(
            f"[{instance_id}] running graded tests ({run_tests})",
            component="soma-task-eval",
        )
        test_result = _run_command(
            ["docker", "exec", "-w", workdir, container_id, "sh", "-lc", run_tests],
            timeout=timeout,
        )
        test_log_path.write_text(
            (test_result.stdout or "") + (test_result.stderr or ""),
            encoding="utf-8",
        )
        payload["exit_code"] = test_result.returncode

        report_copy = _run_command(
            ["docker", "cp", f"{container_id}:{report_in_container}", str(report_path)]
        )
        if report_copy.returncode != 0:
            return {
                **payload,
                "status": "error",
                "patch_successfully_applied": True,
                "error": (
                    f"graded test run produced no report at {report_in_container} "
                    f"(exit code {test_result.returncode}); see {test_log_path}"
                ),
            }
    finally:
        _run_command(["docker", "rm", "-f", "-v", container_id])

    try:
        report_payload = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {
            **payload,
            "status": "error",
            "error": f"could not read the graded test report: {exc}",
        }

    outcomes = _outcomes_from_report(report_payload)
    fail_bucket, fail_missing = _bucket(fail_to_pass_ids, outcomes)
    pass_bucket, pass_missing = _bucket(pass_to_pass_ids, outcomes)
    missing = fail_missing + pass_missing
    resolved = fail_bucket["failure"] == 0 and pass_bucket["failure"] == 0

    if missing:
        emit_progress(
            f"[{instance_id}] {len(missing)} graded test id(s) absent from the report; "
            "counted as failures",
            component="soma-task-eval",
        )

    payload.update(
        {
            "status": "completed",
            "resolved": resolved,
            "missing_tests": missing,
            "reported_test_count": len(outcomes),
            "summary": {
                "resolved": resolved,
                "patch_exists": True,
                "patch_successfully_applied": True,
                "fail_to_pass": fail_bucket,
                "pass_to_pass": pass_bucket,
            },
        }
    )
    emit_progress(
        f"[{instance_id}] SOMA grading resolved={resolved} "
        f"F2P={fail_bucket['success']}/{fail_bucket['total']} "
        f"P2P={pass_bucket['success']}/{pass_bucket['total']}",
        component="soma-task-eval",
    )
    return payload
