"""Support for SOMA task lists (`tasks-*.jsonl`) as a benchmark source.

These task rows are not a Hugging Face dataset: they ship as a plain JSONL file and carry
their own pre-built Docker images (one `env` image the agent works in, one `test` image the
run is graded on) instead of relying on SWE-bench's image-name conventions and its harness
spec maps - none of the repos in these lists are covered by those maps.

Rather than teaching every entry point a second dataset-loading path, `materialize_task_cache`
writes the JSONL into exactly the on-disk row cache `runner_settings` already prefers over the
Hugging Face datasets-server (`<benchmark_cache_root()>/<slug>/...`). Once materialized, the
normal `benchmark-solve` / `benchmark-run-infer` flows resolve these instances offline with no
further changes, and `--benchmark <name>` selects the task list by the name it was cached under.

The accessors below are the single place that knows the row's `images` layout, so backends and
the evaluator never index into that structure by hand.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any, Mapping

from .cache_paths import benchmark_cache_root

ROLE_ENV = "env"
ROLE_TEST = "test"

DEFAULT_BENCHMARK_NAME = "soma-is-tasks"
DEFAULT_DATASET_CONFIG = "default"
DEFAULT_SPLIT = "test"

# Fallbacks matching what the soma-7 image builder bakes in; every row seen so far carries
# these explicitly, so they only ever apply to a row that omits the field entirely.
DEFAULT_TASK_WORKDIR = "/repo"
DEFAULT_TASK_RUN_TESTS = "/soma/run_tests.sh"
DEFAULT_TASK_REPORT_PATH = "/tmp/report.json"


def _cache_slug(value: str) -> str:
    # Mirrors runner_settings.py's private _slug() exactly, so this writes to the same
    # on-disk path the normal benchmark-solve/run-infer flow reads back.
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-._")
    return normalized or "default"


def _image_entry(hidden_eval: Mapping[str, Any], role: str) -> Mapping[str, Any] | None:
    images = hidden_eval.get("images")
    if not isinstance(images, Mapping):
        return None
    entry = images.get(role)
    return entry if isinstance(entry, Mapping) else None


def _entry_string(entry: Mapping[str, Any] | None, key: str) -> str | None:
    if entry is None:
        return None
    value = entry.get(key)
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def is_soma_task(hidden_eval: Mapping[str, Any]) -> bool:
    """True when this instance carries SOMA task images (so SOMA grading applies)."""
    return task_image(hidden_eval, ROLE_TEST) is not None


def task_image(hidden_eval: Mapping[str, Any], role: str) -> str | None:
    return _entry_string(_image_entry(hidden_eval, role), "ref")


def task_image_workdir(hidden_eval: Mapping[str, Any], role: str) -> str | None:
    return _entry_string(_image_entry(hidden_eval, role), "workdir")


def task_run_tests_command(hidden_eval: Mapping[str, Any]) -> str:
    return _entry_string(_image_entry(hidden_eval, ROLE_TEST), "run_tests") or DEFAULT_TASK_RUN_TESTS


def task_report_path(hidden_eval: Mapping[str, Any]) -> str:
    """Where the graded test run leaves its pytest JSON report inside the test image.

    The path is an argument of the row's own `test_command` (every task list seen so far
    passes `--json-report-file=...`), so it is read back from there rather than assumed.
    """
    test_command = hidden_eval.get("test_command")
    if isinstance(test_command, str):
        match = re.search(r"--json-report-file[=\s]+(\S+)", test_command)
        if match:
            return match.group(1)
    return DEFAULT_TASK_REPORT_PATH


def dataset_cache_paths(
    *,
    benchmark_name: str,
    dataset_config: str = DEFAULT_DATASET_CONFIG,
    split: str = DEFAULT_SPLIT,
) -> dict[str, Path]:
    dataset_dir = benchmark_cache_root() / _cache_slug(benchmark_name)
    split_dir = dataset_dir / "splits" / _cache_slug(dataset_config) / _cache_slug(split)
    return {
        "dataset_dir": dataset_dir,
        "dataset_info": dataset_dir / "dataset-info.json",
        "split_dir": split_dir,
        "rows": split_dir / "rows.jsonl",
        "meta": split_dir / "meta.json",
    }


def load_task_rows(tasks_path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(tasks_path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            raw_line = line.strip()
            if not raw_line:
                continue
            try:
                payload = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on task line {line_number}: {exc.msg}") from exc
            if not isinstance(payload, dict):
                raise ValueError(f"Task line {line_number} is not a JSON object")
            instance_id = payload.get("instance_id")
            if not isinstance(instance_id, str) or not instance_id.strip():
                raise ValueError(f"Task line {line_number} is missing instance_id")
            rows.append(payload)
    if not rows:
        raise ValueError(f"No task rows found in {tasks_path}")
    return rows


def materialize_task_cache(
    *,
    tasks_path: str | Path,
    benchmark_name: str = DEFAULT_BENCHMARK_NAME,
    dataset_config: str = DEFAULT_DATASET_CONFIG,
    split: str = DEFAULT_SPLIT,
) -> dict[str, Any]:
    """Write a task JSONL into the benchmark row cache the runner already reads.

    `runner_settings` treats a split cache as authoritative only when its meta marks it
    complete and its row count matches the count advertised by the dataset info, so both
    files are written from the same in-memory row list to keep them consistent.
    """
    rows = load_task_rows(tasks_path)
    paths = dataset_cache_paths(
        benchmark_name=benchmark_name,
        dataset_config=dataset_config,
        split=split,
    )
    paths["split_dir"].mkdir(parents=True, exist_ok=True)

    with paths["rows"].open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps({"row": row}, ensure_ascii=True) + "\n")

    dataset_info = {
        "dataset_info": {
            dataset_config: {
                # `features` is only consulted to confirm the lookup field exists; the
                # runner never reads the declared dtype.
                "features": {key: {"dtype": "string"} for key in sorted(rows[0])},
                "splits": {split: {"name": split, "num_examples": len(rows)}},
            }
        }
    }
    paths["dataset_info"].write_text(json.dumps(dataset_info, indent=2) + "\n", encoding="utf-8")

    meta = {
        "benchmark_name": benchmark_name,
        "dataset_config": dataset_config,
        "split": split,
        "row_count": len(rows),
        "expected_row_count": len(rows),
        "complete": True,
        "cached_at": int(time.time()),
        "source": str(Path(tasks_path).resolve()),
    }
    paths["meta"].write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    return {
        "benchmark_name": benchmark_name,
        "dataset_config": dataset_config,
        "split": split,
        "row_count": len(rows),
        "instance_ids": [str(row["instance_id"]) for row in rows],
        "rows_path": str(paths["rows"]),
        "dataset_info_path": str(paths["dataset_info"]),
        "meta_path": str(paths["meta"]),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m soma_bench.benchmark.soma_tasks")
    parser.add_argument("--tasks", required=True, help="Path to a SOMA task JSONL file.")
    parser.add_argument(
        "--benchmark",
        default=DEFAULT_BENCHMARK_NAME,
        help=(
            "Name to cache the task list under. Pass the same value to --benchmark on "
            "benchmark-solve / benchmark-run-infer."
        ),
    )
    parser.add_argument("--dataset-config", default=DEFAULT_DATASET_CONFIG)
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument(
        "--print-instance-ids",
        action="store_true",
        help="Also print every cached instance_id, one per line, to stderr.",
    )
    args = parser.parse_args(argv)

    summary = materialize_task_cache(
        tasks_path=args.tasks,
        benchmark_name=args.benchmark,
        dataset_config=args.dataset_config,
        split=args.split,
    )
    instance_ids = summary.pop("instance_ids")
    print(json.dumps(summary, indent=2))
    if args.print_instance_ids:
        import sys

        for instance_id in instance_ids:
            print(instance_id, file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
