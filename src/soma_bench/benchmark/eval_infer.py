from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def _load_rows(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    input_path = Path(path)
    if not input_path.exists():
        raise FileNotFoundError(f"Result file does not exist: {input_path}")
    with input_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            raw_line = line.strip()
            if not raw_line:
                continue
            try:
                rows.append(json.loads(raw_line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number}: {exc.msg}") from exc
    return rows


def summarize_results(rows: list[dict[str, Any]]) -> dict[str, Any]:
    status_counts: Counter[str] = Counter()
    patch_eval_status_counts: Counter[str] = Counter()
    patch_rows_with_changes = 0
    patch_rows_without_changes = 0
    patch_resolved_count = 0
    fail_to_pass_success = 0
    fail_to_pass_total = 0
    pass_to_pass_success = 0
    pass_to_pass_total = 0
    for row in rows:
        status_counts[str(row.get("status", "unknown"))] += 1
        metadata = row.get("metadata")
        if not isinstance(metadata, dict):
            continue
        patch_capture = metadata.get("patch_capture")
        if isinstance(patch_capture, dict):
            if patch_capture.get("has_changes"):
                patch_rows_with_changes += 1
            else:
                patch_rows_without_changes += 1
        patch_evaluation = metadata.get("patch_evaluation")
        if not isinstance(patch_evaluation, dict):
            continue
        patch_eval_status_counts[str(patch_evaluation.get("status", "unknown"))] += 1
        if patch_evaluation.get("resolved") is True:
            patch_resolved_count += 1
        summary = patch_evaluation.get("summary")
        if not isinstance(summary, dict):
            continue
        fail_bucket = summary.get("fail_to_pass")
        if isinstance(fail_bucket, dict):
            fail_to_pass_success += int(fail_bucket.get("success", 0) or 0)
            fail_to_pass_total += int(fail_bucket.get("total", 0) or 0)
        pass_bucket = summary.get("pass_to_pass")
        if isinstance(pass_bucket, dict):
            pass_to_pass_success += int(pass_bucket.get("success", 0) or 0)
            pass_to_pass_total += int(pass_bucket.get("total", 0) or 0)
    return {
        "row_count": len(rows),
        "status_counts": dict(status_counts),
        "patch_capture": {
            "rows_with_changes": patch_rows_with_changes,
            "rows_without_changes": patch_rows_without_changes,
        },
        "patch_evaluation": {
            "status_counts": dict(patch_eval_status_counts),
            "resolved_count": patch_resolved_count,
            "fail_to_pass_success": fail_to_pass_success,
            "fail_to_pass_total": fail_to_pass_total,
            "pass_to_pass_success": pass_to_pass_success,
            "pass_to_pass_total": pass_to_pass_total,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m soma_bench.benchmark.eval_infer")
    parser.add_argument("output_json", help="Path to benchmark output.jsonl.")
    parser.add_argument("--output-file", default=None, help="Optional path to write the evaluation summary JSON.")
    args = parser.parse_args(argv)

    summary = summarize_results(_load_rows(args.output_json))
    summary["output_json"] = str(Path(args.output_json))
    if args.output_file:
        output_path = Path(args.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())