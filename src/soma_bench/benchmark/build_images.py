from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .backends import get_runtime_backend, list_runtime_backends
from .config import DEFAULTS, build_agent_image_ref
from .manifest import BenchmarkInstance, load_manifest, load_selected_instance_ids


def build_image_plan(
    instances: list[BenchmarkInstance],
    image: str,
    target: str,
) -> list[dict[str, Any]]:
    deduped_plan: dict[str, dict[str, Any]] = {}
    for instance in instances:
        if instance.instance_id not in deduped_plan:
            deduped_plan[instance.instance_id] = {
                "instance_id": instance.instance_id,
                "repo": instance.repo,
                "agent_image": build_agent_image_ref(instance.instance_id, image=image, target=target),
                "base_commit": instance.base_commit,
                "environment_setup_commit": instance.environment_setup_commit,
                "benchmark_ids": [],
            }
        deduped_plan[instance.instance_id]["benchmark_ids"].append(instance.benchmark_id)
    return list(deduped_plan.values())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m soma_bench.benchmark.build_images")
    parser.add_argument("--dataset", required=True, help="Path to the SOMA benchmark manifest JSONL.")
    parser.add_argument("--split", default="test", help="Logical split label for reporting.")
    parser.add_argument(
        "--runtime-backend",
        choices=list_runtime_backends(),
        default=DEFAULTS.runtime_backend,
        help="Runtime backend whose image defaults should be used.",
    )
    parser.add_argument("--image", default=None, help="Target agent image repository.")
    parser.add_argument("--target", default=None, help="Build target suffix.")
    parser.add_argument("--n-limit", type=int, default=0, help="Limit the number of planned instances.")
    parser.add_argument("--select", default=None, help="Optional file with one instance_id per line.")
    parser.add_argument("--output-file", default=None, help="Optional path to write the image plan JSON.")
    args = parser.parse_args(argv)

    runtime_backend = get_runtime_backend(args.runtime_backend)
    image = args.image or runtime_backend.default_image
    target = args.target or runtime_backend.default_build_target
    selected_ids = load_selected_instance_ids(args.select) if args.select else None
    instances = load_manifest(args.dataset, limit=args.n_limit, selected_instance_ids=selected_ids)
    plan = build_image_plan(instances, image=image, target=target)
    payload = {
        "dataset": str(Path(args.dataset)),
        "split": args.split,
        "runtime_backend": runtime_backend.name,
        "instance_count": len(plan),
        "image": image,
        "target": target,
        "planned_images": plan,
    }
    if args.output_file:
        output_path = Path(args.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
