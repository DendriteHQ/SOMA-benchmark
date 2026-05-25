from __future__ import annotations

import argparse
import json

from soma_bench import __version__


def main() -> int:
    parser = argparse.ArgumentParser(prog="soma-bench")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("info", help="Print the current package summary.")
    subparsers.add_parser("benchmark-info", help="Print the benchmark runtime summary.")

    benchmark_build_images = subparsers.add_parser(
        "benchmark-build-images",
        help="Run the benchmark image planning command.",
        add_help=False,
    )

    benchmark_run_infer = subparsers.add_parser(
        "benchmark-run-infer",
        help="Run the benchmark inference planning command.",
        add_help=False,
    )

    benchmark_solve = subparsers.add_parser(
        "benchmark-solve",
        help="Run one benchmark instance directly from its problem statement.",
        add_help=False,
    )

    benchmark_eval = subparsers.add_parser(
        "benchmark-eval",
        help="Run the benchmark evaluation summary command.",
        add_help=False,
    )

    args, extras = parser.parse_known_args()
    if args.command == "info":
        if extras:
            parser.error(f"unrecognized arguments: {' '.join(extras)}")
        return _run_info()
    if args.command == "benchmark-info":
        if extras:
            parser.error(f"unrecognized arguments: {' '.join(extras)}")
        return _run_benchmark_info()
    if args.command == "benchmark-build-images":
        from soma_bench.benchmark.build_images import main as benchmark_build_images_main

        return benchmark_build_images_main(extras)
    if args.command == "benchmark-run-infer":
        from soma_bench.benchmark.run_infer import main as benchmark_run_infer_main

        return benchmark_run_infer_main(extras)
    if args.command == "benchmark-solve":
        from soma_bench.benchmark.solve import main as benchmark_solve_main

        return benchmark_solve_main(extras)
    if args.command == "benchmark-eval":
        from soma_bench.benchmark.eval_infer import main as benchmark_eval_main

        return benchmark_eval_main(extras)
    raise ValueError(f"Unsupported command: {args.command}")


def _run_info() -> int:
    payload = {
        "package": "soma-bench",
        "version": __version__,
        "components": [
            "benchmark.config",
            "benchmark.manifest",
            "benchmark.solve",
            "benchmark.run_infer",
            "benchmark.eval_infer",
        ],
    }
    print(json.dumps(payload, indent=2))
    return 0


def _run_benchmark_info() -> int:
    from soma_bench.benchmark import list_runtime_backends
    from soma_bench.benchmark.config import DEFAULTS, SUPPORTED_WORKSPACES

    payload = {
        "package": "soma-bench",
        "version": __version__,
        "components": [
            "benchmark.config",
            "benchmark.manifest",
            "benchmark.solve",
            "benchmark.run_infer",
            "benchmark.eval_infer",
        ],
        "runtime_backends": list(list_runtime_backends()),
        "workspace_types": list(SUPPORTED_WORKSPACES),
        "defaults": DEFAULTS.to_dict(),
        "execution_state": "scaffolded",
    }
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())