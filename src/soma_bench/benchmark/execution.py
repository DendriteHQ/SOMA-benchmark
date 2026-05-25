from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed

from .backends.base import RuntimeBackend, RuntimeExecutionContext, RuntimeExecutionResult


def runtime_error_result(
    context: RuntimeExecutionContext,
    exc: Exception,
) -> RuntimeExecutionResult:
    return RuntimeExecutionResult(
        benchmark_id=context.instance.benchmark_id,
        instance_id=context.instance.instance_id,
        backend_name=context.backend_name,
        status="runtime-error",
        error=str(exc),
        metadata={
            "error_type": type(exc).__name__,
        },
    )


def _execute_one(
    runtime_backend: RuntimeBackend,
    context: RuntimeExecutionContext,
) -> RuntimeExecutionResult:
    try:
        return runtime_backend.execute(context)
    except Exception as exc:
        return runtime_error_result(context, exc)


def execute_runtime_contexts(
    runtime_backend: RuntimeBackend,
    contexts: list[RuntimeExecutionContext],
    *,
    concurrency: int,
) -> list[RuntimeExecutionResult]:
    if not contexts:
        return []

    worker_count = max(1, min(concurrency, len(contexts)))
    if runtime_backend.execute_batch is not None:
        try:
            return runtime_backend.execute_batch(contexts, worker_count)
        except Exception as exc:
            return [runtime_error_result(context, exc) for context in contexts]

    if worker_count == 1:
        return [_execute_one(runtime_backend, context) for context in contexts]

    results: list[RuntimeExecutionResult | None] = [None] * len(contexts)
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        future_to_index = {
            executor.submit(_execute_one, runtime_backend, context): index
            for index, context in enumerate(contexts)
        }
        for future in as_completed(future_to_index):
            results[future_to_index[future]] = future.result()

    return [result for result in results if result is not None]
